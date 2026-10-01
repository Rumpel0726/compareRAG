# -*- coding: utf-8 -*-
"""
Скрипт оценки RAG системы.

Запуск из корня проекта:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --config configs/strategies/large_chunks.yaml --top-k 5

Результаты записываются в experiments/{strategy}/{model}/{chunk_size}/full/ и results/
"""

import sys
import copy
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

# Корень проекта (scripts/ -> compareRAG/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.embeddings.factory import create_embedder_from_config
from src.vector_stores.factory import create_vector_store_from_config
from src.retrieval.factory import create_retriever_from_config
from src.retrieval.reranker import Reranker, reranker_short_name
from src.llm.lm_studio_client import LMStudioClient
from src.agents.orchestrator import SimpleRAGOrchestrator
from src.evaluation.questions import EVAL_QUESTIONS as EVAL_QUESTIONS_MEDICAL
from src.evaluation.questions_civil_code import EVAL_QUESTIONS_CIVIL_CODE
from src.evaluation.questions_postgresql import EVAL_QUESTIONS_POSTGRESQL
from src.evaluation.retrieval_metrics import compute_all
from src.evaluation.generation_metrics import LLMJudge
from src.utils.logging_config import setup_logging

DOMAIN_QUESTIONS = {
    "medical":    EVAL_QUESTIONS_MEDICAL,
    "civil_code": EVAL_QUESTIONS_CIVIL_CODE,
    "postgresql": EVAL_QUESTIONS_POSTGRESQL,
}


def get_embedding_folder_and_abbrev(embedding_model: str) -> tuple[str, str]:
    """Returns (folder_name, file_abbrev) for the embedding model.

    Examples:
        text-embedding-bge-m3              → ("bge-m3",    "bge")
        text-embedding-qwen3-embedding-0.6b → ("qwen3-06b", "qwen3")
        text-embedding-nomic-embed-text-v1.5 → ("nomic",    "nomic")
    """
    name = embedding_model.lower()
    if name.startswith("text-embedding-"):
        name = name[len("text-embedding-"):]
    if "/" in name:
        name = name.split("/")[-1]
    parts = name.split("-")
    base = parts[0]

    if base == "bge":
        folder = "-".join(parts[:2]) if len(parts) > 1 else base
        abbrev = base
    elif base.startswith("qwen"):
        # qwen3-embedding-0.6b → qwen3-06b
        version = base  # e.g. qwen3
        version_tag = ""
        for p in parts[2:]:  # skip the 'embedding' token
            if any(c.isdigit() for c in p):
                v = p.replace(".", "")
                if len(v) <= 2 and v[0].isdigit():
                    v = "0" + v
                version_tag = v
                break
        folder = f"{version}-{version_tag}" if version_tag else version
        abbrev = version
    elif base == "nomic":
        folder = "nomic"
        abbrev = "nomic"
    elif "e5" in parts:
        # multilingual-e5-large-instruct → folder "e5-large", abbrev "e5"
        e5_idx = parts.index("e5")
        size = parts[e5_idx + 1] if e5_idx + 1 < len(parts) else ""
        folder = f"e5-{size}" if size and size != "instruct" else "e5"
        abbrev = "e5"
    else:
        folder = base
        abbrev = base

    return folder, abbrev


def get_next_log_number(directory: Path, prefix: str, date_str: str) -> int:
    """Find the next available log file number for a given prefix and date."""
    if not directory.exists():
        return 1
    existing = list(directory.glob(f"{prefix}_{date_str}_*.log"))
    numbers = []
    for f in existing:
        stem_parts = f.stem.rsplit("_", 1)
        if len(stem_parts) == 2 and stem_parts[1].isdigit():
            numbers.append(int(stem_parts[1]))
    return max(numbers, default=0) + 1


def build_experiment_log_paths(project_root: Path, config, domain: str = "medical",
                                reranker_model: str = None) -> tuple[Path, Path]:
    """Build full and results log file paths for the current experiment run."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    retrieval = config.retrieval.strategy          # semantic | hybrid
    chunk_strategy = config.chunking.strategy      # chonkie | sentence | langchain
    chunk_size = str(config.chunking.chunk_size)   # 256, 512, 1024
    emb_folder, emb_abbrev = get_embedding_folder_and_abbrev(config.lm_studio.embedding_model)

    base_dir = project_root / "experiments" / domain / retrieval / emb_folder / chunk_strategy / chunk_size
    # При активном reranker результаты пишутся в подпапку rerank_<short>/
    if reranker_model:
        base_dir = base_dir / f"rerank_{reranker_short_name(reranker_model)}"
    full_dir = base_dir / "full"
    results_dir = base_dir / "results"

    file_prefix = f"{retrieval}_{chunk_strategy}_{chunk_size}_{emb_abbrev}"
    if reranker_model:
        file_prefix = f"{file_prefix}_rerank_{reranker_short_name(reranker_model)}"
    num = max(
        get_next_log_number(full_dir, f"{file_prefix}_full", date_str),
        get_next_log_number(results_dir, f"{file_prefix}_res", date_str),
    )

    full_path = full_dir / f"{file_prefix}_full_{date_str}_{num:02d}.log"
    results_path = results_dir / f"{file_prefix}_res_{date_str}_{num:02d}.log"
    return full_path, results_path


def setup_experiment_logging(full_path: Path, results_path: Path) -> None:
    """Creates experiment directory structure and adds two Loguru sinks."""
    full_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = "{time:YYYY-MM-DD HH:mm:ss} | {message}"

    logger.add(
        str(full_path),
        format=fmt,
        level="DEBUG",
        filter=lambda record: "eval_full" in record["extra"],
        encoding="utf-8",
    )
    logger.add(
        str(results_path),
        format=fmt,
        level="DEBUG",
        filter=lambda record: "eval_results" in record["extra"],
        encoding="utf-8",
    )
    logger.info(f"Full log:    {full_path}")
    logger.info(f"Results log: {results_path}")


def eval_log(msg: str, results: bool = False) -> None:
    """Логирует сообщение в evaluation-лог. results=True также пишет в results-лог."""
    if results:
        logger.bind(eval_full=True, eval_results=True).info(msg)
    else:
        logger.bind(eval_full=True).info(msg)


def vprint(msg: str = "") -> None:
    """Выводит сообщение только в консоль (не в лог-файлы)."""
    print(msg)


def load_ground_truth(gt_path: Path) -> dict[str, str]:
    """Загружает эталонные ответы из JSON-файла.

    Returns:
        Словарь {question_text: reference_answer}.
        Пустой словарь, если файл не найден или содержит ошибки.
    """
    if not gt_path.exists():
        logger.warning(f"Ground truth файл не найден: {gt_path}. Судья будет работать в режиме контекста.")
        return {}
    try:
        with open(gt_path, encoding="utf-8") as f:
            records = json.load(f)
        mapping = {}
        for rec in records:
            q = rec.get("question")
            ref = rec.get("reference_answer")
            if q and ref:
                mapping[q] = ref
        logger.info(f"Ground truth загружен: {len(mapping)} ответов из {gt_path.name}")
        return mapping
    except Exception as e:
        logger.warning(f"Не удалось загрузить ground truth: {e}. Переход в режим контекста.")
        return {}


def initialize_rag(
    config_path: str,
    top_k: int,
    embedding_model: str = None,
    api_key: str = None,
    url: str = None,
    llm_url: str = None,
    embedder_url: str = None,
    llm_model: str = None,
    add_eos_token: bool = False,
    eos_token: str = None,
    timeout: int = None,
    retrieval: str = None,
    reranker_model: str = None,
    rerank_top_n: int = 20,
    args = None
):
    """Инициализация компонентов RAG --- по паттерну web/app.py."""
    logger.info(f"Загрузка конфигурации: {config_path}")
    config = load_config(config_path)

    # Переопределение retrieval strategy (если задана через CLI)
    if retrieval:
        config.retrieval.strategy = retrieval
        logger.info(f"Retrieval strategy (CLI): {retrieval}")

    # Переопределение модели эмбеддингов (если задана через CLI)
    if embedding_model:
        config.lm_studio.embedding_model = embedding_model
        model_id = config.lm_studio.embedding_model_id
        # Убираем недопустимые для ChromaDB символы (слэши и тире могут вызвать проблемы, заменяем на _ и обрезаем, если нужно)
        safe_model_id = model_id.replace("/", "_").replace("-", "_")
        config.chromadb.collection_name = f"{config.chromadb.collection_name}_{safe_model_id}"
        logger.info(f"Модель эмбеддингов (CLI): {embedding_model}")
        logger.info(f"Коллекция (с суффиксом модели): {config.chromadb.collection_name}")

    if add_eos_token:
        config.lm_studio.add_eos_token = True
        logger.info("EOS-токен включён")
    if eos_token:
        config.lm_studio.eos_token = eos_token
        logger.info(f"EOS-токен: {eos_token}")
    if timeout is not None:
        config.lm_studio.timeout = timeout
        logger.info(f"Таймаут LLM (CLI): {timeout}s")
    if api_key:
        config.lm_studio.api_key = api_key
        logger.info("Установлен API ключ из параметров CLI")
    if url:
        config.lm_studio.url = url
        logger.info(f"URL API (общий, CLI): {url}")
    if llm_url:
        config.lm_studio.llm_url = llm_url
        logger.info(f"URL LLM API (CLI): {llm_url}")
    if embedder_url:
        config.lm_studio.embedder_url = embedder_url
        logger.info(f"URL Embedder API (CLI): {embedder_url}")
    if llm_model:
        config.lm_studio.llm_model = llm_model
        logger.info(f"Модель генерации (CLI): {llm_model}")

    logger.info("Инициализация эмбеддера...")
    embedder = create_embedder_from_config(config)

    # Абсолютный путь к ChromaDB относительно PROJECT_ROOT
    chroma_path = Path(config.chromadb.path)
    if not chroma_path.is_absolute():
        chroma_path = PROJECT_ROOT / config.chromadb.path
    config_copy = copy.deepcopy(config)
    config_copy.chromadb.path = str(chroma_path)

    logger.info("Инициализация векторного хранилища...")
    vector_store = create_vector_store_from_config(config_copy)

    stats = vector_store.get_collection_stats()
    logger.info(f"Коллекция: {stats}")
    if stats.get("document_count", 0) == 0:
        logger.error("Векторное хранилище пусто. Запустите: python scripts/ingest_documents.py")
        sys.exit(1)

    reranker = None
    if reranker_model:
        logger.info(f"Инициализация reranker: {reranker_model} (top_n={rerank_top_n})")
        reranker = Reranker(model=reranker_model, top_n=rerank_top_n)

    logger.info("Инициализация ретривера...")
    retriever = create_retriever_from_config(config, embedder, vector_store, reranker=reranker)

    logger.info("Инициализация LLM клиента...")
    llm_client = LMStudioClient(
        url=config.lm_studio.llm_url or config.lm_studio.url,
        model=config.lm_studio.llm_model,
        temperature=config.lm_studio.temperature,
        max_tokens=config.lm_studio.max_tokens,
        timeout=config.lm_studio.timeout,
        api_key=config.lm_studio.api_key,
    )

    logger.info("Инициализация orchestrator...")
    orchestrator = SimpleRAGOrchestrator(
        retriever=retriever,
        llm_client=llm_client,
        top_k=top_k,
    )

    return orchestrator, llm_client, config




def process_question(args: tuple) -> dict:
    """Обрабатывает один вопрос в отдельном потоке. Возвращает словарь с результатами."""
    idx, eq, orchestrator, judge, top_k, verbose = args

    out = {"idx": idx, "eq": eq, "error": None, "ret_metrics": None, "gen_scores": None,
           "result": None, "metric_mode": None, "verbose_lines": []}

    try:
        result = orchestrator.query(eq.question, k=top_k)
    except Exception as e:
        out["error"] = str(e)
        return out

    out["result"] = result

    # Verbose: буферизуем строки для вывода в правильном порядке
    if verbose:
        lines = []
        lines.append("")
        lines.append(f"  ВОПРОС: {eq.question}")
        lines.append(f"  {'─' * 56}")
        lines.append(f"  ЧАНКИ (top-{top_k}):")
        scores = result.get("scores", [])
        for i, chunk in enumerate(result["chunks"][:top_k], 1):
            file_name = chunk.get("file_name", "Unknown")
            page = chunk.get("metadata", {}).get("page_number", "?")
            score = scores[i - 1] if i - 1 < len(scores) else 0.0
            text = chunk.get("text", "").strip()
            lines.append(f"  [{i:02d}] {file_name} | стр.{page} | score={score:.4f}")
            for line in text.splitlines():
                lines.append(f"       {line}")
        lines.append(f"  {'─' * 56}")
        lines.append(f"  ОТВЕТ:")
        for line in result.get("answer", "").splitlines():
            lines.append(f"  {line}")
        lines.append(f"  {'─' * 56}")
        out["verbose_lines"] = lines

    # Retrieval metrics
    if eq.has_page_annotations():
        ret_metrics = compute_all(
            retrieved_chunks=result["chunks"],
            expected_pages=eq.expected_pages,
            k=top_k
        )
        out["metric_mode"] = "page-based"
    else:
        expected_pages_fallback = {src: [999] for src in eq.expected_sources}
        ret_metrics = compute_all(
            retrieved_chunks=result["chunks"],
            expected_pages=expected_pages_fallback,
            k=top_k
        )
        out["metric_mode"] = "file-based"
    out["ret_metrics"] = ret_metrics

    # Generation metrics (LLM Judge)
    context = "\n\n".join([chunk["text"] for chunk in result["chunks"]])
    gen_scores = judge.evaluate(
        query=eq.question,
        answer=result["answer"],
        context=context,
        reference_answer=eq.reference_answer,
    )
    out["gen_scores"] = gen_scores

    return out


def run_evaluation(
    config_path: str,
    top_k: int,
    embedding_model: str = None,
    api_key: str = None,
    url: str = None,
    llm_url: str = None,
    embedder_url: str = None,
    llm_model: str = None,
    add_eos_token: bool = False,
    eos_token: str = None,
    verbose: bool = False,
    domain: str = "medical",
    workers: int = 4,
    timeout: int = None,
    retrieval: str = None,
    reranker_model: str = None,
    rerank_top_n: int = 20,
) -> None:
    """Главный цикл оценки."""
    orchestrator, llm_client, config = initialize_rag(
        config_path=config_path,
        top_k=top_k,
        embedding_model=embedding_model,
        api_key=api_key,
        url=url,
        llm_url=llm_url,
        embedder_url=embedder_url,
        llm_model=llm_model,
        add_eos_token=add_eos_token,
        eos_token=eos_token,
        timeout=timeout,
        retrieval=retrieval,
        reranker_model=reranker_model,
        rerank_top_n=rerank_top_n,
    )
    judge = LLMJudge(llm_client=llm_client)

    EVAL_QUESTIONS = DOMAIN_QUESTIONS[domain]

    # Загружаем эталонные ответы и привязываем к вопросам
    _GT_FILES = {
        "medical":    "ground_truth.json",
        "civil_code": "ground_truth_civil_code.json",
        "postgresql": "ground_truth_postgresql.json",
    }
    gt_path = PROJECT_ROOT / "src" / "evaluation" / _GT_FILES.get(domain, "ground_truth.json")
    gt_map = load_ground_truth(gt_path)
    for eq in EVAL_QUESTIONS:
        eq.reference_answer = gt_map.get(eq.question)
    gt_loaded = sum(1 for eq in EVAL_QUESTIONS if eq.reference_answer)

    run_start = time.time()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -- Заголовок --
    eval_log("=" * 60, results=True)
    eval_log(f"EVALUATION RUN: {now}", results=True)
    eval_log(f"Config:      {config_path}", results=True)
    eval_log(f"Model:       {config.lm_studio.llm_model}", results=True)
    eval_log(f"Embeddings:  {config.lm_studio.embedding_model}", results=True)
    eval_log(f"Collection:  {config.chromadb.collection_name}", results=True)
    eval_log(f"Top-k:       {top_k}", results=True)
    if reranker_model:
        eval_log(f"Reranker:    {reranker_model} (top_n={rerank_top_n})", results=True)
    eval_log(f"Questions:   {len(EVAL_QUESTIONS)}", results=True)
    eval_log(f"Ground truth: {gt_loaded}/{len(EVAL_QUESTIONS)} эталонных ответов загружено", results=True)
    eval_log("=" * 60, results=True)

    all_retrieval: list[dict] = []
    all_generation: list[dict] = []

    total_q = len(EVAL_QUESTIONS)
    vprint(f"Запуск {total_q} вопросов в {workers} потоке(ах)...")

    task_args = [
        (idx, eq, orchestrator, judge, top_k, verbose)
        for idx, eq in enumerate(EVAL_QUESTIONS, 1)
    ]

    # Параллельная обработка вопросов
    completed_results: list[dict] = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {executor.submit(process_question, args): args[0] for args in task_args}
            for future in as_completed(future_to_idx):
                res = future.result()
                completed_results.append(res)
                vprint(f"  [{len(completed_results)}/{total_q} готово] вопрос {res['idx']}")
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем (Ctrl+C). Отменяем оставшиеся задачи...")
        executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(130)

    # Сортируем по исходному порядку перед логированием
    completed_results.sort(key=lambda r: r["idx"])

    # Логируем результаты последовательно в правильном порядке
    for res in completed_results:
        idx = res["idx"]
        eq = res["eq"]

        eval_log("")
        eval_log(f"--- Question {idx}/{total_q} ---")
        eval_log(f"Q: {eq.question}")
        eval_log(f"Expected sources: {eq.expected_sources}")

        if res["error"]:
            eval_log(f"ERROR: запрос не удался: {res['error']}")
            logger.error(f"Question {idx}: {res['error']}")
            continue

        # Verbose: выводим буферизованные строки в порядке вопросов
        if verbose and res["verbose_lines"]:
            for line in res["verbose_lines"]:
                vprint(line)

        result = res["result"]
        ret_metrics = res["ret_metrics"]
        metric_mode = res["metric_mode"]
        gen_scores = res["gen_scores"]

        if eq.has_page_annotations():
            eval_log(f"Используется PAGE-BASED релевантность")
            eval_log(f"Expected pages: {eq.expected_pages}")
        else:
            eval_log(f"Используется FILE-BASED релевантность (legacy)")
            eval_log(f"Expected sources: {eq.expected_sources}")

        all_retrieval.append(ret_metrics)

        eval_log(
            f"Retrieval ({metric_mode}) | "
            f"AP@{top_k}={ret_metrics['map']:.4f} | "
            f"R@{top_k}={ret_metrics['recall_at_k']:.4f} | "
            f"MRR={ret_metrics['mrr']:.4f} | "
            f"NDCG@{top_k}={ret_metrics['ndcg_at_k']:.4f}"
        )

        judge_mode = "reference-based" if eq.reference_answer else "context-based"
        eval_log(f"Generation | Running LLM Judge ({judge_mode})...")

        if gen_scores:
            all_generation.append(gen_scores)
            
            # Determine which keys are present (for logging fallback)
            corr = gen_scores.get("answer_correctness", gen_scores.get("correctness"))
            faith = gen_scores.get("faithfulness", "N/A")
            rel = gen_scores.get("answer_relevancy", gen_scores.get("relevance"))
            comp = gen_scores.get("completeness")
            coh = gen_scores.get("coherence", "N/A")

            eval_log(
                f"Generation | Answer Correctness={corr} | "
                f"Faithfulness={faith} | "
                f"Answer Relevancy={rel} | "
                f"Completeness={comp} | "
                f"Coherence={coh}"
            )
        else:
            eval_log("Generation | ERROR: не удалось получить оценку от LLM Judge")

        eval_log(
            f"Timing | retrieval={result['retrieval_time']}s | "
            f"generation={result['generation_time']}s"
        )

    # -- Summary --
    total_time = time.time() - run_start

    eval_log("", results=True)
    eval_log("=" * 60, results=True)
    eval_log("SUMMARY", results=True)
    eval_log("=" * 60, results=True)

    if all_retrieval:
        n = len(all_retrieval)
        avg_map  = sum(m["map"]         for m in all_retrieval) / n
        avg_r    = sum(m["recall_at_k"] for m in all_retrieval) / n
        avg_mrr  = sum(m["mrr"]         for m in all_retrieval) / n
        avg_ndcg = sum(m["ndcg_at_k"]   for m in all_retrieval) / n

        eval_log(f"Retrieval (avg, {n} questions):", results=True)
        eval_log(f"  MAP@{top_k}:        {avg_map:.4f}", results=True)
        eval_log(f"  Recall@{top_k}:     {avg_r:.4f}", results=True)
        eval_log(f"  MRR:               {avg_mrr:.4f}", results=True)
        eval_log(f"  NDCG@{top_k}:       {avg_ndcg:.4f}", results=True)
    else:
        eval_log("Retrieval: нет данных (все запросы не удались)", results=True)

    if all_generation:
        n = len(all_generation)
        avg_corr = sum(m.get("answer_correctness", m.get("correctness", 0)) for m in all_generation) / n
        avg_faith = sum(m.get("faithfulness", 0) for m in all_generation if "faithfulness" in m) / max(1, sum(1 for m in all_generation if "faithfulness" in m))
        avg_rel  = sum(m.get("answer_relevancy", m.get("relevance", 0)) for m in all_generation) / n
        avg_comp = sum(m.get("completeness", 0) for m in all_generation) / n
        avg_coh  = sum(m.get("coherence", 0) for m in all_generation if "coherence" in m) / max(1, sum(1 for m in all_generation if "coherence" in m))

        eval_log(f"Generation (avg, {n} questions):", results=True)
        eval_log(f"  Answer Correctness: {avg_corr:.2f}", results=True)
        if avg_faith > 0: eval_log(f"  Faithfulness:       {avg_faith:.2f}", results=True)
        eval_log(f"  Answer Relevancy:   {avg_rel:.2f}", results=True)
        eval_log(f"  Completeness:       {avg_comp:.2f}", results=True)
        if avg_coh > 0: eval_log(f"  Coherence:          {avg_coh:.2f}", results=True)
    else:
        eval_log("Generation: нет данных (все оценки не удались)", results=True)

    eval_log(f"Total time: {total_time:.1f}s", results=True)
    eval_log("=" * 60, results=True)

    logger.info("Evaluation завершена. Результаты: experiments/...")


def main():
    parser = argparse.ArgumentParser(description="Оценка RAG системы compareRAG")
    parser.add_argument(
        "--config",
        default="configs/strategies/baseline.yaml",
        help="Путь к конфигурации стратегии (относительно корня проекта)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Количество документов для поиска (default: 10)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Модель эмбеддингов (переопределяет конфиг). "
             "Пример: text-embedding-qwen3-embedding-0.6b. "
             "Имя коллекции автоматически дополняется суффиксом модели."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API ключ для внешних провайдеров LLM (переопределяет конфиг)."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Базовый URL для LLM API (общий), например: https://api.polza.ru/v1"
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default=None,
        help="Специфичный URL для LLM генерации. Переопределяет общий --url."
    )
    parser.add_argument(
        "--embedder-url",
        type=str,
        default=None,
        help="Специфичный URL для эмбеддингов. Переопределяет общий --url."
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Название модели для генерации в LLM API, например: qwen/qwen3-14b"
    )
    parser.add_argument(
        "--add-eos-token",
        action="store_true",
        default=False,
        help="Добавлять EOS-токен в конец каждого текста перед эмбеддингом. "
             "Нужно для некоторых моделей (например qwen3-embedding)."
    )
    parser.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="EOS-токен для добавления (по умолчанию из конфига: </s>). "
             "Пример для Qwen3: '<|endoftext|>'"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Подробный вывод в консоль: вопрос, найденные чанки и ответ для каждого вопроса.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="medical",
        choices=["medical", "civil_code", "postgresql"],
        help="Домен вопросов для оценки: medical (по умолчанию), civil_code, postgresql.",
    )
    parser.add_argument(
        "--retrieval",
        type=str,
        default=None,
        choices=["semantic", "hybrid"],
        help="Переопределить стратегию поиска из конфига (semantic или hybrid).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Количество параллельных потоков для обработки вопросов (default: 4, max: 8). "
             "Используйте 1 для последовательного режима.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Таймаут LLM-запросов в секундах (переопределяет конфиг, default: 120). "
             "Увеличьте до 300-600 при больших промтах или очередях из нескольких потоков.",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=None,
        help="Модель переранжирования (HF ID или LM Studio имя). "
             "Пример: BAAI/bge-reranker-v2-m3 или text-embedding-bge-reranker-v2-m3. "
             "Если задана — после retrieval извлекается top_n кандидатов, "
             "переранжируется и возвращается top_k."
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=20,
        help="Сколько кандидатов извлекать ДО переранжирования (default: 20).",
    )
    args = parser.parse_args()

    # Резолв config относительно PROJECT_ROOT
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / args.config

    log_dir = str(PROJECT_ROOT / "logs")
    setup_logging(log_level="INFO", log_dir=log_dir)

    # Загружаем конфиг заранее, чтобы построить пути к логам эксперимента
    temp_config = load_config(str(config_path))
    if args.embedding_model:
        temp_config.lm_studio.embedding_model = args.embedding_model
    if args.api_key:
        temp_config.lm_studio.api_key = args.api_key
    if args.url:
        temp_config.lm_studio.url = args.url
    if args.llm_model:
        temp_config.lm_studio.llm_model = args.llm_model
    if args.retrieval:
        temp_config.retrieval.strategy = args.retrieval

    full_path, results_path = build_experiment_log_paths(
        PROJECT_ROOT, temp_config, args.domain, reranker_model=args.reranker_model
    )
    setup_experiment_logging(full_path, results_path)

    run_evaluation(
        config_path=str(config_path),
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        api_key=args.api_key,
        url=args.url,
        llm_url=args.llm_url,
        embedder_url=args.embedder_url,
        llm_model=args.llm_model,
        add_eos_token=args.add_eos_token,
        eos_token=args.eos_token,
        verbose=args.verbose,
        domain=args.domain,
        workers=args.workers,
        timeout=args.timeout,
        retrieval=args.retrieval,
        reranker_model=args.reranker_model,
        rerank_top_n=args.rerank_top_n,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем (Ctrl+C)")
        sys.exit(130)