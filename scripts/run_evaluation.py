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

from loguru import logger

# Корень проекта (scripts/ -> compareRAG/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.embeddings.factory import create_embedder_from_config
from src.vector_stores.factory import create_vector_store_from_config
from src.retrieval.factory import create_retriever_from_config
from src.llm.lm_studio_client import LMStudioClient
from src.agents.orchestrator import SimpleRAGOrchestrator
from src.evaluation.questions import EVAL_QUESTIONS
from src.evaluation.retrieval_metrics import compute_all
from src.evaluation.generation_metrics import LLMJudge
from src.utils.logging_config import setup_logging


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


def build_experiment_log_paths(project_root: Path, config) -> tuple[Path, Path]:
    """Build full and results log file paths for the current experiment run."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    retrieval = config.retrieval.strategy          # semantic | hybrid
    chunk_strategy = config.chunking.strategy      # chonkie | sentence | langchain
    chunk_size = str(config.chunking.chunk_size)   # 256, 512, 1024
    emb_folder, emb_abbrev = get_embedding_folder_and_abbrev(config.lm_studio.embedding_model)

    base_dir = project_root / "experiments" / retrieval / emb_folder / chunk_strategy / chunk_size
    full_dir = base_dir / "full"
    results_dir = base_dir / "results"

    file_prefix = f"{retrieval}_{chunk_strategy}_{chunk_size}_{emb_abbrev}"
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
    add_eos_token: bool = False,
    eos_token: str = None,
):
    """Инициализация компонентов RAG --- по паттерну web/app.py."""
    logger.info(f"Загрузка конфигурации: {config_path}")
    config = load_config(config_path)

    # Переопределение модели эмбеддингов (если задана через CLI)
    if embedding_model:
        config.lm_studio.embedding_model = embedding_model
        model_id = config.lm_studio.embedding_model_id
        config.chromadb.collection_name = f"{config.chromadb.collection_name}_{model_id}"
        logger.info(f"Модель эмбеддингов (CLI): {embedding_model}")
        logger.info(f"Коллекция (с суффиксом модели): {config.chromadb.collection_name}")

    if add_eos_token:
        config.lm_studio.add_eos_token = True
        logger.info("EOS-токен включён")
    if eos_token:
        config.lm_studio.eos_token = eos_token
        logger.info(f"EOS-токен: {eos_token}")

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

    logger.info("Инициализация ретривера...")
    retriever = create_retriever_from_config(config, embedder, vector_store)

    logger.info("Инициализация LLM клиента...")
    llm_client = LMStudioClient(
        url=config.lm_studio.url,
        model=config.lm_studio.llm_model,
        temperature=config.lm_studio.temperature,
        max_tokens=config.lm_studio.max_tokens,
        timeout=config.lm_studio.timeout,
    )

    logger.info("Инициализация orchestrator...")
    orchestrator = SimpleRAGOrchestrator(
        retriever=retriever,
        llm_client=llm_client,
        top_k=top_k,
    )

    return orchestrator, llm_client, config




def run_evaluation(
    config_path: str,
    top_k: int,
    embedding_model: str = None,
    add_eos_token: bool = False,
    eos_token: str = None,
    verbose: bool = False,
) -> None:
    """Главный цикл оценки."""
    orchestrator, llm_client, config = initialize_rag(
        config_path, top_k, embedding_model, add_eos_token, eos_token
    )
    judge = LLMJudge(llm_client=llm_client)

    # Загружаем эталонные ответы и привязываем к вопросам
    gt_path = PROJECT_ROOT / "src" / "evaluation" / "ground_truth.json"
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
    eval_log(f"Questions:   {len(EVAL_QUESTIONS)}", results=True)
    eval_log(f"Ground truth: {gt_loaded}/{len(EVAL_QUESTIONS)} эталонных ответов загружено", results=True)
    eval_log("=" * 60, results=True)

    all_retrieval: list[dict] = []
    all_generation: list[dict] = []

    for idx, eq in enumerate(EVAL_QUESTIONS, 1):
        eval_log("")
        eval_log(f"--- Question {idx}/{len(EVAL_QUESTIONS)} ---")
        eval_log(f"Q: {eq.question}")
        eval_log(f"Expected sources: {eq.expected_sources}")

        # -- RAG query --
        try:
            result = orchestrator.query(eq.question, k=top_k)
        except Exception as e:
            eval_log(f"ERROR: запрос не удался: {e}")
            logger.error(f"Question {idx}: {e}")
            continue

        # -- Verbose: чанки и ответ в консоль --
        if verbose:
            vprint()
            vprint(f"  ВОПРОС: {eq.question}")
            vprint(f"  {'─' * 56}")
            vprint(f"  ЧАНКИ (top-{top_k}):")
            scores = result.get("scores", [])
            for i, chunk in enumerate(result["chunks"][:top_k], 1):
                file_name = chunk.get("file_name", "Unknown")
                page = chunk.get("metadata", {}).get("page_number", "?")
                score = scores[i - 1] if i - 1 < len(scores) else 0.0
                text = chunk.get("text", "").strip()
                vprint(f"  [{i:02d}] {file_name} | стр.{page} | score={score:.4f}")
                for line in text.splitlines():
                    vprint(f"       {line}")
            vprint(f"  {'─' * 56}")
            vprint(f"  ОТВЕТ:")
            for line in result.get("answer", "").splitlines():
                vprint(f"  {line}")
            vprint(f"  {'─' * 56}")

        # -- Retrieval metrics --
        # eval_log(f"Retrieved chunks (top-{top_k}):")
        # for i, chunk in enumerate(result["chunks"][:top_k], 1):
        #     file_name = chunk.get("file_name", "Unknown")
        #     page_number = chunk.get("metadata", {}).get("page_number", "?")
        #     score = result.get('scores', [])[i-1] if i-1 < len(result.get('scores', [])) else 0.0
        #     text_preview = chunk.get("text", "")[:80].replace("\n", " ")
        #     eval_log(f"  [{i}] file={file_name[:40]} | page={page_number} | score={score:.4f}")
        #     eval_log(f"      text: {text_preview}...")

        # Выбор формата ground truth (поддержка обратной совместимости)
        if eq.has_page_annotations():
            # Новый формат: page-based
            eval_log(f"Используется PAGE-BASED релевантность")
            eval_log(f"Expected pages: {eq.expected_pages}")

            ret_metrics = compute_all(
                retrieved_chunks=result["chunks"],
                expected_pages=eq.expected_pages,
                k=top_k
            )
            metric_mode = "page-based"
        else:
            # Старый формат: file-based (для обратной совместимости)
            eval_log(f"Используется FILE-BASED релевантность (legacy)")
            eval_log(f"Expected sources: {eq.expected_sources}")

            # Конвертируем expected_sources в expected_pages
            # Используем page=999 как wildcard (любая страница из этого файла)
            expected_pages_fallback = {src: [999] for src in eq.expected_sources}

            ret_metrics = compute_all(
                retrieved_chunks=result["chunks"],
                expected_pages=expected_pages_fallback,
                k=top_k
            )
            metric_mode = "file-based"

        all_retrieval.append(ret_metrics)

        eval_log(
            f"Retrieval ({metric_mode}) | "
            f"AP@{top_k}={ret_metrics['map']:.4f} | "
            f"R@{top_k}={ret_metrics['recall_at_k']:.4f} | "
            f"MRR={ret_metrics['mrr']:.4f} | "
            f"NDCG@{top_k}={ret_metrics['ndcg_at_k']:.4f}"
        )

        # -- Generation metrics (LLM Judge) --
        judge_mode = "reference-based" if eq.reference_answer else "context-based"
        eval_log(f"Generation | Running LLM Judge ({judge_mode})...")
        context = "\n\n".join([chunk["text"] for chunk in result["chunks"]])

        gen_scores = judge.evaluate(
            query=eq.question,
            answer=result["answer"],
            context=context,
            reference_answer=eq.reference_answer,
        )

        if gen_scores:
            all_generation.append(gen_scores)
            eval_log(
                f"Generation | Correctness={gen_scores['correctness']} | "
                f"Relevance={gen_scores['relevance']} | "
                f"Completeness={gen_scores['completeness']} | "
                f"Coherence={gen_scores['coherence']}"
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
        avg_corr = sum(m["correctness"]  for m in all_generation) / n
        avg_rel  = sum(m["relevance"]    for m in all_generation) / n
        avg_comp = sum(m["completeness"] for m in all_generation) / n
        avg_coh  = sum(m["coherence"]    for m in all_generation) / n

        eval_log(f"Generation (avg, {n} questions):", results=True)
        eval_log(f"  Correctness:   {avg_corr:.2f}", results=True)
        eval_log(f"  Relevance:     {avg_rel:.2f}", results=True)
        eval_log(f"  Completeness:  {avg_comp:.2f}", results=True)
        eval_log(f"  Coherence:     {avg_coh:.2f}", results=True)
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

    full_path, results_path = build_experiment_log_paths(PROJECT_ROOT, temp_config)
    setup_experiment_logging(full_path, results_path)

    run_evaluation(
        config_path=str(config_path),
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        add_eos_token=args.add_eos_token,
        eos_token=args.eos_token,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()