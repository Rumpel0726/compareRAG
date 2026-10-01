#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Батч-раннер для запуска всех 54 комбинаций тестов RAG.

3 домена × 3 чанкинг-стратегии × 3 модели эмбеддингов × 2 retrieval-стратегии = 54 eval-теста
(27 уникальных ingestion-задач, т.к. retrieval не влияет на индексацию)

Использование:
    # Dry run — показать все команды без выполнения:
    python scripts/run_batch.py --api-key YOUR_KEY --dry-run

    # Запустить всё:
    python scripts/run_batch.py --api-key YOUR_KEY

    # Запустить только один домен:
    python scripts/run_batch.py --api-key YOUR_KEY --only-domain medical

    # Пропустить ingestion (коллекции уже есть):
    python scripts/run_batch.py --api-key YOUR_KEY --skip-ingest

    # Возобновить после сбоя (прогресс сохраняется автоматически):
    python scripts/run_batch.py --api-key YOUR_KEY
"""

import sys
import json
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROGRESS_FILE = PROJECT_ROOT / "batch_progress.json"
BATCH_LOG_DIR = PROJECT_ROOT / "batch_logs"

# --- Матрица комбинаций ---

DOMAINS = ["medical", "civil_code", "postgresql"]

CHUNKING_STRATEGIES = ["chonkie", "sentence", "recursive"]

EMBEDDING_MODELS = [
    "text-embedding-bge-m3",
    "text-embedding-qwen3-embedding-0.6b",
    "text-embedding-multilingual-e5-large-instruct",
]

RETRIEVAL_STRATEGIES = ["semantic", "hybrid"]

# Маппинг (домен, стратегия) -> путь к конфигу
CONFIG_MAP = {
    ("medical", "chonkie"):      "configs/strategies/medical/baseline.yaml",
    ("medical", "sentence"):     "configs/strategies/medical/sentence_baseline.yaml",
    ("medical", "recursive"):    "configs/strategies/medical/recursive_baseline.yaml",
    ("civil_code", "chonkie"):   "configs/strategies/civil_code/baseline.yaml",
    ("civil_code", "sentence"):  "configs/strategies/civil_code/sentence_baseline.yaml",
    ("civil_code", "recursive"): "configs/strategies/civil_code/recursive_baseline.yaml",
    ("postgresql", "chonkie"):   "configs/strategies/postgresql/baseline.yaml",
    ("postgresql", "sentence"):  "configs/strategies/postgresql/sentence_baseline.yaml",
    ("postgresql", "recursive"): "configs/strategies/postgresql/recursive_baseline.yaml",
}

DATA_DIRS = {
    "medical":    "data/Medical_articles",
    "civil_code": "data/Civil_Code",
    "postgresql": "data/PostrgreSQL",
}

# Короткие имена для таблицы
EMB_SHORT = {
    "text-embedding-bge-m3": "bge-m3",
    "text-embedding-qwen3-embedding-0.6b": "qwen3",
    "text-embedding-multilingual-e5-large-instruct": "e5-large",
}


# --- Прогресс ---

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"ingest": {}, "eval": {}}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


# --- Проверка коллекции ---

def collection_exists(config_path: str, embedding_model: str) -> bool:
    """Проверяет, есть ли документы в ChromaDB коллекции."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.core.config import load_config
        import chromadb

        config = load_config(str(PROJECT_ROOT / config_path))
        config.lm_studio.embedding_model = embedding_model
        model_id = config.lm_studio.embedding_model_id
        safe_model_id = model_id.replace("/", "_").replace("-", "_")
        collection_name = f"{config.chromadb.collection_name}_{safe_model_id}"

        chroma_path = Path(config.chromadb.path)
        if not chroma_path.is_absolute():
            chroma_path = PROJECT_ROOT / config.chromadb.path

        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection(name=collection_name)
        count = collection.count()
        return count > 0
    except Exception:
        return False


# --- Выполнение ---

def run_command(cmd: list[str], task_key: str, dry_run: bool = False) -> tuple[bool, str]:
    """Запускает subprocess и сохраняет лог. Возвращает (success, message)."""
    cmd_str = " ".join(cmd)

    if dry_run:
        print(f"  [DRY RUN] {cmd_str}")
        return True, "dry run"

    BATCH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = BATCH_LOG_DIR / f"{task_key.replace('/', '_')}.log"

    print(f"  CMD: {cmd_str}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=7200,  # 2 часа макс на задачу
        )

        elapsed = time.time() - start

        # Записываем лог
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"CMD: {cmd_str}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"Duration: {elapsed:.1f}s\n")
            f.write(f"\n{'='*60}\nSTDOUT:\n{'='*60}\n")
            f.write(result.stdout)
            f.write(f"\n{'='*60}\nSTDERR:\n{'='*60}\n")
            f.write(result.stderr)

        if result.returncode != 0:
            # Показываем последние строки stderr
            stderr_lines = result.stderr.strip().split("\n")
            last_lines = "\n".join(stderr_lines[-5:])
            return False, f"exit code {result.returncode} ({elapsed:.0f}s)\n{last_lines}"

        return True, f"OK ({elapsed:.0f}s)"

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT (2h)"
    except Exception as e:
        return False, str(e)


def run_ingest(domain: str, chunking: str, embedding: str,
               dry_run: bool = False, force: bool = False) -> tuple[bool, str]:
    """Запускает ingestion для одной комбинации."""
    config_path = CONFIG_MAP[(domain, chunking)]
    data_dir = DATA_DIRS[domain]

    # Проверяем, существует ли коллекция
    if not force and not dry_run:
        if collection_exists(config_path, embedding):
            return True, "SKIP (collection exists)"

    cmd = [
        sys.executable, "scripts/ingest_documents.py",
        "--config", config_path,
        "--data-dir", data_dir,
        "--embedding-model", embedding,
        "--reset",
    ]

    task_key = f"ingest_{domain}_{chunking}_{EMB_SHORT.get(embedding, embedding)}"
    return run_command(cmd, task_key, dry_run)


def run_eval(domain: str, chunking: str, embedding: str, retrieval: str,
             api_key: str, llm_url: str, llm_model: str,
             embedder_url: str, workers: int, timeout: int,
             dry_run: bool = False) -> tuple[bool, str]:
    """Запускает evaluation для одной комбинации."""
    config_path = CONFIG_MAP[(domain, chunking)]

    cmd = [
        sys.executable, "scripts/run_evaluation.py",
        "--config", config_path,
        "--domain", domain,
        "--top-k", "10",
        "--workers", str(workers),
        "--timeout", str(timeout),
        "--embedding-model", embedding,
        "--retrieval", retrieval,
        "--embedder-url", embedder_url,
        "--llm-url", llm_url,
        "--llm-model", llm_model,
        "--api-key", api_key,
    ]

    emb_short = EMB_SHORT.get(embedding, embedding)
    task_key = f"eval_{domain}_{chunking}_{emb_short}_{retrieval}"
    return run_command(cmd, task_key, dry_run)


# --- Summary ---

def print_summary(progress: dict):
    """Выводит итоговую таблицу результатов."""
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ ТАБЛИЦА")
    print("=" * 80)

    # Ingest
    ingest = progress.get("ingest", {})
    eval_data = progress.get("eval", {})

    done = sum(1 for v in eval_data.values() if v.get("status") == "done")
    failed = sum(1 for v in eval_data.values() if v.get("status") == "failed")
    skipped = sum(1 for v in eval_data.values() if v.get("status") == "skipped")
    total = done + failed + skipped

    print(f"\nIngest:  {sum(1 for v in ingest.values() if v.get('status') == 'done')} done, "
          f"{sum(1 for v in ingest.values() if v.get('status') == 'failed')} failed")
    print(f"Eval:    {done} done, {failed} failed, {skipped} skipped (total: {total})")

    if failed > 0:
        print(f"\nFAILED задачи:")
        for key, val in eval_data.items():
            if val.get("status") == "failed":
                print(f"  {key}: {val.get('message', '?')[:100]}")
        for key, val in ingest.items():
            if val.get("status") == "failed":
                print(f"  {key}: {val.get('message', '?')[:100]}")

    print(f"\nЛоги задач: {BATCH_LOG_DIR}/")
    print(f"Прогресс:   {PROGRESS_FILE}")
    print("=" * 80)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Батч-раннер: 54 комбинации тестов RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--api-key", required=True, help="API ключ для LLM (polza.ai)")
    parser.add_argument("--llm-url", default="https://polza.ai/api", help="URL для LLM генерации")
    parser.add_argument("--llm-model", default="qwen/qwen3-14b", help="Модель генерации")
    parser.add_argument("--embedder-url", default="http://127.0.0.1:1234", help="URL для эмбеддингов (LM Studio)")
    parser.add_argument("--workers", type=int, default=4, help="Параллельные потоки для eval (default: 4)")
    parser.add_argument("--timeout", type=int, default=600, help="Таймаут LLM-запросов в секундах (default: 600)")
    parser.add_argument("--skip-ingest", action="store_true", help="Пропустить все ingestion, только eval")
    parser.add_argument("--force-ingest", action="store_true", help="Переиндексировать даже если коллекция существует")
    parser.add_argument("--only-domain", choices=DOMAINS, help="Только один домен")
    parser.add_argument("--only-chunking", choices=CHUNKING_STRATEGIES, help="Только одна стратегия чанкинга")
    parser.add_argument("--only-embedding", type=str, help="Только одна модель эмбеддингов")
    parser.add_argument("--only-retrieval", choices=RETRIEVAL_STRATEGIES, help="Только одна retrieval-стратегия")
    parser.add_argument("--dry-run", action="store_true", help="Показать команды без выполнения")
    parser.add_argument("--reset-progress", action="store_true", help="Сбросить файл прогресса и начать заново")
    args = parser.parse_args()

    # Фильтрация
    domains = [args.only_domain] if args.only_domain else DOMAINS
    chunkings = [args.only_chunking] if args.only_chunking else CHUNKING_STRATEGIES
    embeddings = [args.only_embedding] if args.only_embedding else EMBEDDING_MODELS
    retrievals = [args.only_retrieval] if args.only_retrieval else RETRIEVAL_STRATEGIES

    # Прогресс
    if args.reset_progress and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
    progress = load_progress()

    # Собираем уникальные ingest-задачи и eval-задачи
    ingest_tasks = []
    eval_tasks = []

    for domain in domains:
        for chunking in chunkings:
            for embedding in embeddings:
                ingest_key = f"{domain}/{chunking}/{EMB_SHORT.get(embedding, embedding)}"
                ingest_tasks.append((domain, chunking, embedding, ingest_key))

                for retrieval in retrievals:
                    eval_key = f"{domain}/{chunking}/{EMB_SHORT.get(embedding, embedding)}/{retrieval}"
                    eval_tasks.append((domain, chunking, embedding, retrieval, eval_key))

    # Убираем дубликаты ingest (могут быть при фильтрации retrieval)
    seen_ingest = set()
    unique_ingest = []
    for task in ingest_tasks:
        if task[3] not in seen_ingest:
            seen_ingest.add(task[3])
            unique_ingest.append(task)
    ingest_tasks = unique_ingest

    total_ingest = len(ingest_tasks)
    total_eval = len(eval_tasks)

    print(f"{'='*60}")
    print(f"BATCH RUN: {total_ingest} ingests + {total_eval} evals = {total_ingest + total_eval} задач")
    print(f"Домены:    {domains}")
    print(f"Чанкинг:   {chunkings}")
    print(f"Эмбеддинги: {[EMB_SHORT.get(e, e) for e in embeddings]}")
    print(f"Retrieval: {retrievals}")
    print(f"{'='*60}\n")

    # -- Фаза 1: Ingestion --
    if not args.skip_ingest:
        print(f"{'-'*60}")
        print(f"ФАЗА 1: INGESTION ({total_ingest} задач)")
        print(f"{'-'*60}")

        for i, (domain, chunking, embedding, ingest_key) in enumerate(ingest_tasks, 1):
            emb_short = EMB_SHORT.get(embedding, embedding)
            print(f"\n[{i}/{total_ingest}] INGEST: {domain} / {chunking} / {emb_short}")

            # Проверяем прогресс
            prev = progress.get("ingest", {}).get(ingest_key, {})
            if prev.get("status") == "done" and not args.force_ingest:
                print(f"  SKIP (уже выполнено)")
                continue

            success, message = run_ingest(
                domain, chunking, embedding,
                dry_run=args.dry_run,
                force=args.force_ingest,
            )

            status = "done" if success else "failed"
            if "SKIP" in message:
                status = "done"

            progress.setdefault("ingest", {})[ingest_key] = {
                "status": status,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            }
            if not args.dry_run:
                save_progress(progress)

            print(f"  -> {status.upper()}: {message}")

            if not success and "SKIP" not in message:
                print(f"  WARNING: Ingestion failed — eval задачи для этой комбинации будут пропущены")

    # -- Фаза 2: Evaluation --
    print(f"\n{'-'*60}")
    print(f"ФАЗА 2: EVALUATION ({total_eval} задач)")
    print(f"{'-'*60}")

    for i, (domain, chunking, embedding, retrieval, eval_key) in enumerate(eval_tasks, 1):
        emb_short = EMB_SHORT.get(embedding, embedding)
        print(f"\n[{i}/{total_eval}] EVAL: {domain} / {chunking} / {emb_short} / {retrieval}")

        # Проверяем прогресс
        prev = progress.get("eval", {}).get(eval_key, {})
        if prev.get("status") == "done":
            print(f"  SKIP (уже выполнено)")
            continue

        # Проверяем, прошёл ли ingest
        ingest_key = f"{domain}/{chunking}/{emb_short}"
        ingest_status = progress.get("ingest", {}).get(ingest_key, {}).get("status")
        if not args.skip_ingest and ingest_status == "failed":
            print(f"  SKIP (ingest failed)")
            progress.setdefault("eval", {})[eval_key] = {
                "status": "skipped",
                "message": "ingest failed",
                "timestamp": datetime.now().isoformat(),
            }
            if not args.dry_run:
                save_progress(progress)
            continue

        success, message = run_eval(
            domain, chunking, embedding, retrieval,
            api_key=args.api_key,
            llm_url=args.llm_url,
            llm_model=args.llm_model,
            embedder_url=args.embedder_url,
            workers=args.workers,
            timeout=args.timeout,
            dry_run=args.dry_run,
        )

        status = "done" if success else "failed"
        progress.setdefault("eval", {})[eval_key] = {
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        if not args.dry_run:
            save_progress(progress)

        print(f"  -> {status.upper()}: {message}")

    # -- Summary --
    print_summary(progress)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем (Ctrl+C)")
        sys.exit(130)
