#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тест влияния размера чанка на качество RAG.

3 домена × 3 стратегии × 3 размера (256, 512, 1024) = 27 конфигураций
Эмбеддинги зафиксированы: BGE-M3
Retrieval зафиксирован: hybrid
9 конфигураций (chunk_size=512) уже есть в основных результатах — ingestion будет пропущен автоматически.

Использование:
    # Dry run — показать все команды без выполнения:
    python scripts/run_chunksize_test.py --api-key YOUR_KEY --dry-run

    # Полный запуск:
    python scripts/run_chunksize_test.py --api-key YOUR_KEY

    # Только один домен:
    python scripts/run_chunksize_test.py --api-key YOUR_KEY --only-domain civil_code

    # Только одна стратегия чанкинга:
    python scripts/run_chunksize_test.py --api-key YOUR_KEY --only-chunking sentence

    # Только один размер:
    python scripts/run_chunksize_test.py --api-key YOUR_KEY --only-size 256

    # Возобновить после сбоя:
    python scripts/run_chunksize_test.py --api-key YOUR_KEY
"""

import sys
import json
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROGRESS_FILE = PROJECT_ROOT / "chunksize_progress.json"
BATCH_LOG_DIR = PROJECT_ROOT / "batch_logs"

# --- Матрица комбинаций ---

DOMAINS = ["medical", "civil_code", "postgresql"]
CHUNKING_STRATEGIES = ["chonkie", "sentence", "recursive"]
CHUNK_SIZES = [256, 512, 1024]

EMBEDDING_MODEL = "text-embedding-bge-m3"
RETRIEVAL_STRATEGY = "hybrid"

# Маппинг (домен, стратегия, размер) -> путь к конфигу
CONFIG_MAP = {
    # medical
    ("medical", "chonkie",   256):  "configs/strategies/medical/small_chunks.yaml",
    ("medical", "chonkie",   512):  "configs/strategies/medical/baseline.yaml",
    ("medical", "chonkie",  1024):  "configs/strategies/medical/large_chunks.yaml",
    ("medical", "sentence",  256):  "configs/strategies/medical/sentence_small.yaml",
    ("medical", "sentence",  512):  "configs/strategies/medical/sentence_baseline.yaml",
    ("medical", "sentence", 1024):  "configs/strategies/medical/sentence_large.yaml",
    ("medical", "recursive", 256):  "configs/strategies/medical/recursive_small.yaml",
    ("medical", "recursive", 512):  "configs/strategies/medical/recursive_baseline.yaml",
    ("medical", "recursive", 1024): "configs/strategies/medical/recursive_large.yaml",
    # civil_code
    ("civil_code", "chonkie",   256):  "configs/strategies/civil_code/small_chunks.yaml",
    ("civil_code", "chonkie",   512):  "configs/strategies/civil_code/baseline.yaml",
    ("civil_code", "chonkie",  1024):  "configs/strategies/civil_code/large_chunks.yaml",
    ("civil_code", "sentence",  256):  "configs/strategies/civil_code/sentence_small.yaml",
    ("civil_code", "sentence",  512):  "configs/strategies/civil_code/sentence_baseline.yaml",
    ("civil_code", "sentence", 1024):  "configs/strategies/civil_code/sentence_large.yaml",
    ("civil_code", "recursive", 256):  "configs/strategies/civil_code/recursive_small.yaml",
    ("civil_code", "recursive", 512):  "configs/strategies/civil_code/recursive_baseline.yaml",
    ("civil_code", "recursive", 1024): "configs/strategies/civil_code/recursive_large.yaml",
    # postgresql
    ("postgresql", "chonkie",   256):  "configs/strategies/postgresql/small_chunks.yaml",
    ("postgresql", "chonkie",   512):  "configs/strategies/postgresql/baseline.yaml",
    ("postgresql", "chonkie",  1024):  "configs/strategies/postgresql/large_chunks.yaml",
    ("postgresql", "sentence",  256):  "configs/strategies/postgresql/sentence_small.yaml",
    ("postgresql", "sentence",  512):  "configs/strategies/postgresql/sentence_baseline.yaml",
    ("postgresql", "sentence", 1024):  "configs/strategies/postgresql/sentence_large.yaml",
    ("postgresql", "recursive", 256):  "configs/strategies/postgresql/recursive_small.yaml",
    ("postgresql", "recursive", 512):  "configs/strategies/postgresql/recursive_baseline.yaml",
    ("postgresql", "recursive", 1024): "configs/strategies/postgresql/recursive_large.yaml",
}

DATA_DIRS = {
    "medical":    "data/Medical_articles",
    "civil_code": "data/Civil_Code",
    "postgresql": "data/PostrgreSQL",
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

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"CMD: {cmd_str}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"Duration: {elapsed:.1f}s\n")
            f.write(f"\n{'='*60}\nSTDOUT:\n{'='*60}\n")
            f.write(result.stdout)
            f.write(f"\n{'='*60}\nSTDERR:\n{'='*60}\n")
            f.write(result.stderr)

        if result.returncode != 0:
            stderr_lines = result.stderr.strip().split("\n")
            last_lines = "\n".join(stderr_lines[-5:])
            return False, f"exit code {result.returncode} ({elapsed:.0f}s)\n{last_lines}"

        return True, f"OK ({elapsed:.0f}s)"

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT (2h)"
    except Exception as e:
        return False, str(e)


def run_ingest(domain: str, chunking: str, size: int,
               dry_run: bool = False, force: bool = False) -> tuple[bool, str]:
    """Запускает ingestion для одной комбинации."""
    config_path = CONFIG_MAP[(domain, chunking, size)]
    data_dir = DATA_DIRS[domain]

    if not force and not dry_run:
        if collection_exists(config_path, EMBEDDING_MODEL):
            return True, "SKIP (collection exists)"

    cmd = [
        sys.executable, "scripts/ingest_documents.py",
        "--config", config_path,
        "--data-dir", data_dir,
        "--embedding-model", EMBEDDING_MODEL,
        "--reset",
    ]

    task_key = f"cs_ingest_{domain}_{chunking}_{size}"
    return run_command(cmd, task_key, dry_run)


def run_eval(domain: str, chunking: str, size: int,
             top_k: int, api_key: str, llm_url: str, llm_model: str,
             embedder_url: str, workers: int, timeout: int,
             dry_run: bool = False) -> tuple[bool, str]:
    """Запускает evaluation для одной комбинации."""
    config_path = CONFIG_MAP[(domain, chunking, size)]

    cmd = [
        sys.executable, "scripts/run_evaluation.py",
        "--config", config_path,
        "--domain", domain,
        "--top-k", str(top_k),
        "--workers", str(workers),
        "--timeout", str(timeout),
        "--embedding-model", EMBEDDING_MODEL,
        "--retrieval", RETRIEVAL_STRATEGY,
        "--embedder-url", embedder_url,
        "--llm-url", llm_url,
        "--llm-model", llm_model,
        "--api-key", api_key,
    ]

    task_key = f"cs_eval_{domain}_{chunking}_{size}"
    return run_command(cmd, task_key, dry_run)


# --- Summary ---

def print_summary(progress: dict):
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ ТАБЛИЦА (тест размера чанка)")
    print("=" * 80)

    ingest = progress.get("ingest", {})
    eval_data = progress.get("eval", {})

    done_i = sum(1 for v in ingest.values() if v.get("status") == "done")
    failed_i = sum(1 for v in ingest.values() if v.get("status") == "failed")
    done_e = sum(1 for v in eval_data.values() if v.get("status") == "done")
    failed_e = sum(1 for v in eval_data.values() if v.get("status") == "failed")
    skipped_e = sum(1 for v in eval_data.values() if v.get("status") == "skipped")

    print(f"\nIngest: {done_i} done, {failed_i} failed")
    print(f"Eval:   {done_e} done, {failed_e} failed, {skipped_e} skipped (total: {done_e + failed_e + skipped_e})")

    if failed_i + failed_e > 0:
        print("\nFAILED задачи:")
        for key, val in {**ingest, **eval_data}.items():
            if val.get("status") == "failed":
                print(f"  {key}: {val.get('message', '?')[:100]}")

    print(f"\nЛоги задач: {BATCH_LOG_DIR}/")
    print(f"Прогресс:   {PROGRESS_FILE}")
    print("=" * 80)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Тест размера чанка: 3 домена × 3 стратегии × 3 размера = 27 конфигураций",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--api-key", required=True, help="API ключ для LLM (polza.ai)")
    parser.add_argument("--llm-url", default="https://polza.ai/api", help="URL для LLM генерации")
    parser.add_argument("--llm-model", default="qwen/qwen3-14b", help="Модель генерации")
    parser.add_argument("--embedder-url", default="http://127.0.0.1:1234", help="URL для эмбеддингов (LM Studio)")
    parser.add_argument("--top-k", type=int, default=10, help="Количество документов для retrieval (default: 10)")
    parser.add_argument("--workers", type=int, default=4, help="Параллельные потоки для eval (default: 4)")
    parser.add_argument("--timeout", type=int, default=600, help="Таймаут LLM-запросов в секундах (default: 600)")
    parser.add_argument("--skip-ingest", action="store_true", help="Пропустить все ingestion, только eval")
    parser.add_argument("--force-ingest", action="store_true", help="Переиндексировать даже если коллекция существует")
    parser.add_argument("--only-domain", choices=DOMAINS, help="Только один домен")
    parser.add_argument("--only-chunking", choices=CHUNKING_STRATEGIES, help="Только одна стратегия чанкинга")
    parser.add_argument("--only-size", type=int, choices=CHUNK_SIZES, help="Только один размер чанка")
    parser.add_argument("--dry-run", action="store_true", help="Показать команды без выполнения")
    parser.add_argument("--reset-progress", action="store_true", help="Сбросить файл прогресса и начать заново")
    args = parser.parse_args()

    # Фильтрация
    domains = [args.only_domain] if args.only_domain else DOMAINS
    chunkings = [args.only_chunking] if args.only_chunking else CHUNKING_STRATEGIES
    sizes = [args.only_size] if args.only_size else CHUNK_SIZES

    # Прогресс
    if args.reset_progress and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
    progress = load_progress()

    # Формируем задачи
    tasks = [
        (domain, chunking, size)
        for domain in domains
        for chunking in chunkings
        for size in sizes
    ]

    total = len(tasks)

    print(f"{'='*60}")
    print(f"ТЕСТ РАЗМЕРА ЧАНКА: {total} конфигураций")
    print(f"Домены:    {domains}")
    print(f"Чанкинг:   {chunkings}")
    print(f"Размеры:   {sizes}")
    print(f"Эмбеддинги: {EMBEDDING_MODEL}")
    print(f"Retrieval: {RETRIEVAL_STRATEGY}")
    print(f"top_k:     {args.top_k}")
    print(f"{'='*60}\n")

    # -- Фаза 1: Ingestion --
    if not args.skip_ingest:
        print(f"{'-'*60}")
        print(f"ФАЗА 1: INGESTION ({total} задач, auto-skip если коллекция существует)")
        print(f"{'-'*60}")

        for i, (domain, chunking, size) in enumerate(tasks, 1):
            ingest_key = f"{domain}/{chunking}/{size}"
            print(f"\n[{i}/{total}] INGEST: {domain} / {chunking} / {size}")

            prev = progress.get("ingest", {}).get(ingest_key, {})
            if prev.get("status") == "done" and not args.force_ingest:
                print(f"  SKIP (уже выполнено)")
                continue

            success, message = run_ingest(
                domain, chunking, size,
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
                print(f"  WARNING: Ingestion failed — eval для этой комбинации будет пропущен")

    # -- Фаза 2: Evaluation --
    print(f"\n{'-'*60}")
    print(f"ФАЗА 2: EVALUATION ({total} задач)")
    print(f"{'-'*60}")

    for i, (domain, chunking, size) in enumerate(tasks, 1):
        eval_key = f"{domain}/{chunking}/{size}/eval"
        print(f"\n[{i}/{total}] EVAL: {domain} / {chunking} / {size}")

        prev = progress.get("eval", {}).get(eval_key, {})
        if prev.get("status") == "done":
            print(f"  SKIP (уже выполнено)")
            continue

        ingest_key = f"{domain}/{chunking}/{size}"
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
            domain, chunking, size,
            top_k=args.top_k,
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

    print_summary(progress)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем (Ctrl+C)")
        sys.exit(130)
