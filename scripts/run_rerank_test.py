#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тест переранжирования: 3 лучшие конфигурации × 3 reranker-модели = 9 eval-задач.

Лучшие конфигурации (как в топ-k тесте):
  - civil_code: sentence + hybrid + bge-m3
  - medical:    sentence + semantic + bge-m3
  - postgresql: chonkie  + hybrid + e5-large

Reranker-модели (через sentence-transformers + transformers):
  - BAAI/bge-reranker-v2-m3
  - jinaai/jina-reranker-v2-base-multilingual
  - Qwen/Qwen3-Reranker-0.6B

Retrieval: top_n=20 кандидатов → rerank → top_k=10.
Ingestion не нужен — коллекции уже проиндексированы.

Использование:
    python scripts/run_rerank_test.py --api-key YOUR_KEY --dry-run
    python scripts/run_rerank_test.py --api-key YOUR_KEY
    python scripts/run_rerank_test.py --api-key YOUR_KEY --only-domain medical
    python scripts/run_rerank_test.py --api-key YOUR_KEY --only-reranker bge
"""

import sys
import json
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROGRESS_FILE = PROJECT_ROOT / "rerank_progress.json"
BATCH_LOG_DIR = PROJECT_ROOT / "batch_logs"

# --- Конфигурации ---

BEST_CONFIGS = [
    {
        "domain": "civil_code",
        "chunking": "sentence",
        "embedding": "text-embedding-bge-m3",
        "retrieval": "hybrid",
        "config": "configs/strategies/civil_code/sentence_baseline.yaml",
        "emb_short": "bge-m3",
    },
    {
        "domain": "medical",
        "chunking": "sentence",
        "embedding": "text-embedding-bge-m3",
        "retrieval": "semantic",
        "config": "configs/strategies/medical/sentence_baseline.yaml",
        "emb_short": "bge-m3",
    },
    {
        "domain": "postgresql",
        "chunking": "chonkie",
        "embedding": "text-embedding-multilingual-e5-large-instruct",
        "retrieval": "hybrid",
        "config": "configs/strategies/postgresql/baseline.yaml",
        "emb_short": "e5-large",
    },
]

RERANKERS = [
    {"id": "BAAI/bge-reranker-v2-m3",                          "short": "bge"},
    {"id": "jinaai/jina-reranker-v2-base-multilingual",        "short": "jina"},
    {"id": "Qwen/Qwen3-Reranker-0.6B",                         "short": "qwen3"},
]

TOP_K = 10
RERANK_TOP_N = 20

DOMAINS = [c["domain"] for c in BEST_CONFIGS]
RERANKER_SHORTS = [r["short"] for r in RERANKERS]


# --- Прогресс ---

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"eval": {}}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


# --- Выполнение ---

def run_command(cmd: list[str], task_key: str, dry_run: bool = False,
                stream: bool = False) -> tuple[bool, str]:
    cmd_str = " ".join(cmd)

    if dry_run:
        print(f"  [DRY RUN] {cmd_str}")
        return True, "dry run"

    BATCH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = BATCH_LOG_DIR / f"{task_key.replace('/', '_')}.log"

    print(f"  CMD: {cmd_str}")
    start = time.time()

    try:
        if stream:
            # Stream output в реальном времени + параллельная запись в файл
            with open(log_file, "w", encoding="utf-8", buffering=1) as f:
                f.write(f"CMD: {cmd_str}\n\n")
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(PROJECT_ROOT),
                    bufsize=1,
                    encoding="utf-8",
                    errors="replace",
                )
                try:
                    for line in proc.stdout:
                        print(f"  | {line}", end="")
                        f.write(line)
                    proc.wait(timeout=14400)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    return False, "TIMEOUT (4h)"

            elapsed = time.time() - start
            if proc.returncode != 0:
                return False, f"exit code {proc.returncode} ({elapsed:.0f}s)"
            return True, f"OK ({elapsed:.0f}s)"
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=14400,  # 4 часа: reranker на CPU медленный
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
        return False, "TIMEOUT (4h)"
    except Exception as e:
        return False, str(e)


def run_eval(cfg: dict, reranker: dict,
             api_key: str, llm_url: str, llm_model: str,
             embedder_url: str, workers: int, timeout: int,
             dry_run: bool = False, stream: bool = False) -> tuple[bool, str]:
    cmd = [
        sys.executable, "scripts/run_evaluation.py",
        "--config", cfg["config"],
        "--domain", cfg["domain"],
        "--top-k", str(TOP_K),
        "--workers", str(workers),
        "--timeout", str(timeout),
        "--embedding-model", cfg["embedding"],
        "--retrieval", cfg["retrieval"],
        "--reranker-model", reranker["id"],
        "--rerank-top-n", str(RERANK_TOP_N),
        "--embedder-url", embedder_url,
        "--llm-url", llm_url,
        "--llm-model", llm_model,
        "--api-key", api_key,
    ]

    task_key = f"rerank_{cfg['domain']}_{cfg['chunking']}_{cfg['emb_short']}_{cfg['retrieval']}_{reranker['short']}"
    return run_command(cmd, task_key, dry_run, stream=stream)


# --- Summary ---

def print_summary(progress: dict):
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ ТАБЛИЦА (rerank тест)")
    print("=" * 80)

    eval_data = progress.get("eval", {})
    done = sum(1 for v in eval_data.values() if v.get("status") == "done")
    failed = sum(1 for v in eval_data.values() if v.get("status") == "failed")
    total = done + failed

    print(f"\nEval: {done} done, {failed} failed (total: {total})")

    if failed > 0:
        print("\nFAILED задачи:")
        for key, val in eval_data.items():
            if val.get("status") == "failed":
                print(f"  {key}: {val.get('message', '?')[:100]}")

    print(f"\nЛоги задач: {BATCH_LOG_DIR}/")
    print(f"Прогресс:   {PROGRESS_FILE}")
    print("=" * 80)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Тест переранжирования: 3 конфигурации × 3 reranker = 9 eval-задач",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--api-key", required=True, help="API ключ для LLM (polza.ai)")
    parser.add_argument("--llm-url", default="https://polza.ai/api", help="URL для LLM генерации")
    parser.add_argument("--llm-model", default="qwen/qwen3-14b", help="Модель генерации")
    parser.add_argument("--embedder-url", default="http://127.0.0.1:1234", help="URL для эмбеддингов (LM Studio)")
    parser.add_argument("--workers", type=int, default=4, help="Параллельные потоки для eval (default: 4)")
    parser.add_argument("--timeout", type=int, default=600, help="Таймаут LLM-запросов (default: 600)")
    parser.add_argument("--only-domain", choices=DOMAINS, help="Только один домен")
    parser.add_argument("--only-reranker", choices=RERANKER_SHORTS, help="Только один reranker (bge|jina|qwen3)")
    parser.add_argument("--dry-run", action="store_true", help="Показать команды без выполнения")
    parser.add_argument("--reset-progress", action="store_true", help="Сбросить прогресс")
    parser.add_argument("--stream", action="store_true",
                        help="Показывать вывод subprocess в реальном времени (видно прогресс)")
    args = parser.parse_args()

    configs = [c for c in BEST_CONFIGS if not args.only_domain or c["domain"] == args.only_domain]
    rerankers = [r for r in RERANKERS if not args.only_reranker or r["short"] == args.only_reranker]

    if args.reset_progress and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
    progress = load_progress()

    eval_tasks = []
    for cfg in configs:
        for rer in rerankers:
            eval_key = f"{cfg['domain']}/{cfg['chunking']}/{cfg['emb_short']}/{cfg['retrieval']}/{rer['short']}"
            eval_tasks.append((cfg, rer, eval_key))

    total = len(eval_tasks)

    print(f"{'='*60}")
    print(f"RERANK ТЕСТ: {total} eval-задач")
    print(f"Конфигурации: {[c['domain'] for c in configs]}")
    print(f"Rerankers:    {[r['short'] for r in rerankers]}")
    print(f"top_k={TOP_K}, rerank_top_n={RERANK_TOP_N}")
    print(f"{'='*60}\n")

    for i, (cfg, rer, eval_key) in enumerate(eval_tasks, 1):
        print(f"\n[{i}/{total}] EVAL: {cfg['domain']} / {cfg['chunking']} / "
              f"{cfg['emb_short']} / {cfg['retrieval']} / rerank={rer['short']}")

        prev = progress.get("eval", {}).get(eval_key, {})
        if prev.get("status") == "done":
            print(f"  SKIP (уже выполнено)")
            continue

        success, message = run_eval(
            cfg, rer,
            api_key=args.api_key,
            llm_url=args.llm_url,
            llm_model=args.llm_model,
            embedder_url=args.embedder_url,
            workers=args.workers,
            timeout=args.timeout,
            dry_run=args.dry_run,
            stream=args.stream,
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
