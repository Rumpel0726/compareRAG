#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тест влияния параметра top_k на качество RAG.

Лучшая конфигурация для каждого домена прогоняется с top_k = 1, 3, 5, 10.
Всего: 3 домена × 4 значения k = 12 eval-задач.
Ingestion не нужен — коллекции уже проиндексированы.

Использование:
    # Dry run — показать все команды без выполнения:
    python scripts/run_topk_test.py --api-key YOUR_KEY --dry-run

    # Запустить всё:
    python scripts/run_topk_test.py --api-key YOUR_KEY

    # Только один домен:
    python scripts/run_topk_test.py --api-key YOUR_KEY --only-domain medical

    # Только одно значение k:
    python scripts/run_topk_test.py --api-key YOUR_KEY --only-k 5

    # Возобновить после сбоя:
    python scripts/run_topk_test.py --api-key YOUR_KEY
"""

import sys
import json
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROGRESS_FILE = PROJECT_ROOT / "topk_progress.json"
BATCH_LOG_DIR = PROJECT_ROOT / "batch_logs"

# --- Лучшие конфигурации ---

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

TOP_K_VALUES = [1, 3, 5, 10]

DOMAINS = [c["domain"] for c in BEST_CONFIGS]


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


def run_eval(cfg: dict, top_k: int,
             api_key: str, llm_url: str, llm_model: str,
             embedder_url: str, workers: int, timeout: int,
             dry_run: bool = False) -> tuple[bool, str]:
    """Запускает evaluation для одной комбинации."""
    cmd = [
        sys.executable, "scripts/run_evaluation.py",
        "--config", cfg["config"],
        "--domain", cfg["domain"],
        "--top-k", str(top_k),
        "--workers", str(workers),
        "--timeout", str(timeout),
        "--embedding-model", cfg["embedding"],
        "--retrieval", cfg["retrieval"],
        "--embedder-url", embedder_url,
        "--llm-url", llm_url,
        "--llm-model", llm_model,
        "--api-key", api_key,
    ]

    task_key = f"topk_{cfg['domain']}_{cfg['chunking']}_{cfg['emb_short']}_{cfg['retrieval']}_k{top_k}"
    return run_command(cmd, task_key, dry_run)


# --- Summary ---

def print_summary(progress: dict):
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ ТАБЛИЦА (top_k тест)")
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
        description="Top-K тест: 3 лучшие конфигурации × 4 значения k = 12 eval-задач",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--api-key", required=True, help="API ключ для LLM (polza.ai)")
    parser.add_argument("--llm-url", default="https://polza.ai/api", help="URL для LLM генерации")
    parser.add_argument("--llm-model", default="qwen/qwen3-14b", help="Модель генерации")
    parser.add_argument("--embedder-url", default="http://127.0.0.1:1234", help="URL для эмбеддингов (LM Studio)")
    parser.add_argument("--workers", type=int, default=4, help="Параллельные потоки для eval (default: 4)")
    parser.add_argument("--timeout", type=int, default=600, help="Таймаут LLM-запросов в секундах (default: 600)")
    parser.add_argument("--only-domain", choices=DOMAINS, help="Только один домен")
    parser.add_argument("--only-k", type=int, choices=TOP_K_VALUES, help="Только одно значение k")
    parser.add_argument("--dry-run", action="store_true", help="Показать команды без выполнения")
    parser.add_argument("--reset-progress", action="store_true", help="Сбросить файл прогресса и начать заново")
    args = parser.parse_args()

    # Фильтрация
    configs = [c for c in BEST_CONFIGS if not args.only_domain or c["domain"] == args.only_domain]
    k_values = [k for k in TOP_K_VALUES if not args.only_k or k == args.only_k]

    # Прогресс
    if args.reset_progress and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
    progress = load_progress()

    # Формируем список задач
    eval_tasks = []
    for cfg in configs:
        for k in k_values:
            eval_key = f"{cfg['domain']}/{cfg['chunking']}/{cfg['emb_short']}/{cfg['retrieval']}/k{k}"
            eval_tasks.append((cfg, k, eval_key))

    total_eval = len(eval_tasks)

    print(f"{'='*60}")
    print(f"TOP-K ТЕСТ: {total_eval} eval-задач")
    print(f"Домены:    {[c['domain'] for c in configs]}")
    print(f"k values:  {k_values}")
    print(f"{'='*60}\n")

    for i, (cfg, k, eval_key) in enumerate(eval_tasks, 1):
        print(f"\n[{i}/{total_eval}] EVAL: {cfg['domain']} / {cfg['chunking']} / "
              f"{cfg['emb_short']} / {cfg['retrieval']} / top_k={k}")

        prev = progress.get("eval", {}).get(eval_key, {})
        if prev.get("status") == "done":
            print(f"  SKIP (уже выполнено)")
            continue

        success, message = run_eval(
            cfg, k,
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
