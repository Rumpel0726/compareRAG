"""
Collect experiment results from log files and write one summary .txt per domain.

Output files (project root):
  results_medical.txt
  results_postgresql.txt
  results_civil_code.txt  (skipped if no experiments found)

Usage:
  python scripts/collect_results.py
  python scripts/collect_results.py --retrieval hybrid --embedding bge-m3 --chunking chonkie sentence recursive --chunk-size 256 512 1024
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
EXPERIMENTS = ROOT / "experiments"

# Strip Loguru timestamp prefix: "2026-03-13 20:49:13 | "
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| ?")


def clean_line(line: str) -> str:
    return TIMESTAMP_RE.sub("", line.rstrip())


def label_from_path(log_path: Path) -> str:
    """Build a human-readable label from path components."""
    # Без rerank: experiments/{domain}/{retrieval}/{embedding}/{chunking}/{chunk_size}/results/{filename}
    # С rerank:   experiments/{domain}/{retrieval}/{embedding}/{chunking}/{chunk_size}/rerank_<X>/results/{filename}
    parts = log_path.relative_to(EXPERIMENTS).parts
    retrieval = parts[1] if len(parts) > 1 else "?"
    embedding = parts[2] if len(parts) > 2 else "?"
    chunking  = parts[3] if len(parts) > 3 else "?"
    chunk_size = parts[4] if len(parts) > 4 else "?"
    rerank = parts[5] if len(parts) > 5 and parts[5].startswith("rerank_") else None
    filename = log_path.stem  # without .log

    tokens = filename.split("_")
    date_seq = "_".join(tokens[-2:]) if len(tokens) >= 2 else filename

    rerank_str = f" | {rerank}" if rerank else ""
    return f"{retrieval} | {embedding} | {chunking}/{chunk_size}{rerank_str} | {date_seq}"


def collect_domain(domain: str, filters: dict) -> list[tuple[str, list[str]]]:
    """Return sorted list of (label, cleaned_lines) for a domain.

    filters: optional keys {retrieval, embedding, chunking, chunk_size} → list[str].
    A folder is included only if it matches ALL provided filters.
    Path structure: experiments/{domain}/{retrieval}/{embedding}/{chunking}/{chunk_size}/results/*.log
    """
    domain_dir = EXPERIMENTS / domain
    if not domain_dir.exists():
        return []

    # Group files by their results folder, keep only the last (alphabetically = highest seq)
    folders: dict[Path, list[Path]] = defaultdict(list)
    for log_file in domain_dir.rglob("results/*.log"):
        parts = log_file.relative_to(EXPERIMENTS).parts
        # parts: domain, retrieval, embedding, chunking, chunk_size, results, filename
        if len(parts) < 6:
            continue
        retrieval, embedding, chunking, chunk_size = parts[1], parts[2], parts[3], parts[4]
        rerank_folder = parts[5] if len(parts) > 5 and parts[5].startswith("rerank_") else None
        if filters.get("retrieval") and retrieval not in filters["retrieval"]:
            continue
        if filters.get("embedding") and embedding not in filters["embedding"]:
            continue
        if filters.get("chunking") and chunking not in filters["chunking"]:
            continue
        if filters.get("chunk_size") and chunk_size not in filters["chunk_size"]:
            continue
        if filters.get("rerank_only") and not rerank_folder:
            continue
        if filters.get("rerank") and (not rerank_folder or rerank_folder not in filters["rerank"]):
            continue
        if filters.get("no_rerank") and rerank_folder:
            continue
        folders[log_file.parent].append(log_file)

    results = []
    for folder, files in folders.items():
        log_file = sorted(files)[-1]  # highest date/seq suffix
        label = label_from_path(log_file)
        lines = [clean_line(l) for l in log_file.read_text(encoding="utf-8").splitlines()]
        # Drop blank leading/trailing lines
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        results.append((label, lines))

    return sorted(results, key=lambda x: x[0])


def write_domain_file(domain: str, experiments: list[tuple[str, list[str]]],
                       output_dir: Path = None) -> None:
    out_dir = output_dir if output_dir else ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_{domain}.txt"

    if not experiments:
        print(f"  No experiments for domain '{domain}' — skipping.")
        return

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"{'=' * 70}\n")
        f.write(f"  DOMAIN: {domain.upper()}  ({len(experiments)} experiment(s))\n")
        f.write(f"{'=' * 70}\n\n")

        for label, lines in experiments:
            f.write(f"{'─' * 70}\n")
            f.write(f"EXPERIMENT: {label}\n")
            f.write(f"{'─' * 70}\n")
            f.write("\n".join(lines))
            f.write("\n\n")

    print(f"  Written: {out_path}  ({len(experiments)} experiment(s))")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect experiment results into per-domain summary files")
    parser.add_argument("--retrieval", nargs="*", help="Filter by retrieval folder name(s), e.g. hybrid semantic")
    parser.add_argument("--embedding", nargs="*", help="Filter by embedding folder name(s), e.g. bge-m3 nomic")
    parser.add_argument("--chunking", nargs="*", help="Filter by chunking folder name(s), e.g. chonkie sentence recursive")
    parser.add_argument("--chunk-size", nargs="*", help="Filter by chunk size folder name(s), e.g. 256 512 1024")
    parser.add_argument("--rerank-only", action="store_true",
                        help="Только эксперименты с reranker (папка rerank_*)")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Исключить эксперименты с reranker")
    parser.add_argument("--rerank", nargs="*",
                        help="Конкретные rerank-папки, например: rerank_bge-rerank-v2 rerank_jina-rerank-v2")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Папка для записи results_*.txt (default: корень проекта)")
    args = parser.parse_args()

    filters = {
        "retrieval": args.retrieval,
        "embedding": args.embedding,
        "chunking": args.chunking,
        "chunk_size": args.chunk_size,
        "rerank_only": args.rerank_only,
        "no_rerank": args.no_rerank,
        "rerank": args.rerank,
    }
    output_dir = Path(args.output_dir) if args.output_dir else None
    active = {k: v for k, v in filters.items() if v}
    if active:
        print(f"Filters: {active}")
    if output_dir:
        print(f"Output dir: {output_dir}")
    print("Collecting experiment results...")
    for domain in ("medical", "postgresql", "civil_code"):
        experiments = collect_domain(domain, filters)
        write_domain_file(domain, experiments, output_dir=output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
