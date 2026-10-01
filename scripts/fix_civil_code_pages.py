# -*- coding: utf-8 -*-
"""
Верификация и исправление номеров страниц в ground truth для civil_code.

Читает PDF, проверяет reference_answer каждого вопроса против реальных страниц,
выводит таблицу расхождений и перезаписывает questions_civil_code.py
со всеми 30 вопросами (раскомментированными) и исправленными страницами.

Запуск из корня проекта:
    python scripts/fix_civil_code_pages.py --dry-run     # только проверка
    python scripts/fix_civil_code_pages.py               # проверка + исправление
"""

import re
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import pypdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PDF = PROJECT_ROOT / "data" / "Civil_Code" / "gkodeksrf.pdf"
DEFAULT_GT = PROJECT_ROOT / "src" / "evaluation" / "ground_truth_civil_code.json"
DEFAULT_QUESTIONS_PY = PROJECT_ROOT / "src" / "evaluation" / "questions_civil_code.py"

# Количество лёгких вопросов (для комментариев в коде)
EASY_COUNT = 20


# ---------------------------------------------------------------------------
# PDF loading
# ---------------------------------------------------------------------------

def load_all_pages(pdf_path: Path) -> dict[int, str]:
    """Загружает все страницы PDF. Возвращает {page_num (1-indexed): raw_text}."""
    pages: dict[int, str] = {}
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages[i + 1] = text.strip()
    return pages


# ---------------------------------------------------------------------------
# Improved page finding
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Нормализация: нижний регистр, коллапс пробелов."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_word_ngrams(text: str, n: int = 3) -> list[str]:
    """
    Извлекает все последовательности из n слов (word n-grams) из текста.
    Слова длиной < 3 символов пропускаются.
    """
    norm = normalize(text)
    words = [w for w in norm.split() if len(w) >= 3]
    if len(words) < n:
        return words if words else []
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def find_pages_improved(
    all_pages: dict[int, str],
    answer: str,
    is_medium: bool = False,
) -> tuple[list[int], int]:
    """
    Ищет страницы PDF, содержащие фрагменты из reference_answer.

    Использует word 3-grams из reference_answer и считает, сколько из них
    встречается на каждой странице. Возвращает страницы с наивысшим score.

    Для лёгких вопросов (1 страница) берёт только top-1.
    Для средних (2-3 страницы) берёт страницы с score >= 50% от лучшего.
    """
    if not answer or len(answer.strip()) < 10:
        return [], 0

    answer_wngrams = set(extract_word_ngrams(answer, n=3))
    if not answer_wngrams:
        return [], 0

    # Считаем score для каждой страницы
    page_scores: Counter = Counter()
    for pn, text in all_pages.items():
        norm_page = normalize(text)
        hits = sum(1 for wng in answer_wngrams if wng in norm_page)
        if hits > 0:
            page_scores[pn] = hits

    if not page_scores:
        return [], 0

    max_score = max(page_scores.values())

    if is_medium:
        # Средние вопросы: берём страницы с >= 30% от лучшего score
        threshold = max(2, int(max_score * 0.3))
        result = sorted(pn for pn, s in page_scores.items() if s >= threshold)
    else:
        # Лёгкие вопросы: берём top-1 (или top-2 если score одинаковый)
        threshold = max(2, int(max_score * 0.7))
        result = sorted(pn for pn, s in page_scores.items() if s >= threshold)

    return result, max_score


# ---------------------------------------------------------------------------
# Load old pages from questions_civil_code.py (for comparison)
# ---------------------------------------------------------------------------

def load_old_pages_from_py(path: Path) -> dict[int, list[int]]:
    """
    Парсит expected_pages из questions_civil_code.py (включая закомментированные).
    Возвращает {index (1-based): [pages]}.
    """
    if not path.exists():
        return {}

    content = path.read_text(encoding="utf-8")
    result: dict[int, list[int]] = {}
    idx = 0

    # Ищем все вхождения "gkodeksrf.pdf": [...] — как активные, так и закомментированные
    for match in re.finditer(r'#?\s*"gkodeksrf\.pdf":\s*\[([^\]]*)\]', content):
        idx += 1
        pages_str = match.group(1).strip()
        if pages_str:
            pages = [int(x.strip()) for x in pages_str.split(",") if x.strip()]
        else:
            pages = []
        result[idx] = pages

    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(results: list[dict]) -> None:
    """Печатает таблицу результатов."""
    print(f"\n{'Idx':>3} | {'Старые стр.':<20} | {'Новые стр.':<20} | {'Хиты':<6} | Статус")
    print("-" * 80)
    for r in results:
        old = str(r["old_pages"]) if r["old_pages"] else "—"
        new = str(r["new_pages"])
        hits = r.get("max_hits", 0)
        status = r["status"]
        print(f"{r['index']:>3} | {old:<20} | {new:<20} | {hits:<6} | {status}")


def save_questions_py(
    gt_records: list[dict],
    verified: list[dict],
    path: Path,
    pdf_name: str = "gkodeksrf.pdf",
) -> None:
    """Генерирует questions_civil_code.py с исправленными страницами для всех 30 вопросов."""

    pages_map = {v["index"]: v["new_pages"] for v in verified}

    lines = [
        "# -*- coding: utf-8 -*-",
        '"""',
        "Тестовые вопросы для оценки RAG на Гражданском кодексе РФ.",
        "",
        "Сгенерированы автоматически через scripts/generate_civil_code_questions.py.",
        "Страницы верифицированы скриптом scripts/fix_civil_code_pages.py.",
        f"Документ: data/Civil_Code/{pdf_name}",
        f"{EASY_COUNT} лёгких вопросов (1 страница) + {len(gt_records) - EASY_COUNT} средних (2-3 страницы).",
        '"""',
        "",
        "from src.evaluation.questions import EvalQuestion",
        "",
        "",
        "EVAL_QUESTIONS_CIVIL_CODE: list[EvalQuestion] = [",
        "",
    ]

    for rec in gt_records:
        idx = rec["index"]
        pages = pages_map.get(idx, [])
        difficulty = "Лёгкий" if idx <= EASY_COUNT else "Средний"
        question_escaped = json.dumps(rec["question"], ensure_ascii=False)
        pages_repr = repr(sorted(pages))

        lines += [
            f"    # {difficulty} — вопрос {idx}",
            "    EvalQuestion(",
            f"        question={question_escaped},",
            f"        expected_sources=[{json.dumps(pdf_name)}],",
            "        expected_pages={",
            f"            {json.dumps(pdf_name)}: {pages_repr},",
            "        }",
            "    ),",
            "",
        ]

    lines += ["]", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Верификация страниц civil_code ground truth")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument("--gt", default=str(DEFAULT_GT))
    parser.add_argument("--questions-py", default=str(DEFAULT_QUESTIONS_PY))
    parser.add_argument("--dry-run", action="store_true", help="Только показать, не перезаписывать")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    gt_path = Path(args.gt)
    questions_py_path = Path(args.questions_py)

    if not pdf_path.exists():
        print(f"Ошибка: PDF не найден: {pdf_path}")
        sys.exit(1)

    if not gt_path.exists():
        print(f"Ошибка: ground truth не найден: {gt_path}")
        sys.exit(1)

    # Загрузка PDF
    print(f"PDF: {pdf_path}")
    print("Загрузка PDF...")
    all_pages = load_all_pages(pdf_path)
    print(f"Страниц: {max(all_pages.keys())}")

    # Загрузка ground truth (все 30 вопросов)
    with open(gt_path, encoding="utf-8") as f:
        gt_records = json.load(f)
    print(f"Вопросов в ground truth: {len(gt_records)}")

    # Загрузка старых страниц из questions_civil_code.py
    old_pages_map = load_old_pages_from_py(questions_py_path)
    print(f"Вопросов с текущими страницами: {len(old_pages_map)}")

    # Верификация
    print(f"\nВерификация (word 3-grams)...\n")

    results = []
    fixed = 0
    not_found = 0

    for rec in gt_records:
        idx = rec["index"]
        answer = rec.get("reference_answer", "")
        old_pages = old_pages_map.get(idx, [])
        is_medium = idx > EASY_COUNT

        new_pages, max_hits = find_pages_improved(
            all_pages, answer,
            is_medium=is_medium,
        )

        if not new_pages:
            status = "NOT_FOUND"
            not_found += 1
            use_pages = old_pages
        elif sorted(old_pages) == sorted(new_pages):
            status = "OK"
            use_pages = new_pages
        else:
            status = "FIXED"
            fixed += 1
            use_pages = new_pages

        results.append({
            "index": idx,
            "old_pages": old_pages,
            "new_pages": use_pages,
            "max_hits": max_hits,
            "status": status,
        })

    print_table(results)

    ok_count = sum(1 for r in results if r["status"] == "OK")
    print(f"\nИтого: {ok_count} OK, {fixed} FIXED, {not_found} NOT_FOUND из {len(results)}")

    if args.dry_run:
        print("\n[dry-run] Файл не перезаписан.")
    else:
        print(f"\nОбновляю {questions_py_path}...")
        save_questions_py(gt_records, results, questions_py_path)
        print(f"Готово: все {len(gt_records)} вопросов записаны с исправленными страницами.")


if __name__ == "__main__":
    main()
