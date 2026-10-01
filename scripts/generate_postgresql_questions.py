# -*- coding: utf-8 -*-
"""
Генерация вопросов + эталонных ответов (ground truth) для документации PostgreSQL.

Два уровня сложности:
  - Лёгкие (20): 1 страница → 1 вопрос. Ответ полностью на одной странице.
  - Средние (10): 3 соседних страницы → 1 вопрос. Ответ задействует несколько разделов.

Ключевое: страницы передаются с реальными номерами [Страница N],
поэтому модель возвращает корректные номера страниц по конструкции.

Текст документа на английском, но вопросы генерируются на русском языке.

Запуск из корня проекта:
    python scripts/generate_postgresql_questions.py --api-key <ключ>
    python scripts/generate_postgresql_questions.py  # ключ из POLZA_AI_API_KEY
    python scripts/generate_postgresql_questions.py --resume
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path

import pypdf
from openai import OpenAI

# ---------------------------------------------------------------------------
# Page search — finds actual PDF pages where the answer text appears
# ---------------------------------------------------------------------------

def find_pages_by_answer(all_pages: dict[int, str], answer: str, ngram_len: int = 40) -> list[int]:
    """
    Ищет страницы PDF, содержащие ключевую фразу из answer.
    Использует один длинный (40+ симв.) фрагмент из середины ответа.
    Возвращает отсортированный список номеров страниц.
    """
    if not answer or len(answer) < ngram_len // 2:
        return []

    norm_answer = re.sub(r"\s+", " ", answer.lower()).strip()
    length = len(norm_answer)

    # Берём фрагмент из середины ответа — самая специфичная часть
    mid = length // 2
    start = max(0, mid - ngram_len // 2)
    end = min(length, start + ngram_len)
    phrase = norm_answer[start:end].strip()

    if len(phrase) < 10:
        return []

    found = []
    for pn, text in all_pages.items():
        norm_page = re.sub(r"\s+", " ", text.lower())
        if phrase in norm_page:
            found.append(pn)

    return sorted(found)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL = "qwen/qwen3-235b-a22b"
POLZA_BASE_URL = "https://polza.ai/api/v1"
DEFAULT_PDF = PROJECT_ROOT / "data" / "PostrgreSQL" / "postgres_part1_2.pdf"
DEFAULT_OUTPUT_QUESTIONS = PROJECT_ROOT / "src" / "evaluation" / "questions_postgresql.py"
DEFAULT_OUTPUT_GT = PROJECT_ROOT / "src" / "evaluation" / "ground_truth_postgresql.json"


# ---------------------------------------------------------------------------
# PDF utilities
# ---------------------------------------------------------------------------

def load_all_pages(pdf_path: Path) -> dict[int, str]:
    """Загружает все страницы PDF. Возвращает {page_num (1-indexed): text}."""
    pages: dict[int, str] = {}
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages[i + 1] = text.strip()
    return pages


def select_easy_pages(pages: dict[int, str], count: int, min_chars: int) -> list[int]:
    """
    Выбирает `count` отдельных страниц, равномерно распределённых по документу.
    Фильтрует страницы с текстом < min_chars.
    """
    good = [pn for pn, text in pages.items() if len(text) >= min_chars]
    if not good:
        return []
    step = max(1, len(good) // count)
    selected = []
    for i in range(count):
        idx = min(i * step + step // 2, len(good) - 1)
        selected.append(good[idx])
    return selected[:count]


def select_medium_groups(
    pages: dict[int, str],
    count: int,
    group_size: int,
    min_chars: int,
    easy_pages: set[int],
) -> list[list[int]]:
    """
    Выбирает `count` групп из `group_size` последовательных страниц.
    Все страницы в группе должны быть "хорошими".
    Позиции смещены относительно лёгких вопросов.
    """
    good = set(pn for pn, text in pages.items() if len(text) >= min_chars)
    total = max(pages.keys())

    # Строим список стартовых позиций: берём середину каждой из count секций + сдвиг
    section_size = total // count
    groups: list[list[int]] = []
    for i in range(count):
        # Сдвиг на 50% шага относительно лёгких
        mid = i * section_size + section_size * 3 // 4
        mid = min(mid, total - group_size)
        mid = max(1, mid)

        # Ищем ближайшую позицию где все group_size страниц "хорошие"
        found = False
        for offset in range(0, section_size // 2):
            for direction in (0, 1, -1):
                start = mid + offset * direction
                group = list(range(start, start + group_size))
                if all(p in good for p in group):
                    groups.append(group)
                    found = True
                    break
            if found:
                break
        if not found:
            # Fallback: просто берём mid..mid+group_size
            groups.append(list(range(mid, mid + group_size)))

    return groups[:count]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Ты — эксперт по PostgreSQL и базам данных. "
    "Генерируй вопросы на русском языке для оценки RAG-системы. "
    "Текст документации на английском, но вопросы и ответы должны быть на русском. "
    "Отвечай строго валидным JSON без markdown-блоков и без тегов <think>."
)


def build_easy_prompt(page_num: int, page_text: str, covered_topics: list[str]) -> str:
    topics_str = ", ".join(covered_topics) if covered_topics else "нет"
    return (
        f"Вот текст со страницы {page_num} документации PostgreSQL:\n"
        f"---\n{page_text}\n---\n\n"
        f"Сгенерируй один вопрос на русском языке, ответ на который полностью содержится в этом тексте.\n"
        f"Текст документации на английском, но вопрос и ответ должны быть на русском.\n"
        f"Вопрос должен звучать естественно, как будто его задаёт обычный человек.\n"
        f"Ответ должен быть конкретным: команда, параметр, тип данных, определение или правило.\n\n"
        f"Уже использованные темы (не повторять): {topics_str}\n\n"
        f'Ответ строго JSON:\n{{"question": "...", "answer": "...", "pages": [{page_num}], "topic": "2-4 слова"}}'
    )


def build_medium_prompt(group: list[int], pages: dict[int, str], covered_topics: list[str]) -> str:
    topics_str = ", ".join(covered_topics) if covered_topics else "нет"
    start, end = group[0], group[-1]
    pages_text = "\n\n".join(
        f"[Страница {pn}]\n{pages.get(pn, '').strip()}"
        for pn in group
    )
    pages_list = str(group)
    return (
        f"Вот текст страниц {start}–{end} документации PostgreSQL:\n"
        f"---\n{pages_text}\n---\n\n"
        f"Сгенерируй один вопрос на русском языке, ответ на который задействует информацию "
        f"из нескольких приведённых страниц или разделов.\n"
        f"Текст документации на английском, но вопрос и ответ должны быть на русском.\n"
        f"Вопрос должен звучать естественно. Ответ должен опираться на 2–3 страницы.\n\n"
        f"Уже использованные темы (не повторять): {topics_str}\n\n"
        f"Ответ строго JSON:\n"
        f'{{"question": "...", "answer": "...", "pages": {pages_list}, "topic": "2-4 слова"}}'
    )


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def parse_response(raw: str) -> dict | None:
    """JSON из ответа: прямой парсинг → regex → None."""
    raw = raw.strip()
    # Убираем <think>...</think>
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Убираем markdown ```json ... ```
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE).strip()

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return None


def call_api(
    client: OpenAI,
    model: str,
    user_prompt: str,
    sent_pages: list[int],
    max_retries: int = 2,
) -> dict | None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    sent_set = set(sent_pages)

    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=1000,
            )
            raw = completion.choices[0].message.content or ""
            data = parse_response(raw)
            if data and data.get("question"):
                # Валидируем страницы: только те, что были отправлены
                returned_pages = data.get("pages", [])
                if isinstance(returned_pages, list):
                    valid_pages = [p for p in returned_pages if isinstance(p, int) and p in sent_set]
                    data["pages"] = valid_pages if valid_pages else sent_pages
                else:
                    data["pages"] = sent_pages
                return data
            print(f"    Предупреждение: не удалось распарсить (попытка {attempt + 1})")
            if raw:
                print(f"    Ответ: {raw[:200]}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"    Ошибка API (попытка {attempt + 1}): {e}")
            if attempt < max_retries:
                wait = 3 * (2 ** attempt)
                print(f"    Повтор через {wait}с...")
                time.sleep(wait)

    return None


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def save_gt_json(results: list[dict], path: Path) -> None:
    records = [
        {"index": r["index"], "question": r["question"], "reference_answer": r.get("answer")}
        for r in results if r.get("question")
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def save_questions_py(results: list[dict], path: Path, pdf_name: str = "postgres_part1_2.pdf") -> None:
    lines = [
        "# -*- coding: utf-8 -*-",
        '"""',
        "Тестовые вопросы для оценки RAG на документации PostgreSQL.",
        "",
        "Сгенерированы автоматически через scripts/generate_postgresql_questions.py.",
        f"Документ: data/PostrgreSQL/{pdf_name}",
        "20 лёгких вопросов (1 страница) + 10 средних (2-3 страницы).",
        "Номера страниц гарантированно корректны: модель видела реальные [Страница N].",
        '"""',
        "",
        "from src.evaluation.questions import EvalQuestion",
        "",
        "",
        "EVAL_QUESTIONS_POSTGRESQL: list[EvalQuestion] = [",
        "",
    ]

    for r in results:
        if not r.get("question"):
            continue
        difficulty = r.get("difficulty", "лёгкий")
        topic = r.get("topic", "")
        label = f"Лёгкий" if difficulty == "easy" else "Средний"
        comment = f"    # {label}" + (f" — {topic}" if topic else "")
        pages = sorted(set(r.get("pages", [])))
        pages_repr = repr(pages)
        lines += [
            comment,
            "    EvalQuestion(",
            f'        question={json.dumps(r["question"], ensure_ascii=False)},',
            f'        expected_sources=[{json.dumps(pdf_name)}],',
            "        expected_pages={",
            f'            {json.dumps(pdf_name)}: {pages_repr},',
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
    parser = argparse.ArgumentParser(
        description="Генерация вопросов по документации PostgreSQL: 20 лёгких + 10 средних"
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument("--output-questions", default=str(DEFAULT_OUTPUT_QUESTIONS))
    parser.add_argument("--output-gt", default=str(DEFAULT_OUTPUT_GT))
    parser.add_argument("--easy-count", type=int, default=20)
    parser.add_argument("--medium-count", type=int, default=10)
    parser.add_argument("--medium-pages", type=int, default=3, help="Страниц на средний вопрос")
    parser.add_argument("--min-page-chars", type=int, default=300, help="Мин. символов на странице")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--delay", type=float, default=1.5)
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_questions = Path(args.output_questions)
    out_gt = Path(args.output_gt)

    api_key = args.api_key or os.environ.get("POLZA_AI_API_KEY") or ""
    if not api_key:
        print("Ошибка: не задан API ключ. Используйте --api-key или POLZA_AI_API_KEY.")
        sys.exit(1)

    if not pdf_path.exists():
        print(f"Ошибка: PDF не найден: {pdf_path}")
        sys.exit(1)

    print(f"Модель:  {args.model}")
    print(f"PDF:     {pdf_path}")
    print(f"Лёгких: {args.easy_count}, Средних: {args.medium_count} (по {args.medium_pages} стр.)")
    print()

    print("Загрузка PDF...")
    all_pages = load_all_pages(pdf_path)
    total_pages = max(all_pages.keys())
    good_count = sum(1 for t in all_pages.values() if len(t) >= args.min_page_chars)
    print(f"Страниц: {total_pages}, с текстом >= {args.min_page_chars} симв.: {good_count}\n")

    # Выбор страниц
    easy_page_nums = select_easy_pages(all_pages, args.easy_count, args.min_page_chars)
    medium_groups = select_medium_groups(
        all_pages, args.medium_count, args.medium_pages,
        args.min_page_chars, set(easy_page_nums)
    )

    print(f"Лёгкие страницы:  {easy_page_nums}")
    print(f"Средние группы:   {medium_groups}\n")

    # Resume
    results: list[dict] = []
    start_from = 0
    if args.resume and out_gt.exists():
        try:
            saved = json.load(open(out_gt, encoding="utf-8"))
            for rec in saved:
                results.append({
                    "index": rec["index"],
                    "question": rec["question"],
                    "answer": rec["reference_answer"],
                    "pages": [],
                    "topic": "",
                    "difficulty": "easy" if rec["index"] <= args.easy_count else "medium",
                })
            start_from = len(results)
            print(f"Возобновление: {start_from} уже готовых вопросов.\n")
        except Exception as e:
            print(f"Предупреждение: не удалось загрузить checkpoint: {e}\n")

    client = OpenAI(base_url=POLZA_BASE_URL, api_key=api_key, timeout=60)
    covered_topics: list[str] = [r.get("topic", "") for r in results if r.get("topic")]
    pdf_name = pdf_path.name
    failed = 0

    try:
        # --- Лёгкие вопросы ---
        for i, page_num in enumerate(easy_page_nums):
            idx = i + 1
            if idx <= start_from:
                print(f"[{idx:2d}/30] Лёгкий стр.{page_num} — пропуск (resume)")
                continue

            page_text = all_pages.get(page_num, "")
            print(f"[{idx:2d}/30] Лёгкий — стр.{page_num} ({len(page_text)} симв.)...", end=" ", flush=True)

            prompt = build_easy_prompt(page_num, page_text, covered_topics)
            t0 = time.time()
            data = call_api(client, args.model, prompt, sent_pages=[page_num])
            elapsed = time.time() - t0

            if data:
                topic = data.get("topic", "")
                actual_pages = find_pages_by_answer(all_pages, data.get("answer", ""))
                if actual_pages:
                    used_pages = actual_pages
                    marker = f"(стр.найдены: {actual_pages})"
                else:
                    used_pages = data["pages"]
                    marker = f"(стр.модель: {data['pages']})"
                covered_topics.append(topic)
                results.append({
                    "index": idx,
                    "question": data["question"],
                    "answer": data.get("answer", ""),
                    "pages": used_pages,
                    "topic": topic,
                    "difficulty": "easy",
                })
                print(f"OK ({elapsed:.1f}s) {marker} — {data['question'][:60]}")
            else:
                print(f"ОШИБКА ({elapsed:.1f}s)")
                results.append({"index": idx, "question": None, "answer": None, "pages": [page_num], "topic": "", "difficulty": "easy"})
                failed += 1

            save_gt_json(results, out_gt)
            save_questions_py(results, out_questions, pdf_name=pdf_name)

            if idx < args.easy_count + args.medium_count:
                time.sleep(args.delay)

        # --- Средние вопросы ---
        for i, group in enumerate(medium_groups):
            idx = args.easy_count + i + 1
            if idx <= start_from:
                print(f"[{idx:2d}/30] Средний стр.{group} — пропуск (resume)")
                continue

            total_chars = sum(len(all_pages.get(p, "")) for p in group)
            print(f"[{idx:2d}/30] Средний — стр.{group} ({total_chars} симв.)...", end=" ", flush=True)

            prompt = build_medium_prompt(group, all_pages, covered_topics)
            t0 = time.time()
            data = call_api(client, args.model, prompt, sent_pages=group)
            elapsed = time.time() - t0

            if data:
                topic = data.get("topic", "")
                actual_pages = find_pages_by_answer(all_pages, data.get("answer", ""))
                if actual_pages:
                    used_pages = actual_pages
                    marker = f"(стр.найдены: {actual_pages})"
                else:
                    used_pages = data["pages"]
                    marker = f"(стр.модель: {data['pages']})"
                covered_topics.append(topic)
                results.append({
                    "index": idx,
                    "question": data["question"],
                    "answer": data.get("answer", ""),
                    "pages": used_pages,
                    "topic": topic,
                    "difficulty": "medium",
                })
                print(f"OK ({elapsed:.1f}s) {marker} — {data['question'][:60]}")
            else:
                print(f"ОШИБКА ({elapsed:.1f}s)")
                results.append({"index": idx, "question": None, "answer": None, "pages": group, "topic": "", "difficulty": "medium"})
                failed += 1

            save_gt_json(results, out_gt)
            save_questions_py(results, out_questions, pdf_name=pdf_name)

            if idx < args.easy_count + args.medium_count:
                time.sleep(args.delay)

    except KeyboardInterrupt:
        print("\n\nПрервано пользователем (Ctrl+C).")
        success = sum(1 for r in results if r.get("question"))
        print(f"Сохранено: {success} вопросов. Используйте --resume для продолжения.")
        sys.exit(1)

    # Итог
    success = sum(1 for r in results if r.get("question"))
    print(f"\nГотово: {success}/30 вопросов, {failed} ошибок.")
    print(f"Вопросы:     {out_questions}")
    print(f"Ground truth: {out_gt}")


if __name__ == "__main__":
    main()
