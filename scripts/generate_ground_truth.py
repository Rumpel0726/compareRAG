# -*- coding: utf-8 -*-
"""
Генерация эталонных ответов (ground truth) для 30 вопросов оценки RAG.

Использует модель liquid/lfm2-24b-a2b через LM Studio для генерации
подробных ответов на основе конкретных страниц из PDF-документов.

Запуск из корня проекта:
    python scripts/generate_ground_truth.py
    python scripts/generate_ground_truth.py --output src/evaluation/ground_truth.json
    python scripts/generate_ground_truth.py --data-dir data/ --lm-url http://127.0.0.1:1234
"""

import sys
import json
import argparse
import time
from pathlib import Path

import pypdf

# Корень проекта (scripts/ -> compareRAG/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.questions import EVAL_QUESTIONS
from src.llm.lm_studio_client import LMStudioClient

# Модель для генерации эталонных ответов (лучше основной RAG-модели)
JUDGE_MODEL = "liquid/lfm2-24b-a2b"
DEFAULT_OUTPUT = PROJECT_ROOT / "src" / "evaluation" / "ground_truth.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def extract_pages_text(pdf_path: Path, page_numbers: list[int]) -> str:
    """
    Извлекает текст с указанных страниц PDF.

    Args:
        pdf_path: Путь к PDF файлу
        page_numbers: Список номеров страниц (1-indexed)

    Returns:
        Объединённый текст со всех указанных страниц.
    """
    parts = []
    try:
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            total_pages = len(reader.pages)
            for page_num in sorted(set(page_numbers)):
                idx = page_num - 1  # pypdf использует 0-indexed
                if 0 <= idx < total_pages:
                    page_text = reader.pages[idx].extract_text()
                    if page_text and page_text.strip():
                        parts.append(f"[Страница {page_num}]\n{page_text.strip()}")
                else:
                    print(f"  Предупреждение: страница {page_num} не существует в {pdf_path.name} ({total_pages} стр.)")
    except Exception as e:
        print(f"  Ошибка чтения {pdf_path.name}: {e}")

    return "\n\n".join(parts)


def build_context_for_question(eq, data_dir: Path) -> str:
    """
    Собирает контекст из всех ожидаемых страниц для вопроса.
    """
    context_parts = []

    for filename, page_nums in eq.expected_pages.items():
        pdf_path = data_dir / filename
        if not pdf_path.exists():
            print(f"  Предупреждение: файл не найден: {pdf_path}")
            continue
        text = extract_pages_text(pdf_path, page_nums)
        if text:
            context_parts.append(text)

    return "\n\n---\n\n".join(context_parts)


def generate_reference_answer(client: LMStudioClient, question: str, context: str) -> str | None:
    """
    Генерирует эталонный ответ на вопрос по предоставленному контексту.
    """
    if not context.strip():
        print("  Предупреждение: пустой контекст, пропуск.")
        return None

    prompt = f"""Ты — эксперт-медик. На основе приведённых фрагментов документа дай исчерпывающий и точный ответ на вопрос.
Отвечай строго по тексту документа — не домысливай. Ответ должен быть конкретным и полным.

Фрагменты документа:
{context}

Вопрос: {question}

/no_think"""

    messages = [
        {
            "role": "system",
            "content": (
                "Ты — медицинский эксперт, который даёт точные ответы на вопросы "
                "строго на основе предоставленного текста документов. "
                "Не добавляй информацию, которой нет в тексте."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        answer = client.generate(
            messages=messages,
            temperature=0.1,
            max_tokens=1000,
        )
        return answer.strip() if answer else None
    except Exception as e:
        print(f"  Ошибка генерации: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Генерация эталонных ответов (ground truth) для evaluation"
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Путь к выходному JSON-файлу (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Директория с PDF-файлами (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--lm-url",
        default="http://127.0.0.1:1234",
        help="URL LM Studio API (default: http://127.0.0.1:1234)",
    )
    parser.add_argument(
        "--model",
        default=JUDGE_MODEL,
        help=f"Модель для генерации (default: {JUDGE_MODEL})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить с последнего сохранённого результата (пропустить уже сгенерированные вопросы)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    data_dir = Path(args.data_dir)

    print(f"Модель:    {args.model}")
    print(f"LM Studio: {args.lm_url}")
    print(f"Data dir:  {data_dir}")
    print(f"Output:    {output_path}")
    print()

    # Инициализируем клиент
    print("Подключение к LM Studio...")
    client = LMStudioClient(
        url=args.lm_url,
        model=args.model,
        temperature=0.1,
        max_tokens=1000,
        timeout=180,
    )
    print("Соединение установлено.\n")

    # Загружаем уже сгенерированные результаты (если --resume)
    existing: dict[str, str] = {}
    if args.resume and output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                saved = json.load(f)
            existing = {rec["question"]: rec["reference_answer"] for rec in saved if rec.get("reference_answer")}
            print(f"Возобновление: найдено {len(existing)} уже готовых ответов.\n")
        except Exception as e:
            print(f"Предупреждение: не удалось загрузить существующие результаты: {e}\n")

    results = []
    total = len(EVAL_QUESTIONS)

    for idx, eq in enumerate(EVAL_QUESTIONS, 1):
        print(f"[{idx:2d}/{total}] {eq.question[:80]}...")

        # Пропускаем, если уже есть ответ
        if args.resume and eq.question in existing:
            print(f"       → пропуск (уже сгенерирован)")
            results.append({
                "index": idx,
                "question": eq.question,
                "reference_answer": existing[eq.question],
            })
            continue

        # Собираем контекст из нужных страниц PDF
        context = build_context_for_question(eq, data_dir)
        if not context:
            print(f"       → пропуск (контекст пуст)")
            results.append({
                "index": idx,
                "question": eq.question,
                "reference_answer": None,
            })
            continue

        # Генерируем эталонный ответ
        t0 = time.time()
        reference_answer = generate_reference_answer(client, eq.question, context)
        elapsed = time.time() - t0

        if reference_answer:
            preview = reference_answer[:120].replace("\n", " ")
            print(f"       → OK ({elapsed:.1f}s): {preview}...")
        else:
            print(f"       → ОШИБКА ({elapsed:.1f}s)")

        results.append({
            "index": idx,
            "question": eq.question,
            "reference_answer": reference_answer,
        })

        # Сохраняем после каждого вопроса (защита от прерывания)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Итоговая статистика
    success = sum(1 for r in results if r["reference_answer"])
    failed = total - success
    print(f"\nГотово: {success}/{total} ответов сгенерировано, {failed} ошибок.")
    print(f"Результаты сохранены в: {output_path}")


if __name__ == "__main__":
    main()
