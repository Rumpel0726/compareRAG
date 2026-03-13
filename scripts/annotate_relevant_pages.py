# -*- coding: utf-8 -*-
"""
Автоматическая аннотация релевантных страниц для ground truth с использованием LLM.

Скрипт анализирует каждую страницу документа и определяет, содержит ли она
информацию для ответа на конкретный вопрос.

Запуск:
    python scripts/annotate_relevant_pages.py --config configs/strategies/baseline.yaml
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

from loguru import logger

# Корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.document_processing.pdf_loader import PDFLoader
from src.llm.lm_studio_client import LMStudioClient
from src.evaluation.questions import EVAL_QUESTIONS

# Промпт для определения релевантности страницы
RELEVANCE_CHECK_PROMPT = """Ты - эксперт по анализу научных медицинских текстов.

ЗАДАЧА: Определи, содержит ли данная страница документа информацию, необходимую для ответа на вопрос.

ВОПРОС:
{question}

ТЕКСТ СТРАНИЦЫ {page_num} ИЗ ДОКУМЕНТА "{doc_name}":
{page_text}

ИНСТРУКЦИЯ:
1. Внимательно прочитай вопрос и текст страницы
2. Определи, содержится ли на этой странице информация, которая НАПРЯМУЮ отвечает на вопрос
3. Страница релевантна, если содержит:
   - Прямой ответ на вопрос
   - Ключевые факты, методы, результаты, упомянутые в вопросе
   - Данные, необходимые для полноценного ответа
4. Страница НЕ релевантна, если содержит только:
   - Общую информацию, не связанную с вопросом
   - Косвенные упоминания без деталей
   - Библиографию, титульные данные, оглавление

ОТВЕТ (строго JSON, БЕЗ тегов <think> и дополнительного текста):
{{
  "relevant": true/false,
  "reason": "краткое объяснение (1-2 предложения)"
}}

КРИТИЧЕСКИ ВАЖНО:
- НЕ используй теги <think></think>
- Отвечай ТОЛЬКО валидным JSON
- Никакого дополнительного текста до или после JSON"""


def extract_page_texts(pdf_path: Path) -> Dict[int, str]:
    """Извлекает текст каждой страницы PDF."""
    loader = PDFLoader()

    try:
        document = loader.load(pdf_path)
        page_texts = {}

        # Разбиваем документ по маркерам страниц
        content = document.text  # Исправлено: text вместо content
        pages = content.split("[Страница ")

        for page_part in pages[1:]:  # Пропускаем первый пустой элемент
            try:
                # Извлекаем номер страницы
                page_num_str, page_text = page_part.split("]", 1)
                page_num = int(page_num_str.strip())
                page_texts[page_num] = page_text.strip()
            except Exception as e:
                logger.warning(f"Не удалось распарсить страницу: {e}")
                continue

        logger.info(f"Извлечено {len(page_texts)} страниц из {pdf_path.name}")
        return page_texts

    except Exception as e:
        logger.error(f"Ошибка при загрузке PDF {pdf_path}: {e}")
        return {}


def check_page_relevance(
    llm_client: LMStudioClient,
    question: str,
    doc_name: str,
    page_num: int,
    page_text: str
) -> tuple[bool, str]:
    """Проверяет релевантность страницы для вопроса с помощью LLM."""

    # Ограничиваем длину текста страницы (чтобы не превысить контекст)
    max_chars = 2000
    if len(page_text) > max_chars:
        page_text = page_text[:max_chars] + "\n...[текст обрезан]..."

    prompt = RELEVANCE_CHECK_PROMPT.format(
        question=question,
        doc_name=doc_name,
        page_num=page_num,
        page_text=page_text
    )

    try:
        response = llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Низкая температура для более детерминированных ответов
            max_tokens=500  # Увеличено с 200 до 500 для гарантии полного ответа
        )

        # Пытаемся распарсить JSON из ответа
        response_text = response.strip()

        # Удаляем теги <think>...</think> (внутренние рассуждения qwen3)
        import re
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        response_text = response_text.strip()

        # Ищем JSON в ответе (может быть обернут в ```json```)
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()

        result = json.loads(response_text)
        relevant = result.get("relevant", False)
        reason = result.get("reason", "Нет объяснения")

        return relevant, reason

    except json.JSONDecodeError as e:
        logger.warning(f"Не удалось распарсить JSON ответ для страницы {page_num}: {e}")
        logger.debug(f"Ответ LLM: {response}")
        return False, "Ошибка парсинга ответа LLM"

    except Exception as e:
        logger.error(f"Ошибка при проверке релевантности страницы {page_num}: {e}")
        return False, f"Ошибка: {str(e)}"


def annotate_question(
    llm_client: LMStudioClient,
    question: str,
    expected_sources: List[str],
    data_dir: Path
) -> Dict[str, List[int]]:
    """Аннотирует релевантные страницы для одного вопроса."""

    relevant_pages = {}

    for source in expected_sources:
        pdf_path = data_dir / source

        if not pdf_path.exists():
            logger.error(f"Файл не найден: {pdf_path}")
            continue

        logger.info(f"Обрабатываю документ: {source}")

        # Извлекаем текст страниц
        page_texts = extract_page_texts(pdf_path)

        if not page_texts:
            logger.warning(f"Не удалось извлечь страницы из {source}")
            continue

        # Проверяем каждую страницу
        relevant_page_nums = []

        for page_num in sorted(page_texts.keys()):
            page_text = page_texts[page_num]

            logger.info(f"  Проверяю страницу {page_num}...")
            is_relevant, reason = check_page_relevance(
                llm_client=llm_client,
                question=question,
                doc_name=source,
                page_num=page_num,
                page_text=page_text
            )

            if is_relevant:
                relevant_page_nums.append(page_num)
                logger.info(f"    ✓ Релевантна: {reason}")
            else:
                logger.debug(f"    ✗ Не релевантна: {reason}")

        if relevant_page_nums:
            relevant_pages[source] = relevant_page_nums
            logger.info(f"  Найдено {len(relevant_page_nums)} релевантных страниц: {relevant_page_nums}")
        else:
            logger.warning(f"  Не найдено релевантных страниц!")

    return relevant_pages


def main():
    parser = argparse.ArgumentParser(
        description="Автоматическая аннотация релевантных страниц с помощью LLM"
    )
    parser.add_argument(
        "--config",
        default="configs/strategies/baseline.yaml",
        help="Путь к конфигурации"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Директория с PDF файлами"
    )
    parser.add_argument(
        "--output",
        default="logs/annotated_pages.json",
        help="Путь для сохранения результатов"
    )
    parser.add_argument(
        "--question-ids",
        type=str,
        help="ID вопросов для аннотации (например: 1,2,3). По умолчанию - все вопросы"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API ключ для внешних провайдеров LLM."
    )
    args = parser.parse_args()

    # Загружаем конфигурацию
    config_path = PROJECT_ROOT / args.config
    config = load_config(str(config_path))

    logger.info(f"Конфигурация: {config_path}")
    logger.info(f"LLM модель: {config.lm_studio.llm_model}")

    # Инициализируем LLM клиент
    llm_client = LMStudioClient(
        url=config.lm_studio.url,
        model=config.lm_studio.llm_model,
        temperature=0.1,
        max_tokens=200,
        timeout=120,
        api_key=args.api_key or config.lm_studio.api_key
    )

    data_dir = PROJECT_ROOT / args.data_dir

    # Фильтруем вопросы, если указаны конкретные ID
    questions_to_process = EVAL_QUESTIONS
    if args.question_ids:
        ids = [int(x.strip()) for x in args.question_ids.split(",")]
        questions_to_process = [EVAL_QUESTIONS[i-1] for i in ids if 0 < i <= len(EVAL_QUESTIONS)]
        logger.info(f"Обрабатываю только вопросы: {ids}")

    # Результаты аннотации
    annotations = {}

    logger.info("=" * 60)
    logger.info(f"НАЧАЛО АННОТАЦИИ: {len(questions_to_process)} вопросов")
    logger.info("=" * 60)

    for idx, eq in enumerate(questions_to_process, 1):
        logger.info(f"\n--- Вопрос {idx}/{len(questions_to_process)} ---")
        logger.info(f"Q: {eq.question}")
        logger.info(f"Expected sources: {eq.expected_sources}")

        relevant_pages = annotate_question(
            llm_client=llm_client,
            question=eq.question,
            expected_sources=eq.expected_sources,
            data_dir=data_dir
        )

        annotations[eq.question] = {
            "expected_sources": eq.expected_sources,
            "relevant_pages": relevant_pages
        }

    # Сохраняем результаты
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info(f"РЕЗУЛЬТАТЫ СОХРАНЕНЫ: {output_path}")
    logger.info("=" * 60)

    # Выводим результаты в формате для копирования в questions.py
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ДЛЯ questions.py:")
    print("=" * 60)

    for idx, eq in enumerate(questions_to_process, 1):
        annotation = annotations[eq.question]
        relevant_pages = annotation["relevant_pages"]

        print(f"\n# Вопрос {idx}")
        print(f"# Q: {eq.question[:80]}...")
        print("EvalQuestion(")
        print(f'    question="{eq.question}",')
        print(f"    expected_sources={eq.expected_sources},")
        print("    expected_pages={")
        for source, pages in relevant_pages.items():
            print(f'        "{source}": {pages},')
        print("    }")
        print("),")

    print("\n" + "=" * 60)
    logger.success("Аннотация завершена!")


if __name__ == "__main__":
    main()
