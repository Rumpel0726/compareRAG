# -*- coding: utf-8 -*-
"""
Утилита для извлечения номеров страниц из текста чанков.

PDFLoader добавляет маркеры [Страница N] при загрузке PDF документов.
Эта утилита извлекает номер страницы из текста чанка.
"""

import re
from typing import Optional, List, Tuple


def extract_page_number(text: str) -> Optional[int]:
    """
    Извлекает номер страницы из текста чанка.

    Ищет маркер формата [Страница N] в тексте.
    PDFLoader добавляет эти маркеры при загрузке (pdf_loader.py:139).

    Args:
        text: Текст чанка

    Returns:
        Номер страницы или None если не найден

    Examples:
        >>> extract_page_number("\\n[Страница 5]\\nТекст страницы...")
        5
        >>> extract_page_number("[Страница 123]\\nДругой текст...")
        123
        >>> extract_page_number("Текст без маркера")
        None
    """
    # Паттерн: [Страница N] с учетом возможных пробелов
    pattern = r'\[Страница\s+(\d+)\]'
    match = re.search(pattern, text)

    if match:
        return int(match.group(1))

    return None


def extract_page_ranges(full_text: str) -> List[Tuple[int, int]]:
    """
    Извлекает все маркеры страниц из полного текста документа.

    Args:
        full_text: Полный текст документа с маркерами

    Returns:
        Список кортежей (page_number, start_position)

    Examples:
        >>> text = "[Страница 1]\\nТекст...\\n[Страница 2]\\nЕще текст..."
        >>> extract_page_ranges(text)
        [(1, 0), (2, 30)]
    """
    pattern = r'\[Страница\s+(\d+)\]'
    ranges = []

    for match in re.finditer(pattern, full_text):
        page_num = int(match.group(1))
        position = match.start()
        ranges.append((page_num, position))

    return ranges


def determine_page_number(chunk_start: int, chunk_text: str, page_ranges: List[Tuple[int, int]]) -> Optional[int]:
    """
    Определяет номер страницы для чанка на основе его позиции в документе.

    Args:
        chunk_start: Начальная позиция чанка в полном тексте документа
        chunk_text: Текст чанка
        page_ranges: Список (page_number, start_position) из extract_page_ranges()

    Returns:
        Номер страницы для этого чанка
    """
    if not page_ranges:
        return None

    # Сначала проверим есть ли маркер прямо в чанке
    direct_match = extract_page_number(chunk_text)
    if direct_match is not None:
        return direct_match

    # Найдем последний маркер страницы перед позицией чанка
    current_page = None
    for page_num, pos in page_ranges:
        if pos <= chunk_start:
            current_page = page_num
        else:
            break

    return current_page
