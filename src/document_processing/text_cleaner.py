# -*- coding: utf-8 -*-
"""
Очистка и нормализация текста для русского языка.
"""

import re
from typing import Optional
from loguru import logger


class TextCleaner:
    """
    Класс для очистки и нормализации текста.
    Оптимизирован для русского языка.
    """

    def __init__(
        self,
        remove_extra_whitespace: bool = True,
        normalize_unicode: bool = True,
        remove_special_chars: bool = False,
        preserve_structure: bool = True,
    ):
        """
        Инициализация очистителя.

        Args:
            remove_extra_whitespace: Удалять лишние пробелы
            normalize_unicode: Нормализовать юникод символы
            remove_special_chars: Удалять специальные символы (осторожно!)
            preserve_structure: Сохранять структуру (абзацы, списки)
        """
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_unicode = normalize_unicode
        self.remove_special_chars = remove_special_chars
        self.preserve_structure = preserve_structure

    def clean(self, text: str) -> str:
        """
        Очищает текст.

        Args:
            text: Исходный текст

        Returns:
            Очищенный текст
        """
        if not text:
            return ""

        cleaned = text

        # Нормализация юникода
        if self.normalize_unicode:
            cleaned = self._normalize_unicode(cleaned)

        # Удаление лишних пробелов
        if self.remove_extra_whitespace:
            cleaned = self._remove_extra_whitespace(cleaned)

        # Удаление специальных символов (опционально)
        if self.remove_special_chars:
            cleaned = self._remove_special_chars(cleaned)

        # Сохранение структуры
        if self.preserve_structure:
            cleaned = self._preserve_structure(cleaned)

        # Финальная очистка
        cleaned = cleaned.strip()

        return cleaned

    def _normalize_unicode(self, text: str) -> str:
        """
        Нормализует юникод символы.

        Заменяет различные варианты дефисов, кавычек и т.д.
        на стандартные символы.
        """
        # Нормализация дефисов и тире
        text = text.replace("—", "-")  # em dash
        text = text.replace("–", "-")  # en dash
        text = text.replace("−", "-")  # minus sign

        # Нормализация кавычек
        text = text.replace("«", '"')
        text = text.replace("»", '"')
        text = text.replace("„", '"')
        text = text.replace(""", '"')
        text = text.replace(""", '"')
        text = text.replace("'", "'")
        text = text.replace("'", "'")

        # Нормализация пробелов
        text = text.replace("\u00a0", " ")  # non-breaking space
        text = text.replace("\u202f", " ")  # narrow non-breaking space
        text = text.replace("\u2009", " ")  # thin space

        return text

    def _remove_extra_whitespace(self, text: str) -> str:
        """
        Удаляет лишние пробелы и переносы строк.
        """
        # Удаляем множественные пробелы
        text = re.sub(r"[ \t]+", " ", text)

        # Удаляем пробелы в начале/конце строк
        lines = [line.strip() for line in text.split("\n")]

        # Удаляем множественные пустые строки (оставляем максимум 2)
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    def _remove_special_chars(self, text: str) -> str:
        """
        Удаляет специальные символы, сохраняя русский и латинский алфавит.

        ВНИМАНИЕ: Может удалить важные символы!
        """
        # Паттерн: оставляем русские и латинские буквы, цифры, базовую пунктуацию
        pattern = r"[^а-яА-ЯёЁa-zA-Z0-9\s.,!?;:()\[\]\-\"/«»]"
        text = re.sub(pattern, "", text)

        return text

    def _preserve_structure(self, text: str) -> str:
        """
        Сохраняет структуру текста (абзацы, списки).
        """
        # Обрабатываем маркеры страниц из PDF
        text = re.sub(r"\[Страница \d+\]", lambda m: f"\n{m.group(0)}\n", text)

        # Обрабатываем списки (•, -, *, цифры)
        text = re.sub(r"^([•\-\*]|\d+\.)\s", r"\1 ", text, flags=re.MULTILINE)

        return text

    def clean_for_embedding(self, text: str) -> str:
        """
        Специальная очистка для создания эмбеддингов.

        Более агрессивная очистка, убирает всё лишнее.

        Args:
            text: Исходный текст

        Returns:
            Очищенный текст для эмбеддинга
        """
        # Базовая очистка
        cleaned = self.clean(text)

        # Удаляем маркеры страниц
        cleaned = re.sub(r"\[Страница \d+\]", "", cleaned)

        # Удаляем множественные пробелы
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Удаляем переносы строк
        cleaned = cleaned.replace("\n", " ")

        return cleaned.strip()

    def remove_page_numbers(self, text: str) -> str:
        """
        Удаляет номера страниц из текста.

        Args:
            text: Текст с номерами страниц

        Returns:
            Текст без номеров страниц
        """
        # Удаляем маркеры [Страница N]
        text = re.sub(r"\[Страница \d+\]", "", text)

        # Удаляем одиночные цифры в начале/конце строки (часто это номера страниц)
        text = re.sub(r"^\d+$", "", text, flags=re.MULTILINE)

        return text

    def split_into_sentences(self, text: str) -> list[str]:
        """
        Разбивает текст на предложения.

        Учитывает особенности русского языка.

        Args:
            text: Исходный текст

        Returns:
            Список предложений
        """
        # Простое разбиение по точке, вопросительному и восклицательному знакам
        # Учитываем сокращения (и т.д., и т.п., т.е.)
        sentences = re.split(r"(?<!\w\.\w.)(?<![А-Я][а-я]\.)(?<=\.|\?|\!)\s", text)

        # Убираем пустые строки
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def __repr__(self) -> str:
        return (
            f"TextCleaner(whitespace={self.remove_extra_whitespace}, "
            f"unicode={self.normalize_unicode}, "
            f"special={self.remove_special_chars}, "
            f"structure={self.preserve_structure})"
        )
