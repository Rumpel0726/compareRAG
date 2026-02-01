# -*- coding: utf-8 -*-
"""
Извлечение метаданных из документов.
"""

import re
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger


class MetadataExtractor:
    """
    Класс для извлечения метаданных из текста документа.
    """

    def __init__(self):
        """Инициализация экстрактора метаданных."""
        # Паттерны для поиска ключевых фраз в научных статьях
        self.patterns = {
            "authors": [
                r"Авторы?:\s*(.+)",
                r"Authors?:\s*(.+)",
            ],
            "keywords": [
                r"Ключевые слова:\s*(.+)",
                r"Keywords?:\s*(.+)",
            ],
            "abstract": [
                r"Аннотация:\s*(.+?)(?=\n\n|\nВведение|\nKeywords)",
                r"Abstract:\s*(.+?)(?=\n\n|\nIntroduction|\nАннотация)",
            ],
        }

    def extract(self, text: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Извлекает метаданные из текста документа.

        Args:
            text: Текст документа
            file_path: Путь к файлу (опционально)

        Returns:
            Словарь с метаданными
        """
        metadata = {}

        # Добавляем базовые метаданные
        if file_path:
            path = Path(file_path)
            metadata["file_name"] = path.name
            metadata["file_path"] = str(path)

        # Извлекаем авторов
        authors = self._extract_pattern(text, self.patterns["authors"])
        if authors:
            metadata["authors"] = self._clean_authors(authors)

        # Извлекаем ключевые слова
        keywords = self._extract_pattern(text, self.patterns["keywords"])
        if keywords:
            metadata["keywords"] = self._clean_keywords(keywords)

        # Извлекаем аннотацию
        abstract = self._extract_pattern(text, self.patterns["abstract"])
        if abstract:
            metadata["abstract"] = abstract.strip()

        # Извлекаем язык документа
        metadata["language"] = self._detect_language(text)

        # Подсчитываем статистику
        stats = self._calculate_statistics(text)
        metadata.update(stats)

        return metadata

    def _extract_pattern(self, text: str, patterns: list[str]) -> Optional[str]:
        """
        Извлекает текст по списку регулярных выражений.

        Args:
            text: Текст для поиска
            patterns: Список паттернов

        Returns:
            Найденный текст или None
        """
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return None

    def _clean_authors(self, authors_string: str) -> list[str]:
        """
        Очищает и разбивает строку с авторами на список.

        Args:
            authors_string: Строка с авторами

        Returns:
            Список авторов
        """
        # Разбиваем по запятой или точке с запятой
        authors = re.split(r"[,;]", authors_string)

        # Очищаем каждое имя
        authors = [author.strip() for author in authors if author.strip()]

        # Удаляем email и другие артефакты
        authors = [re.sub(r"\s*\(.*?\)", "", author) for author in authors]
        authors = [re.sub(r"\s*<.*?>", "", author) for author in authors]

        return authors

    def _clean_keywords(self, keywords_string: str) -> list[str]:
        """
        Очищает и разбивает строку с ключевыми словами на список.

        Args:
            keywords_string: Строка с ключевыми словами

        Returns:
            Список ключевых слов
        """
        # Разбиваем по запятой или точке с запятой
        keywords = re.split(r"[,;]", keywords_string)

        # Очищаем каждое ключевое слово
        keywords = [kw.strip().lower() for kw in keywords if kw.strip()]

        return keywords

    def _detect_language(self, text: str) -> str:
        """
        Простое определение языка текста.

        Args:
            text: Текст для анализа

        Returns:
            Код языка ('ru' или 'en')
        """
        # Подсчитываем кириллические и латинские буквы
        cyrillic_count = len(re.findall(r"[а-яА-ЯёЁ]", text))
        latin_count = len(re.findall(r"[a-zA-Z]", text))

        # Если больше кириллицы - русский, иначе - английский
        return "ru" if cyrillic_count > latin_count else "en"

    def _calculate_statistics(self, text: str) -> Dict[str, int]:
        """
        Вычисляет базовую статистику по тексту.

        Args:
            text: Текст для анализа

        Returns:
            Словарь со статистикой
        """
        stats = {}

        # Количество символов
        stats["char_count"] = len(text)

        # Количество слов
        words = re.findall(r"\b\w+\b", text)
        stats["word_count"] = len(words)

        # Количество предложений (приблизительно)
        sentences = re.split(r"[.!?]+", text)
        stats["sentence_count"] = len([s for s in sentences if s.strip()])

        # Количество строк
        lines = text.split("\n")
        stats["line_count"] = len([line for line in lines if line.strip()])

        return stats

    def extract_from_filename(self, filename: str) -> Dict[str, Any]:
        """
        Извлекает метаданные из имени файла.

        Args:
            filename: Имя файла

        Returns:
            Словарь с метаданными
        """
        metadata = {}
        path = Path(filename)

        # Базовые данные
        metadata["file_name"] = path.name
        metadata["file_stem"] = path.stem

        # Пытаемся извлечь год из имени файла
        year_match = re.search(r"(19|20)\d{2}", path.stem)
        if year_match:
            metadata["year"] = int(year_match.group(0))

        # Пытаемся найти ключевые слова в названии (заменяем дефисы на пробелы)
        title_words = path.stem.replace("-", " ").replace("_", " ")
        metadata["title_from_filename"] = title_words

        return metadata

    def __repr__(self) -> str:
        return "MetadataExtractor()"
