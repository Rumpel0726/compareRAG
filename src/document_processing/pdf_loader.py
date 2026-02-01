# -*- coding: utf-8 -*-
"""
Загрузка и извлечение текста из PDF документов.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import pypdf
from loguru import logger

from ..core.base_chunker import Document
from ..utils.exceptions import DocumentProcessingError


class PDFLoader:
    """
    Загрузчик PDF документов с поддержкой русского языка.
    """

    def __init__(self, extract_images: bool = False):
        """
        Инициализация загрузчика.

        Args:
            extract_images: Извлекать ли изображения (пока не реализовано)
        """
        self.extract_images = extract_images

    def load(self, file_path: str) -> Document:
        """
        Загружает один PDF файл.

        Args:
            file_path: Путь к PDF файлу

        Returns:
            Document с текстом и метаданными

        Raises:
            DocumentProcessingError: Если не удалось загрузить файл
        """
        path = Path(file_path)

        if not path.exists():
            raise DocumentProcessingError(f"Файл не найден: {file_path}")

        if not path.suffix.lower() == ".pdf":
            raise DocumentProcessingError(f"Файл не является PDF: {file_path}")

        try:
            logger.info(f"Загрузка PDF: {path.name}")

            # Читаем PDF
            text = self._extract_text(path)

            # Извлекаем метаданные
            metadata = self._extract_metadata(path)

            # Добавляем путь к файлу
            metadata["file_path"] = str(path)
            metadata["file_name"] = path.name

            logger.info(f"PDF загружен: {path.name}, {len(text)} символов")

            return Document(text=text, metadata=metadata)

        except Exception as e:
            logger.error(f"Ошибка загрузки PDF {path.name}: {e}")
            raise DocumentProcessingError(f"Не удалось загрузить PDF: {e}")

    def load_directory(
        self, directory_path: str, pattern: str = "*.pdf"
    ) -> List[Document]:
        """
        Загружает все PDF файлы из директории.

        Args:
            directory_path: Путь к директории
            pattern: Паттерн для поиска файлов

        Returns:
            Список Document объектов

        Raises:
            DocumentProcessingError: Если директория не найдена
        """
        dir_path = Path(directory_path)

        if not dir_path.exists():
            raise DocumentProcessingError(f"Директория не найдена: {directory_path}")

        if not dir_path.is_dir():
            raise DocumentProcessingError(f"Не является директорией: {directory_path}")

        # Находим все PDF файлы
        pdf_files = sorted(dir_path.glob(pattern))

        if not pdf_files:
            logger.warning(f"PDF файлы не найдены в {directory_path}")
            return []

        logger.info(f"Найдено {len(pdf_files)} PDF файлов в {directory_path}")

        documents = []
        for pdf_file in pdf_files:
            try:
                doc = self.load(str(pdf_file))
                documents.append(doc)
            except DocumentProcessingError as e:
                logger.error(f"Пропуск файла {pdf_file.name}: {e}")
                continue

        logger.info(f"Успешно загружено {len(documents)} из {len(pdf_files)} файлов")

        return documents

    def _extract_text(self, path: Path) -> str:
        """
        Извлекает текст из PDF.

        Args:
            path: Путь к PDF файлу

        Returns:
            Извлеченный текст
        """
        text_parts = []

        with open(path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)

            # Проходим по всем страницам
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                try:
                    page_text = page.extract_text()

                    if page_text:
                        # Добавляем номер страницы в начало
                        text_parts.append(f"\n[Страница {page_num}]\n{page_text}")

                except Exception as e:
                    logger.warning(f"Ошибка извлечения текста со страницы {page_num}: {e}")
                    continue

        # Объединяем весь текст
        full_text = "\n".join(text_parts)

        return full_text

    def _extract_metadata(self, path: Path) -> Dict[str, Any]:
        """
        Извлекает метаданные из PDF.

        Args:
            path: Путь к PDF файлу

        Returns:
            Словарь с метаданными
        """
        metadata = {}

        try:
            with open(path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)

                # Количество страниц
                metadata["num_pages"] = len(pdf_reader.pages)

                # Метаданные из PDF
                if pdf_reader.metadata:
                    pdf_meta = pdf_reader.metadata

                    # Извлекаем основные поля
                    if pdf_meta.title:
                        metadata["title"] = pdf_meta.title
                    if pdf_meta.author:
                        metadata["author"] = pdf_meta.author
                    if pdf_meta.subject:
                        metadata["subject"] = pdf_meta.subject
                    if pdf_meta.creator:
                        metadata["creator"] = pdf_meta.creator

        except Exception as e:
            logger.warning(f"Ошибка извлечения метаданных: {e}")

        # Размер файла
        metadata["file_size"] = path.stat().st_size

        return metadata

    def get_page_count(self, file_path: str) -> int:
        """
        Возвращает количество страниц в PDF.

        Args:
            file_path: Путь к PDF файлу

        Returns:
            Количество страниц
        """
        path = Path(file_path)

        try:
            with open(path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                return len(pdf_reader.pages)
        except Exception as e:
            logger.error(f"Ошибка чтения PDF {path.name}: {e}")
            return 0

    def __repr__(self) -> str:
        return f"PDFLoader(extract_images={self.extract_images})"
