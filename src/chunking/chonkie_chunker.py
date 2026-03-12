# -*- coding: utf-8 -*-
"""
Чанкинг с использованием библиотеки Chonkie.
"""

from typing import List, Dict, Any
from loguru import logger

try:
    from chonkie import TokenChunker
    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False
    logger.warning("Chonkie не установлен. Используйте LangChain чанкер как fallback.")

from ..core.base_chunker import BaseChunker, Document, Chunk
from ..utils.exceptions import ChunkingError
from ..utils.page_extractor import extract_page_ranges, determine_page_number


class ChonkieChunker(BaseChunker):
    """
    Чанкер на базе библиотеки Chonkie.

    Chonkie использует токенизацию для более точного разбиения текста.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        model_name: str = "gpt-3.5-turbo",  # Используется только для подсчета токенов
        **kwargs
    ):
        """
        Инициализация Chonkie чанкера.

        Args:
            chunk_size: Размер чанка в токенах
            chunk_overlap: Перекрытие чанков в токенах
            model_name: Название модели для токенизатора
            **kwargs: Дополнительные параметры
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

        if not CHONKIE_AVAILABLE:
            raise ChunkingError(
                "Chonkie не установлен. Установите: pip install chonkie"
            )

        self.model_name = model_name

        try:
            # Инициализируем Chonkie TokenChunker
            self.chunker = TokenChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            logger.info(f"ChonkieChunker инициализирован: size={chunk_size}, overlap={chunk_overlap}")
        except Exception as e:
            logger.error(f"Ошибка инициализации Chonkie: {e}")
            raise ChunkingError(f"Не удалось инициализировать Chonkie: {e}")

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Разбивает документ на чанки.

        Args:
            document: Документ для чанкинга

        Returns:
            Список чанков

        Raises:
            ChunkingError: Если не удалось разбить документ
        """
        try:
            # Используем Chonkie для разбиения
            chunks_data = self.chunker.chunk(document.text)

            # Извлекаем все маркеры страниц из полного текста документа
            page_ranges = extract_page_ranges(document.text)

            chunks = []

            for idx, chonkie_chunk in enumerate(chunks_data):
                chunk_text_str = chonkie_chunk.text if hasattr(chonkie_chunk, 'text') else str(chonkie_chunk)

                # Создаем метаданные для чанка
                chunk_metadata = document.metadata.copy() if document.metadata else {}
                chunk_metadata["chunk_index"] = idx
                chunk_metadata["chunk_size"] = len(chunk_text_str)

                # Используем start_index из Chonkie для точного определения позиции
                chunk_start = getattr(chonkie_chunk, 'start_index', None)
                if chunk_start is None:
                    chunk_start = document.text.find(chunk_text_str)
                    if chunk_start == -1:
                        chunk_start = 0

                # Определяем номер страницы на основе позиции чанка
                page_number = determine_page_number(chunk_start, chunk_text_str, page_ranges)
                if page_number is not None:
                    chunk_metadata["page_number"] = page_number
                else:
                    logger.warning(
                        f"Не удалось определить page_number для chunk {idx} "
                        f"в {document.metadata.get('file_name', 'unknown')}"
                    )

                # Генерируем ID чанка
                chunk_id = self._generate_chunk_id(document, idx)

                # Создаем Chunk объект
                chunk = Chunk(
                    text=chunk_text_str,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id
                )

                chunks.append(chunk)

            logger.debug(f"Документ разбит на {len(chunks)} чанков")

            return chunks

        except Exception as e:
            logger.error(f"Ошибка чанкинга документа: {e}")
            raise ChunkingError(f"Не удалось разбить документ: {e}")

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Разбивает текст на чанки.

        Args:
            text: Текст для чанкинга
            metadata: Опциональные метаданные

        Returns:
            Список чанков
        """
        # Создаем временный Document
        doc = Document(text=text, metadata=metadata or {})

        return self.chunk(doc)

    def _generate_chunk_id(self, document: Document, chunk_index: int) -> str:
        """
        Генерирует уникальный ID для чанка.

        Args:
            document: Исходный документ
            chunk_index: Индекс чанка

        Returns:
            ID чанка
        """
        # Используем имя файла если есть
        if "file_name" in document.metadata:
            base_name = document.metadata["file_name"].replace(".pdf", "")
        else:
            base_name = "doc"

        return f"{base_name}_chunk_{chunk_index}"

    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику чанкера.

        Returns:
            Словарь со статистикой
        """
        return {
            "chunker_type": "chonkie",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "model_name": self.model_name,
            "available": CHONKIE_AVAILABLE,
        }

    def __repr__(self) -> str:
        return (
            f"ChonkieChunker(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, "
            f"model={self.model_name})"
        )
