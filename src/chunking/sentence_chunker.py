# -*- coding: utf-8 -*-
"""
Чанкер на основе SentenceChunker из библиотеки Chonkie.
"""

from typing import List, Dict, Any
from loguru import logger

try:
    from chonkie import SentenceChunker as ChonkieSentenceChunker
    SENTENCE_CHUNKER_AVAILABLE = True
except ImportError:
    SENTENCE_CHUNKER_AVAILABLE = False
    logger.warning("Chonkie SentenceChunker недоступен.")

from ..core.base_chunker import BaseChunker, Document, Chunk
from ..utils.exceptions import ChunkingError
from ..utils.page_extractor import extract_page_ranges, determine_page_number


class SentenceChunkerWrapper(BaseChunker):
    """
    Чанкер на базе Chonkie SentenceChunker.

    Разбивает текст по границам предложений, затем объединяет
    предложения до лимита токенов. Лучше сохраняет семантическую
    целостность чанков по сравнению с TokenChunker.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 12,
        delim: List[str] = None,
        **kwargs
    ):
        """
        Инициализация SentenceChunker.

        Args:
            chunk_size: Максимальное количество токенов на чанк
            chunk_overlap: Перекрытие между чанками в токенах
            min_sentences_per_chunk: Минимум предложений в каждом чанке
            min_characters_per_sentence: Минимальная длина предложения (символов)
            delim: Разделители предложений. По умолчанию оптимизированы для русского
            **kwargs: Дополнительные параметры
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

        if not SENTENCE_CHUNKER_AVAILABLE:
            raise ChunkingError(
                "Chonkie SentenceChunker недоступен. Установите: pip install chonkie"
            )

        # Разделители адаптированы для русского текста
        if delim is None:
            delim = [". ", "! ", "? ", ".\n", "!\n", "?\n", "\n\n"]

        self.delim = delim
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.min_characters_per_sentence = min_characters_per_sentence

        try:
            self.chunker = ChonkieSentenceChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_sentences_per_chunk=min_sentences_per_chunk,
                min_characters_per_sentence=min_characters_per_sentence,
                delim=delim,
            )
            logger.info(
                f"SentenceChunker инициализирован: "
                f"size={chunk_size}, overlap={chunk_overlap}, "
                f"min_sentences={min_sentences_per_chunk}"
            )
        except Exception as e:
            logger.error(f"Ошибка инициализации SentenceChunker: {e}")
            raise ChunkingError(f"Не удалось инициализировать SentenceChunker: {e}")

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Разбивает документ на чанки по границам предложений.

        Args:
            document: Документ для чанкинга

        Returns:
            Список чанков

        Raises:
            ChunkingError: Если не удалось разбить документ
        """
        try:
            chunks_data = self.chunker.chunk(document.text)

            # Извлекаем все маркеры страниц из полного текста документа
            page_ranges = extract_page_ranges(document.text)

            chunks = []
            current_position = 0

            for idx, chunk_obj in enumerate(chunks_data):
                # Извлекаем текст чанка
                if hasattr(chunk_obj, "text"):
                    chunk_text_str = chunk_obj.text
                else:
                    chunk_text_str = str(chunk_obj)

                # Определяем позицию чанка в документе
                # Chonkie SentenceChunker предоставляет start_index напрямую
                if hasattr(chunk_obj, "start_index") and chunk_obj.start_index is not None:
                    chunk_start = chunk_obj.start_index
                else:
                    chunk_start = document.text.find(chunk_text_str, current_position)
                    if chunk_start == -1:
                        chunk_start = current_position

                # Создаем метаданные для чанка
                chunk_metadata = document.metadata.copy() if document.metadata else {}
                chunk_metadata["chunk_index"] = idx
                chunk_metadata["chunk_size"] = len(chunk_text_str)

                if hasattr(chunk_obj, "token_count") and chunk_obj.token_count is not None:
                    chunk_metadata["token_count"] = chunk_obj.token_count

                # Определяем номер страницы на основе позиции чанка
                page_number = determine_page_number(chunk_start, chunk_text_str, page_ranges)
                if page_number is not None:
                    chunk_metadata["page_number"] = page_number
                else:
                    logger.warning(
                        f"Не удалось определить page_number для chunk {idx} "
                        f"в {document.metadata.get('file_name', 'unknown')}"
                    )

                # Обновляем текущую позицию
                current_position = chunk_start + len(chunk_text_str)

                # Генерируем ID чанка
                chunk_id = self._generate_chunk_id(document, idx)

                chunk = Chunk(
                    text=chunk_text_str,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                )
                chunks.append(chunk)

            logger.debug(f"Документ разбит на {len(chunks)} чанков (sentence)")

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
        if "file_name" in document.metadata:
            base_name = document.metadata["file_name"].replace(".pdf", "")
        else:
            base_name = "doc"

        return f"{base_name}_sent_{chunk_index}"

    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику чанкера.

        Returns:
            Словарь со статистикой
        """
        return {
            "chunker_type": "sentence",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_sentences_per_chunk": self.min_sentences_per_chunk,
            "min_characters_per_sentence": self.min_characters_per_sentence,
            "delim": self.delim,
            "available": SENTENCE_CHUNKER_AVAILABLE,
        }

    def __repr__(self) -> str:
        return (
            f"SentenceChunkerWrapper(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, "
            f"min_sentences={self.min_sentences_per_chunk})"
        )
