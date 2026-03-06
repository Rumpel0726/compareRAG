# -*- coding: utf-8 -*-
"""
Чанкер на основе SemanticChunker из библиотеки Chonkie.

Разбивает текст по семантическим границам: находит точки, в которых
смысл резко меняется, и делает разрез именно там.
Требует локальной embedding-модели (не LM Studio).
"""

from typing import List, Dict, Any, Union
from loguru import logger

try:
    from chonkie import SemanticChunker as ChonkieSemanticChunker
    SEMANTIC_CHUNKER_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKER_AVAILABLE = False
    logger.warning("Chonkie SemanticChunker недоступен.")

from ..core.base_chunker import BaseChunker, Document, Chunk
from ..utils.exceptions import ChunkingError
from ..utils.page_extractor import extract_page_ranges, determine_page_number


class SemanticChunkerWrapper(BaseChunker):
    """
    Чанкер на базе Chonkie SemanticChunker.

    Разбивает текст по семантическим границам с помощью локальной
    embedding-модели и peak-detection (Savitzky-Golay фильтр).
    Не требует LM Studio — использует собственную небольшую модель.

    Параметры разбиения:
        threshold: порог сходства для поиска границ (0–1, по умолчанию 0.5)
        similarity_window: сколько предложений объединять при расчёте сходства
        chunk_size: максимальный размер чанка в токенах (жёсткий лимит)
    """

    # Маленькая локальная модель, не требующая LM Studio
    DEFAULT_EMBEDDING_MODEL = "minishlab/potion-base-32M"

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 0,          # SemanticChunker не использует overlap
        embedding_model: Union[str, Any] = None,
        threshold: float = 0.5,
        similarity_window: int = 3,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 24,
        delim: List[str] = None,
        **kwargs
    ):
        """
        Инициализация SemanticChunker.

        Args:
            chunk_size: Максимальный размер чанка в токенах (жёсткий лимит)
            chunk_overlap: Не используется SemanticChunker, принимается для совместимости
            embedding_model: Локальная embedding-модель (строка или объект).
                По умолчанию "minishlab/potion-base-32M" — маленькая и быстрая.
            threshold: Порог семантического сходства для разреза (0–1).
                Ниже → больше чанков; выше → меньше чанков.
            similarity_window: Окно предложений для расчёта сходства
            min_sentences_per_chunk: Минимум предложений в одном чанке
            min_characters_per_sentence: Минимальная длина предложения (символов)
            delim: Разделители предложений. По умолчанию — для русского текста
            **kwargs: Дополнительные параметры
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

        if not SEMANTIC_CHUNKER_AVAILABLE:
            raise ChunkingError(
                "Chonkie SemanticChunker недоступен. Установите: pip install chonkie"
            )

        if embedding_model is None:
            embedding_model = self.DEFAULT_EMBEDDING_MODEL

        # Разделители для русского текста
        if delim is None:
            delim = [". ", "! ", "? ", ".\n", "!\n", "?\n", "\n\n"]

        self.embedding_model_name = (
            embedding_model if isinstance(embedding_model, str) else repr(embedding_model)
        )
        self.threshold = threshold
        self.similarity_window = similarity_window
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.min_characters_per_sentence = min_characters_per_sentence
        self.delim = delim

        try:
            logger.info(
                f"Загрузка SemanticChunker (embedding={self.embedding_model_name})…"
            )
            self.chunker = ChonkieSemanticChunker(
                embedding_model=embedding_model,
                threshold=threshold,
                chunk_size=chunk_size,
                similarity_window=similarity_window,
                min_sentences_per_chunk=min_sentences_per_chunk,
                min_characters_per_sentence=min_characters_per_sentence,
                delim=delim,
            )
            logger.info(
                f"SemanticChunker инициализирован: "
                f"size={chunk_size}, threshold={threshold}, "
                f"window={similarity_window}, model={self.embedding_model_name}"
            )
        except Exception as e:
            logger.error(f"Ошибка инициализации SemanticChunker: {e}")
            raise ChunkingError(f"Не удалось инициализировать SemanticChunker: {e}")

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Разбивает документ на семантически связные чанки.

        Args:
            document: Документ для чанкинга

        Returns:
            Список чанков

        Raises:
            ChunkingError: Если не удалось разбить документ
        """
        try:
            chunks_data = self.chunker.chunk(document.text)

            # Извлекаем маркеры страниц из полного текста
            page_ranges = extract_page_ranges(document.text)

            chunks = []
            current_position = 0

            for idx, chunk_obj in enumerate(chunks_data):
                # Извлекаем текст чанка
                if hasattr(chunk_obj, "text"):
                    chunk_text_str = chunk_obj.text
                else:
                    chunk_text_str = str(chunk_obj)

                # SemanticChunker предоставляет start_index напрямую
                if hasattr(chunk_obj, "start_index") and chunk_obj.start_index is not None:
                    chunk_start = chunk_obj.start_index
                else:
                    chunk_start = document.text.find(chunk_text_str, current_position)
                    if chunk_start == -1:
                        chunk_start = current_position

                # Создаём метаданные
                chunk_metadata = document.metadata.copy() if document.metadata else {}
                chunk_metadata["chunk_index"] = idx
                chunk_metadata["chunk_size"] = len(chunk_text_str)

                if hasattr(chunk_obj, "token_count") and chunk_obj.token_count is not None:
                    chunk_metadata["token_count"] = chunk_obj.token_count

                # Определяем номер страницы
                page_number = determine_page_number(chunk_start, chunk_text_str, page_ranges)
                if page_number is not None:
                    chunk_metadata["page_number"] = page_number
                else:
                    logger.warning(
                        f"Не удалось определить page_number для chunk {idx} "
                        f"в {document.metadata.get('file_name', 'unknown')}"
                    )

                current_position = chunk_start + len(chunk_text_str)

                chunk_id = self._generate_chunk_id(document, idx)

                chunk = Chunk(
                    text=chunk_text_str,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                )
                chunks.append(chunk)

            logger.debug(f"Документ разбит на {len(chunks)} чанков (semantic)")

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
        """Генерирует уникальный ID для чанка."""
        if "file_name" in document.metadata:
            base_name = document.metadata["file_name"].replace(".pdf", "")
        else:
            base_name = "doc"
        return f"{base_name}_sem_{chunk_index}"

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику чанкера."""
        return {
            "chunker_type": "semantic_chunker",
            "chunk_size": self.chunk_size,
            "threshold": self.threshold,
            "similarity_window": self.similarity_window,
            "min_sentences_per_chunk": self.min_sentences_per_chunk,
            "min_characters_per_sentence": self.min_characters_per_sentence,
            "embedding_model": self.embedding_model_name,
            "available": SEMANTIC_CHUNKER_AVAILABLE,
        }

    def __repr__(self) -> str:
        return (
            f"SemanticChunkerWrapper(chunk_size={self.chunk_size}, "
            f"threshold={self.threshold}, "
            f"model={self.embedding_model_name})"
        )
