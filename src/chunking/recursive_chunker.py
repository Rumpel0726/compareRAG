# -*- coding: utf-8 -*-
"""
Чанкер на основе RecursiveChunker из библиотеки Chonkie.
"""

from typing import List, Dict, Any
from loguru import logger

try:
    from chonkie import RecursiveChunker as ChonkieRecursiveChunker
    RECURSIVE_CHUNKER_AVAILABLE = True
except ImportError:
    RECURSIVE_CHUNKER_AVAILABLE = False
    logger.warning("Chonkie RecursiveChunker недоступен.")

from ..core.base_chunker import BaseChunker, Document, Chunk
from ..utils.exceptions import ChunkingError
from ..utils.page_extractor import extract_page_ranges, determine_page_number


class RecursiveChunkerWrapper(BaseChunker):
    """
    Чанкер на базе Chonkie RecursiveChunker.

    Рекурсивно разбивает текст по иерархии разделителей
    (абзацы → предложения → слова → символы).
    Аналог LangChain RecursiveCharacterTextSplitter, но на токенах.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        min_characters_per_chunk: int = 24,
        **kwargs
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

        if not RECURSIVE_CHUNKER_AVAILABLE:
            raise ChunkingError(
                "Chonkie RecursiveChunker недоступен. Установите: pip install chonkie"
            )

        self.min_characters_per_chunk = min_characters_per_chunk

        try:
            self.chunker = ChonkieRecursiveChunker(
                chunk_size=chunk_size,
                min_characters_per_chunk=min_characters_per_chunk,
            )
            logger.info(
                f"RecursiveChunker инициализирован: "
                f"size={chunk_size}, min_chars={min_characters_per_chunk}"
            )
        except Exception as e:
            logger.error(f"Ошибка инициализации RecursiveChunker: {e}")
            raise ChunkingError(f"Не удалось инициализировать RecursiveChunker: {e}")

    def chunk(self, document: Document) -> List[Chunk]:
        try:
            chunks_data = self.chunker.chunk(document.text)

            page_ranges = extract_page_ranges(document.text)

            chunks = []
            current_position = 0

            for idx, chunk_obj in enumerate(chunks_data):
                if hasattr(chunk_obj, "text"):
                    chunk_text_str = chunk_obj.text
                else:
                    chunk_text_str = str(chunk_obj)

                if hasattr(chunk_obj, "start_index") and chunk_obj.start_index is not None:
                    chunk_start = chunk_obj.start_index
                else:
                    chunk_start = document.text.find(chunk_text_str, current_position)
                    if chunk_start == -1:
                        chunk_start = current_position

                chunk_metadata = document.metadata.copy() if document.metadata else {}
                chunk_metadata["chunk_index"] = idx
                chunk_metadata["chunk_size"] = len(chunk_text_str)

                if hasattr(chunk_obj, "token_count") and chunk_obj.token_count is not None:
                    chunk_metadata["token_count"] = chunk_obj.token_count

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

            logger.debug(f"Документ разбит на {len(chunks)} чанков (recursive)")

            return chunks

        except Exception as e:
            logger.error(f"Ошибка чанкинга документа: {e}")
            raise ChunkingError(f"Не удалось разбить документ: {e}")

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        doc = Document(text=text, metadata=metadata or {})
        return self.chunk(doc)

    def _generate_chunk_id(self, document: Document, chunk_index: int) -> str:
        if "file_name" in document.metadata:
            base_name = document.metadata["file_name"].replace(".pdf", "")
        else:
            base_name = "doc"
        return f"{base_name}_rec_{chunk_index}"

    def get_stats(self) -> Dict[str, Any]:
        return {
            "chunker_type": "recursive",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_characters_per_chunk": self.min_characters_per_chunk,
            "available": RECURSIVE_CHUNKER_AVAILABLE,
        }

    def __repr__(self) -> str:
        return (
            f"RecursiveChunkerWrapper(chunk_size={self.chunk_size}, "
            f"min_chars={self.min_characters_per_chunk})"
        )
