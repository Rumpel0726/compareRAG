# -*- coding: utf-8 -*-
"""
Fallback чанкер на базе LangChain.
"""

from typing import List, Dict, Any
from loguru import logger

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        logger.warning("LangChain не установлен.")

from ..core.base_chunker import BaseChunker, Document, Chunk
from ..utils.exceptions import ChunkingError
from ..utils.page_extractor import extract_page_ranges, determine_page_number


class LangChainChunker(BaseChunker):
    """
    Fallback чанкер на базе LangChain RecursiveCharacterTextSplitter.

    Использует рекурсивное разбиение по сепараторам.
    Хорошо работает с русским языком.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: List[str] = None,
        **kwargs
    ):
        """
        Инициализация LangChain чанкера.

        Args:
            chunk_size: Размер чанка в символах
            chunk_overlap: Перекрытие чанков в символах
            separators: Список сепараторов для разбиения
            **kwargs: Дополнительные параметры
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

        if not LANGCHAIN_AVAILABLE:
            raise ChunkingError(
                "LangChain не установлен. Установите: pip install langchain"
            )

        # Сепараторы оптимизированы для русского языка
        if separators is None:
            separators = [
                "\n\n",  # Абзацы
                "\n",    # Строки
                ". ",    # Предложения
                "! ",
                "? ",
                "; ",
                ", ",
                " ",     # Слова
                "",      # Символы
            ]

        self.separators = separators

        try:
            # Инициализируем LangChain text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                length_function=len,  # Измеряем в символах
            )

            logger.info(
                f"LangChainChunker инициализирован: "
                f"size={chunk_size}, overlap={chunk_overlap}"
            )

        except Exception as e:
            logger.error(f"Ошибка инициализации LangChain: {e}")
            raise ChunkingError(f"Не удалось инициализировать LangChain: {e}")

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
            # Используем LangChain для разбиения
            chunks_text = self.text_splitter.split_text(document.text)

            # Извлекаем все маркеры страниц из полного текста документа
            page_ranges = extract_page_ranges(document.text)

            chunks = []
            current_position = 0  # Отслеживаем позицию в документе

            for idx, chunk_text in enumerate(chunks_text):
                # Создаем метаданные для чанка
                chunk_metadata = document.metadata.copy() if document.metadata else {}
                chunk_metadata["chunk_index"] = idx
                chunk_metadata["chunk_size"] = len(chunk_text)

                # Находим позицию чанка в документе
                chunk_start = document.text.find(chunk_text, current_position)
                if chunk_start == -1:
                    # Если не нашли точное совпадение, используем текущую позицию
                    chunk_start = current_position

                # Определяем номер страницы на основе позиции чанка
                page_number = determine_page_number(chunk_start, chunk_text, page_ranges)
                if page_number is not None:
                    chunk_metadata["page_number"] = page_number
                else:
                    logger.warning(
                        f"Не удалось определить page_number для chunk {idx} "
                        f"в {document.metadata.get('file_name', 'unknown')}"
                    )

                # Обновляем текущую позицию
                current_position = chunk_start + len(chunk_text)

                # Генерируем ID чанка
                chunk_id = self._generate_chunk_id(document, idx)

                # Создаем Chunk объект
                chunk = Chunk(
                    text=chunk_text,
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
            "chunker_type": "langchain",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": self.separators,
            "available": LANGCHAIN_AVAILABLE,
        }

    def __repr__(self) -> str:
        return (
            f"LangChainChunker(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )
