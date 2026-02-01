# -*- coding: utf-8 -*-
"""
Базовый абстрактный класс для стратегий чанкинга.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Document:
    """Представление документа с метаданными."""
    text: str
    metadata: Dict[str, Any]

    def __repr__(self) -> str:
        return f"Document(text={self.text[:50]}..., metadata={self.metadata})"


@dataclass
class Chunk:
    """Представление чанка документа."""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str = ""

    def __repr__(self) -> str:
        return f"Chunk(id={self.chunk_id}, text={self.text[:30]}..., metadata={self.metadata})"


class BaseChunker(ABC):
    """
    Базовый абстрактный класс для всех стратегий чанкинга.

    Реализует Strategy pattern для различных подходов к разбиению
    документов на чанки.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, **kwargs):
        """
        Инициализация чанкера.

        Args:
            chunk_size: Размер чанка в токенах/символах
            chunk_overlap: Размер перекрытия между чанками
            **kwargs: Дополнительные параметры для конкретных реализаций
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kwargs = kwargs
        self._chunk_count = 0

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Разбивает документ на чанки.

        Args:
            document: Документ для чанкинга

        Returns:
            Список чанков с метаданными
        """
        pass

    @abstractmethod
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Разбивает текст на чанки.

        Args:
            text: Текст для чанкинга
            metadata: Опциональные метаданные

        Returns:
            Список чанков
        """
        pass

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Разбивает список документов на чанки.

        Args:
            documents: Список документов

        Returns:
            Список всех чанков
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику чанкинга.

        Returns:
            Словарь со статистикой
        """
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "total_chunks_created": self._chunk_count,
            "strategy": self.__class__.__name__
        }

    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """
        Генерирует уникальный ID для чанка.

        Args:
            doc_id: ID документа
            chunk_index: Индекс чанка в документе

        Returns:
            Уникальный ID чанка
        """
        return f"{doc_id}_chunk_{chunk_index}"
