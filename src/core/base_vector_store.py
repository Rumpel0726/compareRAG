# -*- coding: utf-8 -*-
"""
Базовый абстрактный класс для векторных хранилищ.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .base_chunker import Chunk


@dataclass
class SearchResult:
    """Результат поиска в векторном хранилище."""
    chunk: Chunk
    score: float
    distance: float = None

    def __repr__(self) -> str:
        return f"SearchResult(score={self.score:.4f}, text={self.chunk.text[:50]}...)"


class BaseVectorStore(ABC):
    """
    Базовый абстрактный класс для всех векторных хранилищ.

    Определяет интерфейс для сохранения и поиска векторных представлений.
    """

    def __init__(self, collection_name: str = "documents", **kwargs):
        """
        Инициализация векторного хранилища.

        Args:
            collection_name: Название коллекции
            **kwargs: Дополнительные параметры
        """
        self.collection_name = collection_name
        self.kwargs = kwargs
        self._document_count = 0

    @abstractmethod
    def add_documents(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Добавляет документы с их эмбеддингами в хранилище.

        Args:
            chunks: Список чанков документов
            embeddings: Список векторов эмбеддингов
            ids: Опциональные ID для документов
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Выполняет поиск похожих документов по вектору запроса.

        Args:
            query_embedding: Вектор запроса
            k: Количество результатов
            filter_dict: Опциональный фильтр по метаданным

        Returns:
            Список результатов поиска
        """
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Удаляет коллекцию из хранилища."""
        pass

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику коллекции.

        Returns:
            Словарь со статистикой
        """
        pass

    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Упрощенный метод для добавления текстов.

        Args:
            texts: Список текстов
            embeddings: Список эмбеддингов
            metadatas: Опциональные метаданные
            ids: Опциональные ID
        """
        from .base_chunker import Chunk

        if metadatas is None:
            metadatas = [{} for _ in texts]

        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]

        chunks = [
            Chunk(text=text, metadata=meta, chunk_id=chunk_id)
            for text, meta, chunk_id in zip(texts, metadatas, ids)
        ]

        self.add_documents(chunks, embeddings, ids)
        self._document_count += len(chunks)

    def similarity_search_with_relevance_scores(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        Поиск с фильтрацией по порогу релевантности.

        Args:
            query_embedding: Вектор запроса
            k: Количество результатов
            filter_dict: Фильтр по метаданным
            score_threshold: Минимальный порог score

        Returns:
            Отфильтрованные результаты
        """
        results = self.similarity_search(query_embedding, k, filter_dict)
        return [r for r in results if r.score >= score_threshold]

    @property
    def document_count(self) -> int:
        """Возвращает количество документов в хранилище."""
        return self._document_count

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(collection={self.collection_name}, docs={self.document_count})"
