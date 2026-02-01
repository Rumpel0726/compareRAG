# -*- coding: utf-8 -*-
"""
Базовый абстрактный класс для стратегий поиска (retrieval).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .base_chunker import Chunk
from .base_vector_store import SearchResult


@dataclass
class RetrievalResult:
    """Результат извлечения документов."""
    chunks: List[Chunk]
    scores: List[float]
    query: str
    metadata: Dict[str, Any] = None

    def __repr__(self) -> str:
        return f"RetrievalResult(query='{self.query[:30]}...', found={len(self.chunks)} chunks)"


class BaseRetriever(ABC):
    """
    Базовый абстрактный класс для всех стратегий поиска.

    Определяет интерфейс для извлечения релевантных документов.
    """

    def __init__(self, top_k: int = 5, **kwargs):
        """
        Инициализация ретривера.

        Args:
            top_k: Количество документов для извлечения
            **kwargs: Дополнительные параметры
        """
        self.top_k = top_k
        self.kwargs = kwargs

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Извлекает релевантные документы по запросу.

        Args:
            query: Поисковый запрос
            k: Количество документов (по умолчанию self.top_k)
            filter_dict: Опциональный фильтр по метаданным

        Returns:
            Результат поиска с документами и score
        """
        pass

    @abstractmethod
    def retrieve_batch(
        self,
        queries: List[str],
        k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Извлекает документы для пакета запросов.

        Args:
            queries: Список запросов
            k: Количество документов

        Returns:
            Список результатов для каждого запроса
        """
        pass

    def rerank_results(
        self,
        results: List[SearchResult],
        query: str
    ) -> List[SearchResult]:
        """
        Перестраивает (re-ranks) результаты поиска.

        Может быть переопределен для более сложных стратегий re-ranking.

        Args:
            results: Изначальные результаты поиска
            query: Поисковый запрос

        Returns:
            Переранжированные результаты
        """
        # По умолчанию возвращаем без изменений
        return results

    def filter_by_relevance(
        self,
        results: List[SearchResult],
        threshold: float = 0.5
    ) -> List[SearchResult]:
        """
        Фильтрует результаты по порогу релевантности.

        Args:
            results: Результаты поиска
            threshold: Минимальный score

        Returns:
            Отфильтрованные результаты
        """
        return [r for r in results if r.score >= threshold]

    def get_strategy_name(self) -> str:
        """Возвращает название стратегии поиска."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(top_k={self.top_k})"
