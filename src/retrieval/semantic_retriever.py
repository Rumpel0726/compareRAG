# -*- coding: utf-8 -*-
"""
Семантический ретривер на базе векторного поиска.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from ..core.base_retriever import BaseRetriever, RetrievalResult
from ..core.base_embedder import BaseEmbedder
from ..core.base_vector_store import BaseVectorStore
from ..utils.exceptions import RetrievalError


class SemanticRetriever(BaseRetriever):
    """
    Семантический ретривер использующий векторный поиск.

    Генерирует эмбеддинги для запроса и ищет похожие документы
    в векторном хранилище по косинусному сходству.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        top_k: int = 5,
        relevance_threshold: float = 0.5,
        reranker=None,
        **kwargs
    ):
        """
        Инициализация семантического ретривера.

        Args:
            embedder: Генератор эмбеддингов
            vector_store: Векторное хранилище
            top_k: Количество документов для извлечения
            relevance_threshold: Порог релевантности (минимальный score)
            reranker: Опциональный Reranker для переранжирования top-N → top-k
            **kwargs: Дополнительные параметры
        """
        super().__init__(top_k=top_k, **kwargs)

        self.embedder = embedder
        self.vector_store = vector_store
        self.relevance_threshold = relevance_threshold
        self.reranker = reranker

        logger.debug(
            f"SemanticRetriever инициализирован: top_k={top_k}, "
            f"threshold={relevance_threshold}, "
            f"reranker={reranker.model_name if reranker else 'none'}"
        )

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

        Raises:
            RetrievalError: Если не удалось выполнить поиск
        """
        if not query or not query.strip():
            logger.warning("Пустой запрос")
            return RetrievalResult(
                chunks=[],
                scores=[],
                query=query,
                metadata={"warning": "empty_query"}
            )

        k = k or self.top_k

        try:
            # Если reranker задан — берём top_n кандидатов, затем переранжируем до k
            fetch_k = self.reranker.top_n if self.reranker else k
            logger.debug(f"Поиск по запросу: '{query[:50]}...', k={k}, fetch_k={fetch_k}")

            # Генерируем эмбеддинг для запроса
            query_embedding = self.embedder.embed_query(query)

            # Ищем в векторной БД
            search_results = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                k=fetch_k,
                filter_dict=filter_dict
            )

            # Фильтруем по релевантности
            search_results = self.filter_by_relevance(
                search_results,
                threshold=self.relevance_threshold
            )

            # Извлекаем чанки и scores
            chunks = [r.chunk for r in search_results]
            scores = [r.score for r in search_results]

            # Reranker (если задан)
            if self.reranker and chunks:
                chunks, scores = self.reranker.rerank_chunks(query, chunks, scores, k)
                logger.debug(f"Reranker: {fetch_k} → {len(chunks)} чанков")

            logger.debug(f"Найдено {len(chunks)} релевантных документов")

            return RetrievalResult(
                chunks=chunks,
                scores=scores,
                query=query,
                metadata={
                    "k": k,
                    "threshold": self.relevance_threshold,
                    "filter": filter_dict
                }
            )

        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            raise RetrievalError(f"Не удалось выполнить поиск: {e}")

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

        Raises:
            RetrievalError: Если не удалось выполнить батчевый поиск
        """
        if not queries:
            logger.warning("Пустой список запросов")
            return []

        k = k or self.top_k

        try:
            logger.debug(f"Батчевый поиск для {len(queries)} запросов, k={k}")

            results = []

            # Простая реализация - последовательный поиск
            # Можно оптимизировать через батчевую генерацию эмбеддингов
            for query in queries:
                result = self.retrieve(query, k=k)
                results.append(result)

            logger.debug(f"Батчевый поиск завершен: {len(results)} результатов")

            return results

        except Exception as e:
            logger.error(f"Ошибка батчевого поиска: {e}")
            raise RetrievalError(f"Не удалось выполнить батчевый поиск: {e}")

    def retrieve_with_context(
        self,
        query: str,
        k: Optional[int] = None,
        window_size: int = 0
    ) -> RetrievalResult:
        """
        Извлекает документы с дополнительным контекстом.

        Опциональная функция для извлечения соседних чанков.

        Args:
            query: Поисковый запрос
            k: Количество документов
            window_size: Количество соседних чанков с каждой стороны

        Returns:
            Результат поиска с расширенным контекстом
        """
        # Базовый поиск
        result = self.retrieve(query, k=k)

        if window_size == 0:
            return result

        # TODO: Реализовать извлечение соседних чанков
        # Требует поддержки в векторном хранилище
        logger.warning("Извлечение с контекстом пока не реализовано")

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику ретривера.

        Returns:
            Словарь со статистикой
        """
        vector_store_stats = self.vector_store.get_collection_stats()

        return {
            "retriever_type": "semantic",
            "top_k": self.top_k,
            "relevance_threshold": self.relevance_threshold,
            "vector_store": vector_store_stats
        }

    def __repr__(self) -> str:
        return (
            f"SemanticRetriever(top_k={self.top_k}, "
            f"threshold={self.relevance_threshold})"
        )
