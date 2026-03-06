# -*- coding: utf-8 -*-
"""
Гибридный ретривер: семантический поиск + BM25 с Reciprocal Rank Fusion.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from ..core.base_retriever import BaseRetriever, RetrievalResult
from ..utils.exceptions import RetrievalError


class HybridRetriever(BaseRetriever):
    """
    Гибридный ретривер, объединяющий семантический и лексический поиск.

    Запускает оба ретривера независимо, затем сливает результаты через
    Reciprocal Rank Fusion (RRF). Это позволяет находить документы,
    релевантные и по смыслу, и по ключевым словам.

    RRF формула: score(d) = sum_i( 1 / (k + rank_i(d)) )
    где k=60 — константа сглаживания, rank_i — позиция в i-м списке.
    """

    def __init__(
        self,
        semantic_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        rrf_k: int = 60,
        top_k: int = 5,
        **kwargs
    ):
        """
        Инициализация гибридного ретривера.

        Args:
            semantic_retriever: Семантический ретривер (векторный поиск)
            bm25_retriever: Лексический ретривер (BM25)
            rrf_k: Константа RRF (обычно 60; выше — более равномерное слияние)
            top_k: Количество финальных результатов после fusion
            **kwargs: Дополнительные параметры
        """
        super().__init__(top_k=top_k, **kwargs)

        self.semantic = semantic_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k

        logger.info(
            f"HybridRetriever инициализирован: top_k={top_k}, rrf_k={rrf_k}, "
            f"semantic={semantic_retriever}, bm25={bm25_retriever}"
        )

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Гибридный поиск: semantic + BM25 → RRF fusion.

        Args:
            query: Поисковый запрос
            k: Количество финальных результатов (по умолчанию self.top_k)
            filter_dict: Фильтр по метаданным (передаётся только в semantic)

        Returns:
            Результат поиска с объединёнными chunks и RRF scores

        Raises:
            RetrievalError: Если не удалось выполнить поиск
        """
        if not query or not query.strip():
            logger.warning("HybridRetriever: пустой запрос")
            return RetrievalResult(
                chunks=[],
                scores=[],
                query=query,
                metadata={"warning": "empty_query"}
            )

        k = k or self.top_k
        # Берём больше кандидатов от каждого ретривера для лучшего покрытия
        candidate_k = k * 3

        try:
            logger.debug(f"HybridRetriever: поиск '{query[:50]}...', k={k}, candidate_k={candidate_k}")

            # Запускаем оба ретривера
            sem_result = self.semantic.retrieve(
                query, k=candidate_k, filter_dict=filter_dict
            )
            bm25_result = self.bm25.retrieve(query, k=candidate_k)

            logger.debug(
                f"Semantic: {len(sem_result.chunks)} чанков, "
                f"BM25: {len(bm25_result.chunks)} чанков"
            )

            # RRF fusion
            rrf_scores: Dict[str, float] = {}
            chunk_map: Dict[str, Any] = {}

            for rank, chunk in enumerate(sem_result.chunks):
                rrf_scores[chunk.chunk_id] = (
                    rrf_scores.get(chunk.chunk_id, 0.0)
                    + 1.0 / (self.rrf_k + rank + 1)
                )
                chunk_map[chunk.chunk_id] = chunk

            for rank, chunk in enumerate(bm25_result.chunks):
                rrf_scores[chunk.chunk_id] = (
                    rrf_scores.get(chunk.chunk_id, 0.0)
                    + 1.0 / (self.rrf_k + rank + 1)
                )
                chunk_map[chunk.chunk_id] = chunk

            # Сортируем по убыванию RRF score, берём top-k
            sorted_ids = sorted(
                rrf_scores.keys(),
                key=lambda cid: rrf_scores[cid],
                reverse=True
            )[:k]

            result_chunks = [chunk_map[cid] for cid in sorted_ids]
            result_scores = [rrf_scores[cid] for cid in sorted_ids]

            logger.debug(
                f"HybridRetriever: после fusion {len(result_chunks)} финальных чанков"
            )

            return RetrievalResult(
                chunks=result_chunks,
                scores=result_scores,
                query=query,
                metadata={
                    "retriever": "hybrid",
                    "k": k,
                    "rrf_k": self.rrf_k,
                    "semantic_found": len(sem_result.chunks),
                    "bm25_found": len(bm25_result.chunks),
                    "filter": filter_dict,
                }
            )

        except Exception as e:
            logger.error(f"HybridRetriever: ошибка поиска: {e}")
            raise RetrievalError(f"Гибридный поиск не удался: {e}")

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
        if not queries:
            return []

        return [self.retrieve(query, k=k) for query in queries]

    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику гибридного ретривера."""
        return {
            "retriever_type": "hybrid",
            "top_k": self.top_k,
            "rrf_k": self.rrf_k,
            "semantic": self.semantic.get_statistics(),
            "bm25": self.bm25.get_statistics(),
        }

    def __repr__(self) -> str:
        return (
            f"HybridRetriever(top_k={self.top_k}, rrf_k={self.rrf_k})"
        )
