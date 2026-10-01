# -*- coding: utf-8 -*-
"""
Factory для создания ретриверов.
"""

from typing import Dict, Type
from loguru import logger

from ..core.base_retriever import BaseRetriever
from ..core.base_embedder import BaseEmbedder
from ..core.base_vector_store import BaseVectorStore
from ..utils.exceptions import RetrievalError

from .semantic_retriever import SemanticRetriever
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever


# Реестр доступных ретриверов
RETRIEVER_REGISTRY: Dict[str, Type[BaseRetriever]] = {
    "semantic": SemanticRetriever,
    "bm25": BM25Retriever,
}


class RetrieverFactory:
    """
    Factory для создания ретриверов.
    """

    @staticmethod
    def create(
        retriever_type: str,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        **kwargs
    ) -> BaseRetriever:
        """
        Создает ретривер по типу.

        Args:
            retriever_type: Тип ретривера ("semantic", "bm25")
            embedder: Генератор эмбеддингов (не используется для bm25)
            vector_store: Векторное хранилище
            **kwargs: Параметры для ретривера

        Returns:
            Экземпляр BaseRetriever

        Raises:
            RetrievalError: Если тип не найден

        Examples:
            >>> retriever = RetrieverFactory.create(
            ...     "semantic",
            ...     embedder=embedder,
            ...     vector_store=vector_store,
            ...     top_k=5
            ... )
        """
        retriever_type = retriever_type.lower()

        if retriever_type not in RETRIEVER_REGISTRY:
            available = ", ".join(RETRIEVER_REGISTRY.keys())
            raise RetrievalError(
                f"Неизвестный тип ретривера: {retriever_type}. "
                f"Доступные: {available}"
            )

        retriever_class = RETRIEVER_REGISTRY[retriever_type]

        try:
            if retriever_type == "bm25":
                # BM25 не использует embedder
                retriever = retriever_class(
                    vector_store=vector_store,
                    **kwargs
                )
            else:
                retriever = retriever_class(
                    embedder=embedder,
                    vector_store=vector_store,
                    **kwargs
                )
            logger.info(f"Создан ретривер: {retriever}")
            return retriever
        except Exception as e:
            logger.error(f"Ошибка создания ретривера {retriever_type}: {e}")
            raise RetrievalError(f"Не удалось создать ретривер: {e}")

    @staticmethod
    def get_available_retrievers() -> list[str]:
        """
        Возвращает список доступных ретриверов.

        Returns:
            Список названий ретриверов
        """
        return list(RETRIEVER_REGISTRY.keys()) + ["hybrid"]

    @staticmethod
    def register_retriever(name: str, retriever_class: Type[BaseRetriever]) -> None:
        """
        Регистрирует новый ретривер в реестре.

        Args:
            name: Название ретривера
            retriever_class: Класс ретривера (наследник BaseRetriever)

        Raises:
            RetrievalError: Если класс не наследуется от BaseRetriever
        """
        if not issubclass(retriever_class, BaseRetriever):
            raise RetrievalError(
                f"Retriever класс должен наследоваться от BaseRetriever: {retriever_class}"
            )

        RETRIEVER_REGISTRY[name.lower()] = retriever_class
        logger.info(f"Зарегистрирован новый ретривер: {name}")


def create_retriever_from_config(
    config,
    embedder: BaseEmbedder,
    vector_store: BaseVectorStore,
    reranker=None,
) -> BaseRetriever:
    """
    Создает ретривер из конфигурации.

    Стратегия поиска определяется полем config.retrieval.strategy:
      - "semantic" — только векторный поиск (SemanticRetriever)
      - "hybrid"   — векторный + BM25 с RRF fusion (HybridRetriever)

    Args:
        config: Объект конфигурации с полями retrieval
        embedder: Генератор эмбеддингов
        vector_store: Векторное хранилище

    Returns:
        Экземпляр BaseRetriever

    Raises:
        RetrievalError: Если стратегия неизвестна
    """
    strategy = config.retrieval.strategy.lower()
    top_k = config.retrieval.top_k

    if strategy == "semantic":
        return RetrieverFactory.create(
            retriever_type="semantic",
            embedder=embedder,
            vector_store=vector_store,
            top_k=top_k,
            relevance_threshold=config.retrieval.score_threshold,
            reranker=reranker,
        )

    elif strategy == "hybrid":
        # Если есть reranker — нужно достаточно RRF-кандидатов, чтобы было что переранжировать
        rerank_top_n = reranker.top_n if reranker else 0
        candidate_k = max(top_k * 3, rerank_top_n * 2)

        semantic = SemanticRetriever(
            embedder=embedder,
            vector_store=vector_store,
            top_k=candidate_k,
            relevance_threshold=0.0,  # фильтрация выполняется после RRF
            # sub-retriever не переранжирует — переранжирование выполняется в HybridRetriever
        )

        bm25 = BM25Retriever(
            vector_store=vector_store,
            top_k=candidate_k,
            k1=config.retrieval.bm25_k1,
            b=config.retrieval.bm25_b,
        )

        return HybridRetriever(
            semantic_retriever=semantic,
            bm25_retriever=bm25,
            rrf_k=config.retrieval.rrf_k,
            top_k=top_k,
            reranker=reranker,
        )

    else:
        raise RetrievalError(
            f"Неизвестная стратегия поиска: '{strategy}'. "
            f"Доступные: semantic, hybrid"
        )
