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


# Реестр доступных ретриверов
RETRIEVER_REGISTRY: Dict[str, Type[BaseRetriever]] = {
    "semantic": SemanticRetriever,
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
            retriever_type: Тип ретривера ("semantic")
            embedder: Генератор эмбеддингов
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
        return list(RETRIEVER_REGISTRY.keys())

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
    vector_store: BaseVectorStore
) -> BaseRetriever:
    """
    Создает ретривер из конфигурации.

    Args:
        config: Объект конфигурации с полями retrieval
        embedder: Генератор эмбеддингов
        vector_store: Векторное хранилище

    Returns:
        Экземпляр BaseRetriever
    """
    return RetrieverFactory.create(
        retriever_type="semantic",  # По умолчанию семантический поиск
        embedder=embedder,
        vector_store=vector_store,
        top_k=config.retrieval.top_k,
        relevance_threshold=config.retrieval.score_threshold
    )
