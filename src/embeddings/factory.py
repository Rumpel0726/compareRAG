# -*- coding: utf-8 -*-
"""
Factory для создания эмбеддеров.
"""

from typing import Dict, Type
from loguru import logger

from ..core.base_embedder import BaseEmbedder
from ..utils.exceptions import EmbeddingError

from .lm_studio_embedder import LMStudioEmbedder


# Реестр доступных эмбеддеров
EMBEDDER_REGISTRY: Dict[str, Type[BaseEmbedder]] = {
    "lm_studio": LMStudioEmbedder,
}


class EmbedderFactory:
    """
    Factory для создания эмбеддеров.
    """

    @staticmethod
    def create(embedder_type: str, **kwargs) -> BaseEmbedder:
        """
        Создает эмбеддер по типу.

        Args:
            embedder_type: Тип эмбеддера ("lm_studio")
            **kwargs: Параметры для эмбеддера

        Returns:
            Экземпляр BaseEmbedder

        Raises:
            EmbeddingError: Если тип не найден

        Examples:
            >>> embedder = EmbedderFactory.create(
            ...     "lm_studio",
            ...     url="http://127.0.0.1:1234",
            ...     model="text-embedding-nomic-embed-text-v1.5"
            ... )
        """
        embedder_type = embedder_type.lower()

        if embedder_type not in EMBEDDER_REGISTRY:
            available = ", ".join(EMBEDDER_REGISTRY.keys())
            raise EmbeddingError(
                f"Неизвестный тип эмбеддера: {embedder_type}. "
                f"Доступные: {available}"
            )

        embedder_class = EMBEDDER_REGISTRY[embedder_type]

        try:
            embedder = embedder_class(**kwargs)
            logger.info(f"Создан эмбеддер: {embedder}")
            return embedder
        except Exception as e:
            logger.error(f"Ошибка создания эмбеддера {embedder_type}: {e}")
            raise EmbeddingError(f"Не удалось создать эмбеддер: {e}")

    @staticmethod
    def get_available_embedders() -> list[str]:
        """
        Возвращает список доступных эмбеддеров.

        Returns:
            Список названий эмбеддеров
        """
        return list(EMBEDDER_REGISTRY.keys())

    @staticmethod
    def register_embedder(name: str, embedder_class: Type[BaseEmbedder]) -> None:
        """
        Регистрирует новый эмбеддер в реестре.

        Args:
            name: Название эмбеддера
            embedder_class: Класс эмбеддера (наследник BaseEmbedder)

        Raises:
            EmbeddingError: Если класс не наследуется от BaseEmbedder
        """
        if not issubclass(embedder_class, BaseEmbedder):
            raise EmbeddingError(
                f"Embedder класс должен наследоваться от BaseEmbedder: {embedder_class}"
            )

        EMBEDDER_REGISTRY[name.lower()] = embedder_class
        logger.info(f"Зарегистрирован новый эмбеддер: {name}")


def create_embedder_from_config(config) -> BaseEmbedder:
    """
    Создает эмбеддер из конфигурации.

    Args:
        config: Объект конфигурации с полями lm_studio

    Returns:
        Экземпляр BaseEmbedder
    """
    return EmbedderFactory.create(
        embedder_type="lm_studio",
        url=config.lm_studio.url,
        model=config.lm_studio.embedding_model,
        timeout=config.lm_studio.timeout,
    )
