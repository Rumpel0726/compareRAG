# -*- coding: utf-8 -*-
"""
Factory для создания векторных хранилищ.
"""

from typing import Dict, Type
from loguru import logger

from ..core.base_vector_store import BaseVectorStore
from ..utils.exceptions import VectorStoreError

from .chroma_store import ChromaStore, CHROMA_AVAILABLE


# Реестр доступных векторных хранилищ
VECTOR_STORE_REGISTRY: Dict[str, Type[BaseVectorStore]] = {
    "chroma": ChromaStore,
}


class VectorStoreFactory:
    """
    Factory для создания векторных хранилищ.
    """

    @staticmethod
    def create(store_type: str, **kwargs) -> BaseVectorStore:
        """
        Создает векторное хранилище по типу.

        Args:
            store_type: Тип хранилища ("chroma")
            **kwargs: Параметры для хранилища

        Returns:
            Экземпляр BaseVectorStore

        Raises:
            VectorStoreError: Если тип не найден или недоступен

        Examples:
            >>> store = VectorStoreFactory.create(
            ...     "chroma",
            ...     collection_name="medical_docs",
            ...     path="./chroma_db"
            ... )
        """
        store_type = store_type.lower()

        if store_type not in VECTOR_STORE_REGISTRY:
            available = ", ".join(VECTOR_STORE_REGISTRY.keys())
            raise VectorStoreError(
                f"Неизвестный тип векторного хранилища: {store_type}. "
                f"Доступные: {available}"
            )

        store_class = VECTOR_STORE_REGISTRY[store_type]

        # Проверяем доступность
        if store_type == "chroma" and not CHROMA_AVAILABLE:
            raise VectorStoreError(
                "ChromaDB недоступен. Установите: pip install chromadb"
            )

        try:
            store = store_class(**kwargs)
            logger.info(f"Создано векторное хранилище: {store}")
            return store
        except Exception as e:
            logger.error(f"Ошибка создания векторного хранилища {store_type}: {e}")
            raise VectorStoreError(f"Не удалось создать хранилище: {e}")

    @staticmethod
    def get_available_stores() -> list[str]:
        """
        Возвращает список доступных векторных хранилищ.

        Returns:
            Список названий хранилищ
        """
        available = []

        for name in VECTOR_STORE_REGISTRY.keys():
            if name == "chroma" and CHROMA_AVAILABLE:
                available.append(name)

        return available

    @staticmethod
    def register_store(name: str, store_class: Type[BaseVectorStore]) -> None:
        """
        Регистрирует новое векторное хранилище в реестре.

        Args:
            name: Название хранилища
            store_class: Класс хранилища (наследник BaseVectorStore)

        Raises:
            VectorStoreError: Если класс не наследуется от BaseVectorStore
        """
        if not issubclass(store_class, BaseVectorStore):
            raise VectorStoreError(
                f"Store класс должен наследоваться от BaseVectorStore: {store_class}"
            )

        VECTOR_STORE_REGISTRY[name.lower()] = store_class
        logger.info(f"Зарегистрировано новое векторное хранилище: {name}")


def create_vector_store_from_config(config) -> BaseVectorStore:
    """
    Создает векторное хранилище из конфигурации.

    Args:
        config: Объект конфигурации с полями chromadb

    Returns:
        Экземпляр BaseVectorStore
    """
    return VectorStoreFactory.create(
        store_type="chroma",
        collection_name=config.chromadb.collection_name,
        path=config.chromadb.path,
        distance_function=config.chromadb.distance_function,
    )
