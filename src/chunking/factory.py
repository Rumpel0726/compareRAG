# -*- coding: utf-8 -*-
"""
Factory для создания чанкеров по стратегии.
"""

from typing import Dict, Type
from loguru import logger

from ..core.base_chunker import BaseChunker
from ..utils.exceptions import ChunkingError

from .chonkie_chunker import ChonkieChunker, CHONKIE_AVAILABLE
from .langchain_chunker import LangChainChunker, LANGCHAIN_AVAILABLE
from .sentence_chunker import SentenceChunkerWrapper, SENTENCE_CHUNKER_AVAILABLE
from .semantic_chunker import SemanticChunkerWrapper, SEMANTIC_CHUNKER_AVAILABLE
from .recursive_chunker import RecursiveChunkerWrapper, RECURSIVE_CHUNKER_AVAILABLE


# Реестр доступных чанкеров
CHUNKER_REGISTRY: Dict[str, Type[BaseChunker]] = {
    "chonkie": ChonkieChunker,
    "langchain": LangChainChunker,
    "recursive": RecursiveChunkerWrapper,
    "sentence": SentenceChunkerWrapper,
    "semantic_chunker": SemanticChunkerWrapper,
}


class ChunkingStrategyFactory:
    """
    Factory для создания чанкеров по названию стратегии.

    Использует Strategy pattern для легкой смены стратегий.
    """

    @staticmethod
    def create(strategy_name: str, **kwargs) -> BaseChunker:
        """
        Создает чанкер по названию стратегии.

        Args:
            strategy_name: Название стратегии ("chonkie", "langchain")
            **kwargs: Параметры для чанкера (chunk_size, chunk_overlap и т.д.)

        Returns:
            Экземпляр BaseChunker

        Raises:
            ChunkingError: Если стратегия не найдена или недоступна

        Examples:
            >>> chunker = ChunkingStrategyFactory.create("chonkie", chunk_size=512)
            >>> chunks = chunker.chunk(document)
        """
        strategy_name = strategy_name.lower()

        # Проверяем, что стратегия существует
        if strategy_name not in CHUNKER_REGISTRY:
            available = ", ".join(CHUNKER_REGISTRY.keys())
            raise ChunkingError(
                f"Неизвестная стратегия чанкинга: {strategy_name}. "
                f"Доступные: {available}"
            )

        chunker_class = CHUNKER_REGISTRY[strategy_name]

        # Проверяем доступность библиотеки
        if strategy_name == "chonkie" and not CHONKIE_AVAILABLE:
            logger.warning(
                "Chonkie недоступен, переключение на LangChain fallback"
            )
            return ChunkingStrategyFactory.create("langchain", **kwargs)

        if strategy_name == "langchain" and not LANGCHAIN_AVAILABLE:
            raise ChunkingError(
                "LangChain недоступен. Установите: pip install langchain"
            )

        if strategy_name == "sentence" and not SENTENCE_CHUNKER_AVAILABLE:
            raise ChunkingError(
                "Chonkie SentenceChunker недоступен. Установите: pip install chonkie"
            )

        if strategy_name == "semantic_chunker" and not SEMANTIC_CHUNKER_AVAILABLE:
            raise ChunkingError(
                "Chonkie SemanticChunker недоступен. Установите: pip install chonkie"
            )

        if strategy_name == "recursive" and not RECURSIVE_CHUNKER_AVAILABLE:
            raise ChunkingError(
                "Chonkie RecursiveChunker недоступен. Установите: pip install chonkie"
            )

        # Создаем и возвращаем экземпляр чанкера
        try:
            chunker = chunker_class(**kwargs)
            logger.info(f"Создан чанкер: {chunker}")
            return chunker
        except Exception as e:
            logger.error(f"Ошибка создания чанкера {strategy_name}: {e}")
            raise ChunkingError(f"Не удалось создать чанкер: {e}")

    @staticmethod
    def get_available_strategies() -> list[str]:
        """
        Возвращает список доступных стратегий чанкинга.

        Returns:
            Список названий стратегий
        """
        available = []

        for name, chunker_class in CHUNKER_REGISTRY.items():
            # Проверяем доступность
            if name == "chonkie" and CHONKIE_AVAILABLE:
                available.append(name)
            elif name == "langchain" and LANGCHAIN_AVAILABLE:
                available.append(name)
            elif name == "sentence" and SENTENCE_CHUNKER_AVAILABLE:
                available.append(name)
            elif name == "recursive" and RECURSIVE_CHUNKER_AVAILABLE:
                available.append(name)
            elif name == "semantic_chunker" and SEMANTIC_CHUNKER_AVAILABLE:
                available.append(name)

        return available

    @staticmethod
    def register_chunker(name: str, chunker_class: Type[BaseChunker]) -> None:
        """
        Регистрирует новый чанкер в реестре.

        Позволяет добавлять кастомные чанкеры.

        Args:
            name: Название стратегии
            chunker_class: Класс чанкера (наследник BaseChunker)

        Raises:
            ChunkingError: Если класс не наследуется от BaseChunker
        """
        if not issubclass(chunker_class, BaseChunker):
            raise ChunkingError(
                f"Chunker класс должен наследоваться от BaseChunker: {chunker_class}"
            )

        CHUNKER_REGISTRY[name.lower()] = chunker_class
        logger.info(f"Зарегистрирован новый чанкер: {name}")


def create_chunker_from_config(config) -> BaseChunker:
    """
    Создает чанкер из конфигурации.

    Args:
        config: Объект конфигурации с полями chunking.strategy, chunk_size, chunk_overlap

    Returns:
        Экземпляр BaseChunker
    """
    return ChunkingStrategyFactory.create(
        strategy_name=config.chunking.strategy,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
    )
