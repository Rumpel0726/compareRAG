# -*- coding: utf-8 -*-
"""
Базовый абстрактный класс для агентов.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class AgentResult:
    """Результат работы агента."""
    success: bool
    data: Any
    metadata: Dict[str, Any] = None
    error: Optional[str] = None

    def __repr__(self) -> str:
        status = "Success" if self.success else "Failed"
        return f"AgentResult(status={status}, error={self.error})"


class BaseAgent(ABC):
    """
    Базовый абстрактный класс для всех агентов.

    Агенты выполняют специфические задачи в RAG pipeline:
    - Анализ запроса
    - Поиск документов
    - Генерация ответа
    - Оценка качества
    """

    def __init__(self, name: str, **kwargs):
        """
        Инициализация агента.

        Args:
            name: Название агента
            **kwargs: Дополнительные параметры
        """
        self.name = name
        self.kwargs = kwargs

        logger.debug(f"Агент {self.name} инициализирован")

    @abstractmethod
    def execute(self, input_data: Any, **kwargs) -> AgentResult:
        """
        Выполняет основную задачу агента.

        Args:
            input_data: Входные данные
            **kwargs: Дополнительные параметры

        Returns:
            Результат работы агента
        """
        pass

    def validate_input(self, input_data: Any) -> bool:
        """
        Проверяет корректность входных данных.

        Args:
            input_data: Входные данные

        Returns:
            True если данные корректны
        """
        # Базовая проверка - может быть переопределена
        return input_data is not None

    def log_execution(self, result: AgentResult) -> None:
        """
        Логирует результат выполнения.

        Args:
            result: Результат работы агента
        """
        if result.success:
            logger.info(f"Агент {self.name} успешно выполнен")
        else:
            logger.error(f"Агент {self.name} завершился с ошибкой: {result.error}")

    def get_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию об агенте.

        Returns:
            Словарь с информацией
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "kwargs": self.kwargs
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
