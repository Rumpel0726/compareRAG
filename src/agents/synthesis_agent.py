# -*- coding: utf-8 -*-
"""
Агент для синтеза финального ответа на основе контекста.
"""

from typing import List, Dict, Any
from loguru import logger

from .base_agent import BaseAgent, AgentResult
from ..llm.lm_studio_client import LMStudioClient
from ..llm.prompts import SYSTEM_PROMPT, create_synthesis_prompt
from ..core.base_chunker import Chunk
from ..utils.exceptions import LLMError


class SynthesisAgent(BaseAgent):
    """
    Агент для генерации финального ответа.

    Принимает вопрос и релевантные документы,
    генерирует структурированный ответ с источниками.
    """

    def __init__(
        self,
        llm_client: LMStudioClient,
        name: str = "SynthesisAgent",
        system_prompt: str = SYSTEM_PROMPT,
        **kwargs
    ):
        """
        Инициализация агента синтеза.

        Args:
            llm_client: Клиент для генерации текста
            name: Название агента
            system_prompt: Системный промпт
            **kwargs: Дополнительные параметры
        """
        super().__init__(name=name, **kwargs)

        self.llm_client = llm_client
        self.system_prompt = system_prompt

        logger.info(f"SynthesisAgent инициализирован")

    def execute(
        self,
        input_data: Dict[str, Any],
        **kwargs
    ) -> AgentResult:
        """
        Генерирует ответ на основе вопроса и контекста.

        Args:
            input_data: Словарь с ключами:
                - query (str): Вопрос пользователя
                - chunks (List[Chunk]): Релевантные документы
            **kwargs: Дополнительные параметры

        Returns:
            AgentResult с сгенерированным ответом
        """
        try:
            # Валидация входных данных
            if not self.validate_input(input_data):
                return AgentResult(
                    success=False,
                    data=None,
                    error="Некорректные входные данные"
                )

            query = input_data.get("query")
            chunks = input_data.get("chunks", [])

            logger.debug(
                f"Генерация ответа для запроса: '{query[:50]}...' "
                f"с {len(chunks)} чанками"
            )

            # Проверяем наличие контекста
            if not chunks:
                return AgentResult(
                    success=True,
                    data="К сожалению, не нашел релевантной информации в документах.",
                    metadata={"has_context": False}
                )

            # Формируем контекст для промпта
            context_chunks = []
            for chunk in chunks:
                context_chunks.append({
                    "text": chunk.text,
                    "metadata": chunk.metadata
                })

            # Создаем промпт для генерации
            user_prompt = create_synthesis_prompt(query, context_chunks)

            # Генерируем ответ через LLM
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            response = self.llm_client.generate(
                messages=messages,
                temperature=kwargs.get("temperature"),
                max_tokens=kwargs.get("max_tokens")
            )

            logger.debug(f"Ответ сгенерирован: {len(response)} символов")

            result = AgentResult(
                success=True,
                data=response,
                metadata={
                    "query": query,
                    "num_chunks_used": len(chunks),
                    "has_context": True
                }
            )

            self.log_execution(result)

            return result

        except LLMError as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return AgentResult(
                success=False,
                data=None,
                error=f"Ошибка генерации ответа: {e}"
            )

        except Exception as e:
            logger.exception(f"Неожиданная ошибка в SynthesisAgent: {e}")
            return AgentResult(
                success=False,
                data=None,
                error=f"Неожиданная ошибка: {e}"
            )

    def validate_input(self, input_data: Any) -> bool:
        """
        Проверяет корректность входных данных.

        Args:
            input_data: Входные данные

        Returns:
            True если данные корректны
        """
        if not isinstance(input_data, dict):
            logger.error("input_data должен быть словарем")
            return False

        if "query" not in input_data:
            logger.error("Отсутствует обязательное поле 'query'")
            return False

        query = input_data.get("query")
        if not query or not query.strip():
            logger.error("Пустой запрос")
            return False

        return True
