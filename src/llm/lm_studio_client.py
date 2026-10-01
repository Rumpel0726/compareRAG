# -*- coding: utf-8 -*-
"""
Клиент для LM Studio API (генерация текста).
"""

import httpx
from typing import List, Dict, Any, Optional
from loguru import logger
import time

from ..utils.exceptions import LLMError


class LMStudioClient:
    """
    HTTP клиент для LM Studio API (генерация текста).

    LM Studio предоставляет OpenAI-compatible API для chat completions.
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:1234",
        model: str = "qwen/qwen3-14b",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Инициализация LM Studio клиента.

        Args:
            url: URL LM Studio API
            model: Название модели для генерации
            temperature: Температура генерации (0.0-1.0)
            max_tokens: Максимальное количество токенов
            timeout: Таймаут запросов в секундах
            max_retries: Максимальное количество попыток
            retry_delay: Задержка между попытками в секундах
            **kwargs: Дополнительные параметры
        """
        self.url = url.rstrip("/")
        if self.url.endswith("/v1"):
            self.url = self.url[:-3]
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_key = api_key

        # Endpoint для chat completions (OpenAI compatible)
        self.chat_endpoint = f"{self.url}/v1/chat/completions"

        logger.info(
            f"LMStudioClient инициализирован: url={self.url}, "
            f"model={self.model}, temp={self.temperature}"
        )

        # Проверяем доступность API
        self._check_connection()

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Генерирует ответ на основе сообщений.

        Args:
            messages: Список сообщений в формате OpenAI:
                [
                    {"role": "system", "content": "..."},
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."}
                ]
            temperature: Температура генерации (перекрывает дефолтную)
            max_tokens: Максимум токенов (перекрывает дефолтный)
            stream: Использовать streaming (не реализовано)

        Returns:
            Сгенерированный текст

        Raises:
            LLMError: Если не удалось сгенерировать ответ
        """
        if not messages:
            raise LLMError("Пустой список сообщений")

        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens

        try:
            logger.debug(f"Генерация ответа для {len(messages)} сообщений")

            response_text = self._generate_with_retry(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            logger.debug(f"Сгенерирован ответ: {len(response_text)} символов")

            return response_text

        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            raise LLMError(f"Не удалось сгенерировать ответ: {e}")

    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Генерирует ответ с учетом контекста (для RAG).

        Args:
            query: Вопрос пользователя
            context: Контекст из документов
            system_prompt: Системный промпт (опционально)
            temperature: Температура генерации
            max_tokens: Максимум токенов

        Returns:
            Сгенерированный ответ

        Raises:
            LLMError: Если не удалось сгенерировать ответ
        """
        # Формируем сообщения
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Объединяем контекст и запрос
        user_message = f"Контекст из документов:\n{context}\n\nВопрос: {query}"
        messages.append({"role": "user", "content": user_message})

        return self.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def _generate_with_retry(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        Генерирует ответ с retry логикой.

        Args:
            messages: Список сообщений
            temperature: Температура генерации
            max_tokens: Максимум токенов

        Returns:
            Сгенерированный текст

        Raises:
            LLMError: Если все попытки не удались
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Формируем запрос в формате OpenAI API
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                # Отправляем запрос
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        self.chat_endpoint,
                        json=payload,
                        headers=headers
                    )

                    response.raise_for_status()

                    # Парсим ответ
                    result = response.json()

                    # Извлекаем сгенерированный текст
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        return content
                    else:
                        raise LLMError("Неожиданный формат ответа от LM Studio")

            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(
                    f"HTTP ошибка при генерации (попытка {attempt + 1}/{self.max_retries}): "
                    f"{e.response.status_code} - {e.response.text}"
                )

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"Таймаут при генерации (попытка {attempt + 1}/{self.max_retries})"
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Ошибка при генерации (попытка {attempt + 1}/{self.max_retries}): {e}"
                )

            # Ждем перед следующей попыткой
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))

        # Если все попытки не удались
        raise LLMError(
            f"Не удалось сгенерировать ответ после {self.max_retries} попыток: {last_error}"
        )

    def _check_connection(self) -> bool:
        """
        Проверяет доступность LM Studio API.

        Returns:
            True если API доступен

        Raises:
            LLMError: Если API недоступен
        """
        is_remote = not any(
            local in self.url for local in ["localhost", "127.0.0.1", "0.0.0.0"]
        )
        check_timeout = 15 if is_remote else 5

        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            with httpx.Client(timeout=check_timeout) as client:
                # Пытаемся получить список моделей
                response = client.get(f"{self.url}/v1/models", headers=headers)

                if response.status_code == 200:
                    logger.info("LLM API доступен" + (" (remote)" if is_remote else ""))
                    return True
                elif is_remote and response.status_code in (401, 403, 404, 405):
                    # Удалённые API могут не поддерживать /v1/models — это ОК
                    logger.info(
                        f"Remote LLM API ответил {response.status_code} на /v1/models — "
                        f"пропускаем проверку (будет проверен при первом запросе)"
                    )
                    return True
                else:
                    logger.warning(
                        f"LLM API вернул статус {response.status_code}"
                    )

        except Exception as e:
            if is_remote:
                logger.warning(
                    f"Не удалось проверить remote LLM API ({self.url}): {e}. "
                    f"Продолжаем — будет проверен при первом запросе."
                )
                return True
            logger.error(f"Не удалось подключиться к LM Studio API: {e}")
            raise LLMError(
                f"LM Studio недоступен по адресу {self.url}. "
                f"Убедитесь, что LM Studio запущен и модель загружена."
            )

        return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о модели.

        Returns:
            Словарь с информацией
        """
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            with httpx.Client(timeout=5) as client:
                response = client.get(f"{self.url}/v1/models", headers=headers)

                if response.status_code == 200:
                    return response.json()

        except Exception as e:
            logger.error(f"Не удалось получить информацию о моделях: {e}")

        return {"error": "Не удалось получить информацию о моделях"}

    def __repr__(self) -> str:
        return (
            f"LMStudioClient(url={self.url}, "
            f"model={self.model}, "
            f"temp={self.temperature})"
        )
