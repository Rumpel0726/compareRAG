# -*- coding: utf-8 -*-
"""
Интеграция с LM Studio для генерации эмбеддингов.
"""

import httpx
from typing import List, Dict, Any
from loguru import logger
import time

from ..core.base_embedder import BaseEmbedder
from ..utils.exceptions import EmbeddingError


class LMStudioEmbedder(BaseEmbedder):
    """
    Embedder на базе LM Studio API.

    LM Studio предоставляет OpenAI-compatible API для эмбеддингов.
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:1234",
        model: str = "text-embedding-nomic-embed-text-v1.5",
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        add_eos_token: bool = False,
        eos_token: str = "</s>",
        **kwargs
    ):
        """
        Инициализация LM Studio embedder.

        Args:
            url: URL LM Studio API
            model: Название модели для эмбеддингов
            timeout: Таймаут запросов в секундах
            max_retries: Максимальное количество попыток
            retry_delay: Задержка между попытками в секундах
            add_eos_token: Добавлять EOS-токен в конец каждого текста
            eos_token: EOS-токен (зависит от модели, например </s> или <|endoftext|>)
            **kwargs: Дополнительные параметры
        """
        super().__init__(**kwargs)

        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.add_eos_token = add_eos_token
        self.eos_token = eos_token

        # Endpoint для эмбеддингов (OpenAI compatible)
        self.embeddings_endpoint = f"{self.url}/v1/embeddings"

        logger.info(
            f"LMStudioEmbedder инициализирован: url={self.url}, model={self.model}, "
            f"add_eos_token={self.add_eos_token}"
        )

        # Проверяем доступность API
        self._check_connection()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Генерирует эмбеддинги для списка текстов.

        Args:
            texts: Список текстов

        Returns:
            Список векторов эмбеддингов

        Raises:
            EmbeddingError: Если не удалось сгенерировать эмбеддинги
        """
        if not texts:
            return []

        try:
            logger.debug(f"Генерация эмбеддингов для {len(texts)} текстов")

            embeddings = []

            # Используем батчинг для оптимизации
            for batch in self._batch_texts(texts, batch_size=32):
                batch_embeddings = self._embed_batch(batch)
                embeddings.extend(batch_embeddings)

            logger.debug(f"Сгенерировано {len(embeddings)} эмбеддингов")

            return embeddings

        except Exception as e:
            logger.error(f"Ошибка генерации эмбеддингов: {e}")
            raise EmbeddingError(f"Не удалось сгенерировать эмбеддинги: {e}")

    def embed_query(self, text: str) -> List[float]:
        """
        Генерирует эмбеддинг для одного текста (запроса).

        Args:
            text: Текст запроса

        Returns:
            Вектор эмбеддинга

        Raises:
            EmbeddingError: Если не удалось сгенерировать эмбеддинг
        """
        try:
            embeddings = self._embed_batch([text])
            return embeddings[0]

        except Exception as e:
            logger.error(f"Ошибка генерации эмбеддинга для запроса: {e}")
            raise EmbeddingError(f"Не удалось сгенерировать эмбеддинг: {e}")

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Генерирует эмбеддинги для батча текстов с retry логикой.

        Args:
            texts: Батч текстов

        Returns:
            Список эмбеддингов

        Raises:
            EmbeddingError: Если все попытки не удались
        """
        last_error = None

        # Добавляем EOS-токен если требуется моделью (например qwen3-embedding)
        if self.add_eos_token:
            texts = [t + self.eos_token for t in texts]

        for attempt in range(self.max_retries):
            try:
                # Формируем запрос в формате OpenAI API
                payload = {
                    "model": self.model,
                    "input": texts,
                }

                # Отправляем запрос
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        self.embeddings_endpoint,
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )

                    response.raise_for_status()

                    # Парсим ответ
                    result = response.json()

                    # Извлекаем эмбеддинги
                    embeddings = [item["embedding"] for item in result["data"]]

                    return embeddings

            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(
                    f"HTTP ошибка при генерации эмбеддингов (попытка {attempt + 1}/{self.max_retries}): "
                    f"{e.response.status_code} - {e.response.text}"
                )

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"Таймаут при генерации эмбеддингов (попытка {attempt + 1}/{self.max_retries})"
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Ошибка при генерации эмбеддингов (попытка {attempt + 1}/{self.max_retries}): {e}"
                )

            # Ждем перед следующей попыткой
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))

        # Если все попытки не удались
        raise EmbeddingError(
            f"Не удалось сгенерировать эмбеддинги после {self.max_retries} попыток: {last_error}"
        )

    def _batch_texts(self, texts: List[str], batch_size: int) -> List[List[str]]:
        """
        Разбивает список текстов на батчи.

        Args:
            texts: Список текстов
            batch_size: Размер батча

        Returns:
            Список батчей
        """
        return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    def _check_connection(self) -> bool:
        """
        Проверяет доступность LM Studio API.

        Returns:
            True если API доступен

        Raises:
            EmbeddingError: Если API недоступен
        """
        try:
            with httpx.Client(timeout=5) as client:
                # Пытаемся получить список моделей
                response = client.get(f"{self.url}/v1/models")

                if response.status_code == 200:
                    logger.info("LM Studio API доступен")
                    return True
                else:
                    logger.warning(
                        f"LM Studio API вернул статус {response.status_code}"
                    )

        except Exception as e:
            logger.error(f"Не удалось подключиться к LM Studio API: {e}")
            raise EmbeddingError(
                f"LM Studio недоступен по адресу {self.url}. "
                f"Убедитесь, что LM Studio запущен и API включен."
            )

        return False

    def get_embedding_dimension(self) -> int:
        """
        Возвращает размерность эмбеддингов.

        Returns:
            Размерность вектора
        """
        # Генерируем тестовый эмбеддинг для определения размерности
        try:
            test_embedding = self.embed_query("test")
            return len(test_embedding)
        except Exception as e:
            raise EmbeddingError(f"Не удалось определить размерность эмбеддингов модели {self.model}: {e}")

    def __repr__(self) -> str:
        return (
            f"LMStudioEmbedder(url={self.url}, "
            f"model={self.model}, "
            f"timeout={self.timeout})"
        )
