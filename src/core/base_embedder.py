# -*- coding: utf-8 -*-
"""
Базовый абстрактный класс для генерации эмбеддингов.
"""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class BaseEmbedder(ABC):
    """
    Базовый абстрактный класс для всех эмбеддеров.

    Определяет интерфейс для генерации векторных представлений текста.
    """

    def __init__(self, model_name: str = None, **kwargs):
        """
        Инициализация эмбеддера.

        Args:
            model_name: Название модели для эмбеддингов
            **kwargs: Дополнительные параметры
        """
        self.model_name = model_name
        self.kwargs = kwargs
        self._embedding_dim = None

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Генерирует эмбеддинги для списка документов.

        Args:
            texts: Список текстов для векторизации

        Returns:
            Список векторов эмбеддингов
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Генерирует эмбеддинг для поискового запроса.

        Может отличаться от embed_documents для оптимизации поиска.

        Args:
            text: Текст запроса

        Returns:
            Вектор эмбеддинга
        """
        pass

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Генерирует эмбеддинги пакетами для оптимизации.

        Args:
            texts: Список текстов
            batch_size: Размер пакета

        Returns:
            Список векторов эмбеддингов
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embed_documents(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    @property
    def embedding_dimension(self) -> int:
        """
        Возвращает размерность векторов эмбеддингов.

        Returns:
            Размерность вектора
        """
        if self._embedding_dim is None:
            # Генерируем тестовый эмбеддинг для определения размерности
            test_embedding = self.embed_query("test")
            self._embedding_dim = len(test_embedding)
        return self._embedding_dim

    def normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Нормализует эмбеддинги (L2 normalization).

        Args:
            embeddings: Список векторов эмбеддингов

        Returns:
            Нормализованные векторы
        """
        normalized = []
        for emb in embeddings:
            arr = np.array(emb)
            norm = np.linalg.norm(arr)
            if norm > 0:
                normalized.append((arr / norm).tolist())
            else:
                normalized.append(emb)
        return normalized

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, dim={self.embedding_dimension})"
