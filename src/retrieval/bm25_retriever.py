# -*- coding: utf-8 -*-
"""
Лексический ретривер на базе BM25.
"""

import re
from typing import List, Dict, Any, Optional

import numpy as np
from loguru import logger

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank-bm25 не установлен. Установите: pip install rank-bm25")

try:
    import pymorphy3 as pymorphy2  # pymorphy3 — современная замена pymorphy2 с тем же API
    PYMORPHY2_AVAILABLE = True
except ImportError:
    try:
        import pymorphy2
        PYMORPHY2_AVAILABLE = True
    except ImportError:
        PYMORPHY2_AVAILABLE = False
        logger.warning("pymorphy3/pymorphy2 не установлен — будет использован простой split()")

from ..core.base_retriever import BaseRetriever, RetrievalResult
from ..core.base_vector_store import BaseVectorStore
from ..utils.exceptions import RetrievalError


class BM25Retriever(BaseRetriever):
    """
    Лексический ретривер на базе BM25.

    При инициализации загружает все чанки из векторного хранилища
    и строит BM25 индекс в памяти. Лемматизация через pymorphy2
    обеспечивает корректный матчинг русских словоформ.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        top_k: int = 5,
        k1: float = 1.5,
        b: float = 0.75,
        **kwargs
    ):
        """
        Инициализация BM25 ретривера.

        Args:
            vector_store: Векторное хранилище (источник текстов для индекса)
            top_k: Количество документов для извлечения
            k1: BM25 параметр насыщения частоты термина (обычно 1.2–2.0)
            b: BM25 параметр нормализации длины документа (0–1)
            **kwargs: Дополнительные параметры
        """
        if not BM25_AVAILABLE:
            raise RetrievalError(
                "rank-bm25 не установлен. Установите: pip install rank-bm25"
            )

        super().__init__(top_k=top_k, **kwargs)

        self.vector_store = vector_store
        self.k1 = k1
        self.b = b

        self._chunks = []
        self._bm25: Optional[BM25Okapi] = None
        self._morph = None

        self._build_index()

    def _build_index(self) -> None:
        """Загружает все чанки и строит BM25 индекс."""
        logger.info("Построение BM25 индекса...")

        # Инициализируем лемматизатор один раз
        if PYMORPHY2_AVAILABLE:
            self._morph = pymorphy2.MorphAnalyzer()
            logger.debug("pymorphy2 MorphAnalyzer инициализирован")
        else:
            logger.warning("pymorphy2 недоступен, используется простая токенизация")

        # Загружаем все чанки из хранилища
        self._chunks = self.vector_store.get_all_chunks()

        if not self._chunks:
            logger.warning("BM25 индекс пуст — в хранилище нет документов")
            self._bm25 = None
            return

        # Токенизируем и строим корпус
        corpus = [self._tokenize(chunk.text) for chunk in self._chunks]

        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)

        logger.info(
            f"BM25Retriever инициализирован: {len(self._chunks)} документов "
            f"проиндексировано, k1={self.k1}, b={self.b}"
        )

    def _tokenize(self, text: str) -> List[str]:
        """
        Токенизирует текст с лемматизацией через pymorphy2.

        Args:
            text: Входной текст

        Returns:
            Список лемм (нормальных форм слов)
        """
        # Убираем пунктуацию, приводим к нижнему регистру
        clean = re.sub(r'[^\w\s]', ' ', text.lower())
        words = clean.split()

        if self._morph is not None:
            # Лемматизация: «пациентов» → «пациент»
            return [self._morph.parse(w)[0].normal_form for w in words if w]
        else:
            return [w for w in words if w]

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Извлекает релевантные документы по лексическому сходству (BM25).

        Args:
            query: Поисковый запрос
            k: Количество документов (по умолчанию self.top_k)
            filter_dict: Не поддерживается в BM25, игнорируется

        Returns:
            Результат поиска с документами и BM25 scores

        Raises:
            RetrievalError: Если не удалось выполнить поиск
        """
        if not query or not query.strip():
            logger.warning("BM25: пустой запрос")
            return RetrievalResult(
                chunks=[],
                scores=[],
                query=query,
                metadata={"warning": "empty_query"}
            )

        if self._bm25 is None or not self._chunks:
            logger.warning("BM25: индекс пуст")
            return RetrievalResult(
                chunks=[],
                scores=[],
                query=query,
                metadata={"warning": "empty_index"}
            )

        if filter_dict:
            logger.debug("BM25: filter_dict не поддерживается, игнорируется")

        k = k or self.top_k

        try:
            query_tokens = self._tokenize(query)
            raw_scores = self._bm25.get_scores(query_tokens)

            # Берём top-k индексов по убыванию score
            top_indices = np.argsort(raw_scores)[::-1][:k]

            chunks = []
            scores = []
            for idx in top_indices:
                score = float(raw_scores[idx])
                if score > 0.0:
                    chunks.append(self._chunks[idx])
                    scores.append(score)

            logger.debug(f"BM25: найдено {len(chunks)} документов для '{query[:50]}'")

            return RetrievalResult(
                chunks=chunks,
                scores=scores,
                query=query,
                metadata={
                    "retriever": "bm25",
                    "k": k,
                    "k1": self.k1,
                    "b": self.b,
                }
            )

        except Exception as e:
            logger.error(f"BM25: ошибка поиска: {e}")
            raise RetrievalError(f"BM25 поиск не удался: {e}")

    def retrieve_batch(
        self,
        queries: List[str],
        k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Извлекает документы для пакета запросов.

        Args:
            queries: Список запросов
            k: Количество документов

        Returns:
            Список результатов для каждого запроса
        """
        if not queries:
            return []

        return [self.retrieve(query, k=k) for query in queries]

    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику ретривера."""
        return {
            "retriever_type": "bm25",
            "top_k": self.top_k,
            "k1": self.k1,
            "b": self.b,
            "indexed_documents": len(self._chunks),
            "lemmatization": "pymorphy2" if self._morph is not None else "split",
        }

    def __repr__(self) -> str:
        return (
            f"BM25Retriever(top_k={self.top_k}, "
            f"docs={len(self._chunks)}, k1={self.k1}, b={self.b})"
        )
