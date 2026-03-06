# -*- coding: utf-8 -*-
"""
Orchestrator - координатор RAG pipeline.
"""

from typing import Dict, Any, Optional
from loguru import logger
import time

from ..core.base_retriever import BaseRetriever
from ..llm.lm_studio_client import LMStudioClient
from .synthesis_agent import SynthesisAgent
from ..utils.exceptions import RAGPipelineError


class SimpleRAGOrchestrator:
    """
    Простой orchestrator для RAG pipeline.

    Координирует процесс:
    1. Получение запроса от пользователя
    2. Поиск релевантных документов (retriever)
    3. Генерация ответа (synthesis agent)
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm_client: LMStudioClient,
        top_k: int = 5,
        **kwargs
    ):
        """
        Инициализация orchestrator.

        Args:
            retriever: Ретривер для поиска документов
            llm_client: LLM клиент для генерации ответов
            top_k: Количество документов для поиска
            **kwargs: Дополнительные параметры
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.top_k = top_k

        # Инициализируем synthesis agent
        self.synthesis_agent = SynthesisAgent(llm_client=llm_client)

        logger.info(
            f"SimpleRAGOrchestrator инициализирован с top_k={top_k}"
        )

    def query(
        self,
        user_query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Обрабатывает запрос пользователя через RAG pipeline.

        Args:
            user_query: Вопрос пользователя
            k: Количество документов (по умолчанию self.top_k)
            filter_dict: Фильтр по метаданным
            **kwargs: Дополнительные параметры для генерации

        Returns:
            Словарь с ответом и метаданными:
                {
                    "query": str,
                    "answer": str,
                    "sources": List[str],
                    "retrieval_time": float,
                    "generation_time": float,
                    "total_time": float,
                    "num_chunks_found": int
                }

        Raises:
            RAGPipelineError: Если произошла ошибка в pipeline
        """
        if not user_query or not user_query.strip():
            raise RAGPipelineError("Пустой запрос")

        k = k or self.top_k

        start_time = time.time()

        try:
            logger.info(f"Обработка запроса: '{user_query[:50]}...'")

            # Этап 1: Поиск релевантных документов
            logger.debug(f"Этап 1/2: Поиск документов (k={k})")
            retrieval_start = time.time()

            retrieval_result = self.retriever.retrieve(
                query=user_query,
                k=k,
                filter_dict=filter_dict
            )

            retrieval_time = time.time() - retrieval_start

            logger.info(
                f"Найдено {len(retrieval_result.chunks)} релевантных документов "
                f"за {retrieval_time:.2f}с"
            )

            # Этап 2: Генерация ответа
            logger.debug(f"Этап 2/2: Генерация ответа")
            generation_start = time.time()

            synthesis_input = {
                "query": user_query,
                "chunks": retrieval_result.chunks
            }

            synthesis_result = self.synthesis_agent.execute(
                input_data=synthesis_input,
                **kwargs
            )

            generation_time = time.time() - generation_start

            if not synthesis_result.success:
                raise RAGPipelineError(
                    f"Ошибка генерации ответа: {synthesis_result.error}"
                )

            answer = synthesis_result.data

            logger.info(
                f"Ответ сгенерирован за {generation_time:.2f}с "
                f"({len(answer)} символов)"
            )

            # Извлекаем источники
            sources = self._extract_sources(retrieval_result.chunks)

            # Сериализуем chunks для передачи в API
            chunks_data = self._serialize_chunks(retrieval_result.chunks, retrieval_result.scores)

            # Формируем результат
            total_time = time.time() - start_time

            result = {
                "query": user_query,
                "answer": answer,
                "sources": sources,
                "chunks": chunks_data,
                "retrieval_time": round(retrieval_time, 2),
                "generation_time": round(generation_time, 2),
                "total_time": round(total_time, 2),
                "num_chunks_found": len(retrieval_result.chunks),
                "scores": retrieval_result.scores
            }

            logger.info(
                f"✅ Запрос обработан успешно за {total_time:.2f}с"
            )

            return result

        except RAGPipelineError:
            raise

        except Exception as e:
            logger.exception(f"Ошибка в RAG pipeline: {e}")
            raise RAGPipelineError(f"Ошибка обработки запроса: {e}")

    def query_batch(
        self,
        queries: list[str],
        k: Optional[int] = None,
        **kwargs
    ) -> list[Dict[str, Any]]:
        """
        Обрабатывает пакет запросов.

        Args:
            queries: Список запросов
            k: Количество документов
            **kwargs: Дополнительные параметры

        Returns:
            Список результатов для каждого запроса

        Raises:
            RAGPipelineError: Если произошла ошибка
        """
        if not queries:
            raise RAGPipelineError("Пустой список запросов")

        logger.info(f"Обработка {len(queries)} запросов")

        results = []

        for idx, query in enumerate(queries, 1):
            logger.debug(f"Обработка запроса {idx}/{len(queries)}")

            try:
                result = self.query(query, k=k, **kwargs)
                results.append(result)

            except Exception as e:
                logger.error(f"Ошибка обработки запроса {idx}: {e}")
                results.append({
                    "query": query,
                    "answer": None,
                    "error": str(e)
                })

        logger.info(f"Обработано {len(results)} запросов")

        return results

    def _extract_sources(self, chunks) -> list[str]:
        """
        Извлекает уникальные источники из чанков.

        Args:
            chunks: Список чанков

        Returns:
            Список названий файлов-источников
        """
        sources = set()

        for chunk in chunks:
            file_name = chunk.metadata.get("file_name")
            if file_name:
                sources.add(file_name)

        return sorted(list(sources))

    def _serialize_chunks(self, chunks, scores) -> list[Dict[str, Any]]:
        """
        Сериализует chunks в словари для передачи в API.

        Args:
            chunks: Список чанков
            scores: Список scores релевантности

        Returns:
            Список словарей с данными чанков
        """
        chunks_data = []

        for idx, chunk in enumerate(chunks):
            chunk_dict = {
                "text": chunk.text,
                "file_name": chunk.metadata.get("file_name", "Unknown"),
                "chunk_id": chunk.chunk_id,
                "score": scores[idx] if idx < len(scores) else None,
                "metadata": chunk.metadata
            }
            chunks_data.append(chunk_dict)

        return chunks_data

    def get_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику orchestrator.

        Returns:
            Словарь со статистикой
        """
        retriever_stats = self.retriever.get_statistics()

        return {
            "orchestrator_type": "SimpleRAG",
            "top_k": self.top_k,
            "retriever": retriever_stats,
            "llm_model": self.llm_client.model
        }

    def __repr__(self) -> str:
        return f"SimpleRAGOrchestrator(top_k={self.top_k})"
