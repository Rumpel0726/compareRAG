# -*- coding: utf-8 -*-
"""
Векторное хранилище на базе ChromaDB.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger
import uuid

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB не установлен")

from ..core.base_vector_store import BaseVectorStore, SearchResult
from ..core.base_chunker import Chunk
from ..utils.exceptions import VectorStoreError


class ChromaStore(BaseVectorStore):
    """
    Векторное хранилище на базе ChromaDB.

    ChromaDB - легковесная векторная БД с persistence (SQLite).
    """

    def __init__(
        self,
        collection_name: str = "documents",
        path: str = "./chroma_db",
        distance_function: str = "cosine",
        **kwargs
    ):
        """
        Инициализация ChromaDB хранилища.

        Args:
            collection_name: Название коллекции
            path: Путь для хранения данных
            distance_function: Функция расстояния ("cosine", "l2", "ip")
            **kwargs: Дополнительные параметры
        """
        super().__init__(collection_name=collection_name, **kwargs)

        if not CHROMA_AVAILABLE:
            raise VectorStoreError(
                "ChromaDB не установлен. Установите: pip install chromadb"
            )

        self.path = path
        self.distance_function = distance_function

        # Создаем директорию если нужно
        Path(path).mkdir(parents=True, exist_ok=True)

        try:
            # Инициализируем ChromaDB с persistence
            self.client = chromadb.PersistentClient(
                path=path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )

            # Получаем или создаем коллекцию
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": distance_function}
            )

            # Обновляем счетчик документов
            self._document_count = self.collection.count()

            logger.info(
                f"ChromaStore инициализирован: collection={collection_name}, "
                f"path={path}, docs={self._document_count}"
            )

        except Exception as e:
            logger.error(f"Ошибка инициализации ChromaDB: {e}")
            raise VectorStoreError(f"Не удалось инициализировать ChromaDB: {e}")

    def add_documents(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Добавляет документы с эмбеддингами в хранилище.

        Args:
            chunks: Список чанков документов
            embeddings: Список векторов эмбеддингов
            ids: Опциональные ID для документов

        Raises:
            VectorStoreError: Если не удалось добавить документы
        """
        if not chunks or not embeddings:
            logger.warning("Пустой список чанков или эмбеддингов")
            return

        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Количество чанков ({len(chunks)}) не совпадает с количеством "
                f"эмбеддингов ({len(embeddings)})"
            )

        try:
            # Генерируем ID если не предоставлены
            if ids is None:
                ids = [chunk.chunk_id or str(uuid.uuid4()) for chunk in chunks]

            # Подготавливаем данные
            documents = [chunk.text for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]

            # ChromaDB имеет ограничение на размер батча (~5461)
            # Разбиваем на батчи по 2000 для безопасности
            BATCH_SIZE = 2000
            total_added = 0

            for i in range(0, len(chunks), BATCH_SIZE):
                batch_end = min(i + BATCH_SIZE, len(chunks))

                # Добавляем батч в ChromaDB
                self.collection.add(
                    ids=ids[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    documents=documents[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )

                total_added += (batch_end - i)
                logger.debug(f"Добавлено {batch_end - i} документов в ChromaDB (всего {total_added}/{len(chunks)})")

            self._document_count += len(chunks)

            logger.info(f"Успешно добавлено {len(chunks)} документов в ChromaDB")

        except Exception as e:
            logger.error(f"Ошибка добавления документов: {e}")
            raise VectorStoreError(f"Не удалось добавить документы: {e}")

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Выполняет поиск похожих документов по вектору запроса.

        Args:
            query_embedding: Вектор запроса
            k: Количество результатов
            filter_dict: Опциональный фильтр по метаданным

        Returns:
            Список результатов поиска

        Raises:
            VectorStoreError: Если не удалось выполнить поиск
        """
        try:
            # Выполняем запрос
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_dict if filter_dict else None,
                include=["documents", "metadatas", "distances"]
            )

            # Преобразуем результаты в SearchResult
            search_results = []

            if results and results["ids"] and len(results["ids"][0]) > 0:
                for idx in range(len(results["ids"][0])):
                    # Извлекаем данные
                    chunk_id = results["ids"][0][idx]
                    text = results["documents"][0][idx]
                    metadata = results["metadatas"][0][idx]
                    distance = results["distances"][0][idx]

                    # Преобразуем distance в score (чем меньше distance, тем больше score)
                    # Для cosine distance: score = 1 - distance
                    if self.distance_function == "cosine":
                        score = 1.0 - distance
                    elif self.distance_function == "l2":
                        score = 1.0 / (1.0 + distance)
                    else:
                        score = distance

                    # Создаем Chunk
                    chunk = Chunk(
                        text=text,
                        metadata=metadata,
                        chunk_id=chunk_id
                    )

                    # Создаем SearchResult
                    search_result = SearchResult(
                        chunk=chunk,
                        score=score,
                        distance=distance
                    )

                    search_results.append(search_result)

            logger.debug(f"Найдено {len(search_results)} результатов")

            return search_results

        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            raise VectorStoreError(f"Не удалось выполнить поиск: {e}")

    def delete_collection(self) -> None:
        """
        Удаляет коллекцию из хранилища.

        Raises:
            VectorStoreError: Если не удалось удалить коллекцию
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self._document_count = 0
            logger.info(f"Коллекция {self.collection_name} удалена")

        except Exception as e:
            logger.error(f"Ошибка удаления коллекции: {e}")
            raise VectorStoreError(f"Не удалось удалить коллекцию: {e}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику коллекции.

        Returns:
            Словарь со статистикой
        """
        try:
            count = self.collection.count()
            self._document_count = count

            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "path": self.path,
                "distance_function": self.distance_function,
            }

        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {
                "collection_name": self.collection_name,
                "error": str(e)
            }

    def get_all_chunks(self) -> List[Chunk]:
        """
        Возвращает все хранимые чанки.

        Используется для построения BM25 индекса.

        Returns:
            Список всех чанков в коллекции

        Raises:
            VectorStoreError: Если не удалось получить чанки
        """
        try:
            data = self.collection.get(include=["documents", "metadatas"])

            chunks = [
                Chunk(text=doc, metadata=meta, chunk_id=doc_id)
                for doc_id, doc, meta in zip(
                    data["ids"], data["documents"], data["metadatas"]
                )
            ]

            logger.debug(f"Получено {len(chunks)} чанков из ChromaDB для BM25 индекса")
            return chunks

        except Exception as e:
            logger.error(f"Ошибка получения всех чанков: {e}")
            raise VectorStoreError(f"Не удалось получить все чанки: {e}")

    def reset(self) -> None:
        """
        Очищает все данные в коллекции (не удаляя саму коллекцию).
        """
        try:
            # Получаем все ID
            all_ids = self.collection.get()["ids"]

            if all_ids:
                # Удаляем все документы
                self.collection.delete(ids=all_ids)
                self._document_count = 0
                logger.info(f"Коллекция {self.collection_name} очищена")

        except Exception as e:
            logger.error(f"Ошибка очистки коллекции: {e}")
            raise VectorStoreError(f"Не удалось очистить коллекцию: {e}")

    def __repr__(self) -> str:
        return (
            f"ChromaStore(collection={self.collection_name}, "
            f"path={self.path}, "
            f"docs={self._document_count})"
        )
