# -*- coding: utf-8 -*-
"""
Скрипт для индексации PDF документов в векторное хранилище.

Загружает документы, разбивает на чанки, генерирует эмбеддинги
и сохраняет в ChromaDB для последующего поиска.
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from tqdm import tqdm

from src.core.config import load_config
from src.core.base_chunker import Chunk
from src.document_processing.pdf_loader import PDFLoader
from src.chunking.factory import ChunkingStrategyFactory
from src.embeddings.factory import create_embedder_from_config
from src.vector_stores.factory import create_vector_store_from_config
from src.utils.logging_config import setup_logging
from src.utils.exceptions import (
    DocumentProcessingError,
    ChunkingError,
    EmbeddingError,
    VectorStoreError,
)


def parse_arguments():
    """
    Парсинг аргументов командной строки.

    Returns:
        argparse.Namespace с аргументами
    """
    parser = argparse.ArgumentParser(
        description="Индексация PDF документов для RAG системы",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Путь к YAML конфигурации"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Путь к директории с PDF документами"
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Очистить существующую коллекцию перед индексацией"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Размер батча для генерации эмбеддингов"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Уровень логирования"
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Модель эмбеддингов (переопределяет конфиг). "
             "Пример: text-embedding-qwen3-embedding-0.6b. "
             "Имя коллекции автоматически дополняется суффиксом модели."
    )

    parser.add_argument(
        "--add-eos-token",
        action="store_true",
        default=False,
        help="Добавлять EOS-токен в конец каждого текста перед эмбеддингом. "
             "Нужно для некоторых моделей (например qwen3-embedding)."
    )

    parser.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="EOS-токен для добавления (по умолчанию из конфига: </s>). "
             "Пример для Qwen3: '<|endoftext|>'"
    )

    return parser.parse_args()


def validate_paths(config_path: str, data_dir: str) -> None:
    """
    Проверяет существование путей к конфигу и данным.

    Args:
        config_path: Путь к конфигурации
        data_dir: Путь к директории с данными

    Raises:
        FileNotFoundError: Если пути не существуют
    """
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Конфигурация не найдена: {config_path}")

    data_dir_obj = Path(data_dir)
    if not data_dir_obj.exists():
        raise FileNotFoundError(f"Директория с данными не найдена: {data_dir}")

    # Проверяем наличие PDF файлов
    pdf_files = list(data_dir_obj.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"В директории {data_dir} нет PDF файлов")

    logger.info(f"Найдено {len(pdf_files)} PDF файлов в {data_dir}")


def process_documents(
    pdf_loader: PDFLoader,
    chunker,
    data_dir: str
) -> List[Chunk]:
    """
    Загружает и чанкует документы.

    Args:
        pdf_loader: Загрузчик PDF
        chunker: Чанкер документов
        data_dir: Путь к директории с PDF

    Returns:
        Список чанков

    Raises:
        DocumentProcessingError: Если не удалось обработать документы
    """
    try:
        logger.info(f"Загрузка документов из {data_dir}...")

        # Загружаем все PDF
        documents = pdf_loader.load_directory(data_dir)
        logger.info(f"Загружено {len(documents)} документов")

        # Чанкуем документы с progress bar
        logger.info("Чанкинг документов...")
        all_chunks = []

        for doc in tqdm(documents, desc="Чанкинг", unit="doc"):
            try:
                chunks = chunker.chunk(doc)
                all_chunks.extend(chunks)
                logger.debug(
                    f"Документ {doc.metadata.get('file_name', 'unknown')}: "
                    f"{len(chunks)} чанков"
                )
            except Exception as e:
                logger.error(
                    f"Ошибка чанкинга документа "
                    f"{doc.metadata.get('file_name', 'unknown')}: {e}"
                )
                # Продолжаем обработку остальных документов

        if not all_chunks:
            raise ChunkingError("Не удалось создать ни одного чанка")

        logger.info(f"Создано {len(all_chunks)} чанков")

        # Статистика по чанкам
        avg_chunk_length = sum(len(c.text) for c in all_chunks) / len(all_chunks)
        logger.info(f"Средняя длина чанка: {avg_chunk_length:.0f} символов")

        return all_chunks

    except Exception as e:
        logger.error(f"Ошибка обработки документов: {e}")
        raise DocumentProcessingError(f"Не удалось обработать документы: {e}")


def generate_embeddings(
    embedder,
    chunks: List[Chunk],
    batch_size: int = 32
) -> List[List[float]]:
    """
    Генерирует эмбеддинги для чанков.

    Args:
        embedder: Генератор эмбеддингов
        chunks: Список чанков
        batch_size: Размер батча

    Returns:
        Список эмбеддингов

    Raises:
        EmbeddingError: Если не удалось сгенерировать эмбеддинги
    """
    try:
        logger.info(f"Генерация эмбеддингов для {len(chunks)} чанков...")

        # Извлекаем тексты
        texts = [chunk.text for chunk in chunks]

        # Генерируем эмбеддинги с progress bar
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Эмбеддинги", unit="batch"):
            batch = texts[i:i + batch_size]
            batch_embeddings = embedder.embed_documents(batch)
            embeddings.extend(batch_embeddings)

        logger.info(f"Сгенерировано {len(embeddings)} эмбеддингов")

        # Проверяем размерность
        if embeddings:
            embedding_dim = len(embeddings[0])
            logger.info(f"Размерность эмбеддингов: {embedding_dim}")

        return embeddings

    except Exception as e:
        logger.error(f"Ошибка генерации эмбеддингов: {e}")
        raise EmbeddingError(f"Не удалось сгенерировать эмбеддинги: {e}")


def store_documents(
    vector_store,
    chunks: List[Chunk],
    embeddings: List[List[float]],
    reset: bool = False
) -> None:
    """
    Сохраняет документы в векторное хранилище.

    Args:
        vector_store: Векторное хранилище
        chunks: Список чанков
        embeddings: Список эмбеддингов
        reset: Очистить коллекцию перед сохранением

    Raises:
        VectorStoreError: Если не удалось сохранить документы
    """
    try:
        # Очистка коллекции если нужно
        if reset:
            logger.warning("Очистка существующей коллекции...")
            vector_store.reset()
            logger.info("Коллекция очищена")

        # Сохраняем документы
        logger.info(f"Сохранение {len(chunks)} документов в векторное хранилище...")

        vector_store.add_documents(
            chunks=chunks,
            embeddings=embeddings
        )

        logger.info("Документы успешно сохранены")

        # Выводим статистику
        stats = vector_store.get_collection_stats()
        logger.info(f"Статистика коллекции: {stats}")

    except Exception as e:
        logger.error(f"Ошибка сохранения документов: {e}")
        raise VectorStoreError(f"Не удалось сохранить документы: {e}")


def main():
    """
    Основная функция скрипта индексации.
    """
    # Парсинг аргументов
    args = parse_arguments()

    # Настройка логирования
    setup_logging(log_level=args.log_level)

    logger.info("=" * 60)
    logger.info("Запуск индексации документов")
    logger.info("=" * 60)
    logger.info(f"Конфигурация: {args.config}")
    logger.info(f"Директория с данными: {args.data_dir}")
    logger.info(f"Очистка коллекции: {args.reset}")
    logger.info(f"Размер батча: {args.batch_size}")

    try:
        # Валидация путей
        validate_paths(args.config, args.data_dir)

        # Загрузка конфигурации
        logger.info("Загрузка конфигурации...")
        config = load_config(args.config)

        # Переопределение модели эмбеддингов (если задана через CLI)
        if args.embedding_model:
            config.lm_studio.embedding_model = args.embedding_model
            model_id = config.lm_studio.embedding_model_id
            config.chromadb.collection_name = f"{config.chromadb.collection_name}_{model_id}"
            logger.info(f"Модель эмбеддингов (CLI): {args.embedding_model}")
            logger.info(f"Коллекция (с суффиксом модели): {config.chromadb.collection_name}")

        if args.add_eos_token:
            config.lm_studio.add_eos_token = True
            logger.info("EOS-токен включён")
        if args.eos_token:
            config.lm_studio.eos_token = args.eos_token
            logger.info(f"EOS-токен: {args.eos_token}")

        logger.info(f"Стратегия чанкинга: {config.chunking.strategy}")
        logger.info(f"Размер чанка: {config.chunking.chunk_size}")
        logger.info(f"Перекрытие: {config.chunking.chunk_overlap}")
        logger.info(f"Модель эмбеддингов: {config.lm_studio.embedding_model}")
        logger.info(f"Коллекция: {config.chromadb.collection_name}")

        # Инициализация компонентов
        logger.info("Инициализация компонентов...")

        pdf_loader = PDFLoader()
        logger.debug("PDF loader инициализирован")

        chunker = ChunkingStrategyFactory.create(
            strategy_name=config.chunking.strategy,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap
        )
        logger.debug(f"Chunker инициализирован: {chunker}")

        embedder = create_embedder_from_config(config)
        logger.debug(f"Embedder инициализирован: {embedder}")

        vector_store = create_vector_store_from_config(config)
        logger.debug(f"Vector store инициализирован: {vector_store}")

        # Pipeline индексации
        logger.info("\n" + "=" * 60)
        logger.info("Этап 1/3: Загрузка и чанкинг документов")
        logger.info("=" * 60)
        chunks = process_documents(pdf_loader, chunker, args.data_dir)

        logger.info("\n" + "=" * 60)
        logger.info("Этап 2/3: Генерация эмбеддингов")
        logger.info("=" * 60)
        embeddings = generate_embeddings(embedder, chunks, args.batch_size)

        logger.info("\n" + "=" * 60)
        logger.info("Этап 3/3: Сохранение в векторное хранилище")
        logger.info("=" * 60)
        store_documents(vector_store, chunks, embeddings, args.reset)

        # Финальная статистика
        logger.info("\n" + "=" * 60)
        logger.info("✅ Индексация завершена успешно!")
        logger.info("=" * 60)
        logger.info(f"Обработано документов: {len(set(c.metadata.get('file_name') for c in chunks))}")
        logger.info(f"Создано чанков: {len(chunks)}")
        logger.info(f"Сгенерировано эмбеддингов: {len(embeddings)}")
        logger.info(f"Коллекция: {vector_store.collection_name}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"❌ Ошибка: {e}")
        return 1

    except (DocumentProcessingError, ChunkingError, EmbeddingError, VectorStoreError) as e:
        logger.error(f"❌ Ошибка обработки: {e}")
        return 1

    except Exception as e:
        logger.exception(f"❌ Неожиданная ошибка: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
