# -*- coding: utf-8 -*-
"""
Система конфигурации для RAG системы с использованием Pydantic.
"""

from typing import Optional, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path
import yaml


class LMStudioConfig(BaseSettings):
    """Конфигурация для LM Studio."""

    url: str = Field(default="http://127.0.0.1:1234", description="URL LM Studio API")
    llm_model: str = Field(default="qwen/qwen3-14b", description="Модель для генерации")
    embedding_model: str = Field(default="text-embedding-nomic-embed-text-v1.5", description="Модель для эмбеддингов")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Температура генерации")
    max_tokens: int = Field(default=2000, ge=1, description="Максимальное количество токенов")
    timeout: int = Field(default=120, description="Таймаут запросов в секундах")

    class Config:
        env_prefix = "LM_STUDIO_"


class ChromaDBConfig(BaseSettings):
    """Конфигурация для ChromaDB."""

    path: str = Field(default="./chroma_db", description="Путь к ChromaDB")
    collection_name: str = Field(default="medical_docs", description="Имя коллекции")
    distance_function: str = Field(default="cosine", description="Функция расстояния")

    class Config:
        env_prefix = "CHROMADB_"


class ChunkingConfig(BaseSettings):
    """Конфигурация для чанкинга."""

    strategy: str = Field(default="chonkie", description="Стратегия чанкинга")
    chunk_size: int = Field(default=512, ge=1, description="Размер чанка")
    chunk_overlap: int = Field(default=50, ge=0, description="Перекрытие чанков")

    class Config:
        env_prefix = "CHUNKING_"


class RetrievalConfig(BaseSettings):
    """Конфигурация для поиска."""

    strategy: str = Field(default="semantic", description="Стратегия поиска")
    top_k: int = Field(default=5, ge=1, description="Количество результатов")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Порог score")

    class Config:
        env_prefix = "RETRIEVAL_"


class LoggingConfig(BaseSettings):
    """Конфигурация для логирования."""

    level: str = Field(default="INFO", description="Уровень логирования")
    log_dir: str = Field(default="./logs", description="Директория для логов")
    rotation: str = Field(default="500 MB", description="Ротация логов")
    retention: str = Field(default="10 days", description="Время хранения логов")

    class Config:
        env_prefix = "LOG_"


class RAGConfig(BaseSettings):
    """Главная конфигурация RAG системы."""

    # Подконфигурации
    lm_studio: LMStudioConfig = Field(default_factory=LMStudioConfig)
    chromadb: ChromaDBConfig = Field(default_factory=ChromaDBConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Общие настройки
    data_dir: str = Field(default="./data", description="Директория с данными")
    results_dir: str = Field(default="./results", description="Директория для результатов")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RAGConfig":
        """
        Загружает конфигурацию из YAML файла.

        Args:
            yaml_path: Путь к YAML файлу

        Returns:
            Экземпляр RAGConfig
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, yaml_path: str) -> None:
        """
        Сохраняет конфигурацию в YAML файл.

        Args:
            yaml_path: Путь для сохранения
        """
        config_dict = self.model_dump()

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    def merge_with_dict(self, config_dict: Dict[str, Any]) -> "RAGConfig":
        """
        Объединяет текущую конфигурацию со словарем.

        Args:
            config_dict: Словарь с параметрами

        Returns:
            Новый экземпляр конфигурации
        """
        current_dict = self.model_dump()
        current_dict.update(config_dict)
        return RAGConfig(**current_dict)


def load_config(config_path: Optional[str] = None) -> RAGConfig:
    """
    Загружает конфигурацию из файла или переменных окружения.

    Порядок приоритета:
    1. YAML файл (если указан)
    2. Переменные окружения
    3. Значения по умолчанию

    Args:
        config_path: Опциональный путь к YAML конфигу

    Returns:
        Экземпляр RAGConfig
    """
    if config_path and Path(config_path).exists():
        return RAGConfig.from_yaml(config_path)
    else:
        return RAGConfig()


# Глобальный экземпляр конфигурации
_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """
    Возвращает глобальный экземпляр конфигурации.

    Returns:
        RAGConfig
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: RAGConfig) -> None:
    """
    Устанавливает глобальную конфигурацию.

    Args:
        config: Экземпляр RAGConfig
    """
    global _config
    _config = config
