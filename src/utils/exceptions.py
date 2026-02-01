# -*- coding: utf-8 -*-
"""
Кастомные исключения для системы сравнения RAG.
"""


class RAGException(Exception):
    """Базовое исключение для всех RAG ошибок."""
    pass


class DocumentProcessingError(RAGException):
    """Ошибка при обработке документов."""
    pass


class ChunkingError(RAGException):
    """Ошибка при чанкинге текста."""
    pass


class EmbeddingError(RAGException):
    """Ошибка при генерации эмбеддингов."""
    pass


class VectorStoreError(RAGException):
    """Ошибка при работе с векторным хранилищем."""
    pass


class RetrievalError(RAGException):
    """Ошибка при извлечении документов."""
    pass


class LLMError(RAGException):
    """Ошибка при работе с LLM."""
    pass


class ConfigurationError(RAGException):
    """Ошибка конфигурации."""
    pass


class AgentError(RAGException):
    """Ошибка в работе агента."""
    pass


class EvaluationError(RAGException):
    """Ошибка при оценке RAG системы."""
    pass


class RAGPipelineError(RAGException):
    """Ошибка в RAG pipeline."""
    pass
