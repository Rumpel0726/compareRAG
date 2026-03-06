# -*- coding: utf-8 -*-
"""
API эндпоинты для чата.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional
from loguru import logger

router = APIRouter()


class ChatRequest(BaseModel):
    """Запрос для чата."""
    query: str = Field(..., min_length=1, description="Вопрос пользователя")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Количество документов")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Температура генерации")


class Source(BaseModel):
    """Источник информации."""
    file_name: str
    score: Optional[float] = None


class ChunkData(BaseModel):
    """Данные чанка."""
    text: str
    file_name: str
    chunk_id: str
    score: Optional[float] = None
    metadata: dict


class ChatResponse(BaseModel):
    """Ответ чата."""
    query: str
    answer: str
    sources: List[str]
    chunks: List[ChunkData]
    retrieval_time: float
    generation_time: float
    total_time: float
    num_chunks_found: int


@router.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest, request: Request):
    """
    Обрабатывает вопрос пользователя и возвращает ответ.

    Args:
        chat_request: Запрос с вопросом
        request: FastAPI Request для доступа к app.state

    Returns:
        Ответ с информацией и источниками

    Raises:
        HTTPException: Если произошла ошибка
    """
    orchestrator = getattr(request.app.state, "orchestrator", None)

    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="RAG система не инициализирована"
        )

    try:
        logger.info(f"Получен запрос: '{chat_request.query[:50]}...'")

        # Обрабатываем запрос через orchestrator
        result = orchestrator.query(
            user_query=chat_request.query,
            k=chat_request.top_k,
            temperature=chat_request.temperature
        )

        logger.info(
            f"Запрос обработан за {result['total_time']}с: "
            f"{result['num_chunks_found']} чанков, "
            f"{len(result['sources'])} источников"
        )

        return ChatResponse(**result)

    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки запроса: {str(e)}"
        )


@router.get("/stats")
async def get_statistics(request: Request):
    """
    Возвращает статистику RAG системы.

    Args:
        request: FastAPI Request для доступа к app.state

    Returns:
        Словарь со статистикой
    """
    orchestrator = getattr(request.app.state, "orchestrator", None)

    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="RAG система не инициализирована"
        )

    try:
        stats = orchestrator.get_statistics()
        return stats

    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка получения статистики: {str(e)}"
        )
