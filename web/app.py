# -*- coding: utf-8 -*-
"""
FastAPI приложение для RAG системы.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys

# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.core.config import load_config
from src.embeddings.factory import create_embedder_from_config
from src.vector_stores.factory import create_vector_store_from_config
from src.retrieval.factory import create_retriever_from_config
from src.llm.lm_studio_client import LMStudioClient
from src.agents.orchestrator import SimpleRAGOrchestrator
from src.utils.logging_config import setup_logging

# Настройка логирования
setup_logging(log_level="INFO")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом приложения.
    """

    # Startup
    try:
        logger.info("=" * 60)
        logger.info("Запуск RAG системы...")
        logger.info("=" * 60)

        # Загрузка конфигурации
        config_path = project_root / "configs" / "default.yaml"
        logger.info(f"Загрузка конфигурации из {config_path}")
        config = load_config(str(config_path))

        # Инициализация компонентов
        logger.info("Инициализация эмбеддера...")
        embedder = create_embedder_from_config(config)

        logger.info("Инициализация векторного хранилища...")
        # Преобразуем относительный путь к ChromaDB в абсолютный относительно корня проекта
        chroma_path = Path(config.chromadb.path)
        if not chroma_path.is_absolute():
            chroma_path = project_root / config.chromadb.path

        # Создаем временную копию конфигурации с абсолютным путём
        import copy
        config_copy = copy.deepcopy(config)
        config_copy.chromadb.path = str(chroma_path)

        vector_store = create_vector_store_from_config(config_copy)

        # Проверяем наличие документов в векторном хранилище
        stats = vector_store.get_collection_stats()
        logger.info(f"Статистика коллекции: {stats}")

        if stats.get("document_count", 0) == 0:
            logger.warning(
                "⚠️  Векторное хранилище пусто! "
                "Запустите скрипт индексации: python scripts/ingest_documents.py"
            )

        logger.info("Инициализация ретривера...")
        retriever = create_retriever_from_config(config, embedder, vector_store)

        logger.info("Инициализация LLM клиента...")
        llm_client = LMStudioClient(
            url=config.lm_studio.url,
            model=config.lm_studio.llm_model,
            temperature=config.lm_studio.temperature,
            max_tokens=config.lm_studio.max_tokens,
            timeout=config.lm_studio.timeout,
            api_key=config.lm_studio.api_key
        )

        logger.info("Инициализация orchestrator...")
        app.state.orchestrator = SimpleRAGOrchestrator(
            retriever=retriever,
            llm_client=llm_client,
            top_k=config.retrieval.top_k
        )

        logger.info("=" * 60)
        logger.info("✅ RAG система успешно запущена!")
        logger.info("=" * 60)
        logger.info(f"Модель генерации: {config.lm_studio.llm_model}")
        logger.info(f"Модель эмбеддингов: {config.lm_studio.embedding_model}")
        logger.info(f"Коллекция: {config.chromadb.collection_name}")
        logger.info(f"Документов в базе: {stats.get('document_count', 0)}")

    except Exception as e:
        logger.exception(f"❌ Ошибка инициализации RAG системы: {e}")
        raise

    yield  # Приложение работает

    # Shutdown
    logger.info("Остановка RAG системы...")


# Создаем FastAPI приложение
app = FastAPI(
    title="compareRAG",
    description="Система сравнения стратегий RAG для медицинских документов",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware для разработки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройка путей
web_dir = Path(__file__).parent
static_dir = web_dir / "static"
templates_dir = web_dir / "templates"

# Статические файлы и шаблоны
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))


@app.get("/")
async def index(request: Request):
    """
    Главная страница с чат-интерфейсом.
    """
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "compareRAG - Медицинский ассистент"}
    )


@app.get("/health")
async def health_check(request: Request):
    """
    Проверка здоровья системы.
    """
    orchestrator = getattr(request.app.state, "orchestrator", None)

    if orchestrator is None:
        return {
            "status": "error",
            "message": "Orchestrator не инициализирован"
        }

    try:
        stats = orchestrator.get_statistics()
        return {
            "status": "ok",
            "message": "Система работает",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Ошибка health check: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


# Импортируем chat API routes
from web.api import chat
app.include_router(chat.router, prefix="/api", tags=["chat"])


if __name__ == "__main__":
    import uvicorn

    # Для Windows лучше запускать без reload из-за проблем с multiprocessing
    # Для dev режима используйте: uvicorn app:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(
        app,  # Передаем объект напрямую, а не строку
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
