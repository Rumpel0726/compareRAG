# compareRAG - Система сравнения стратегий RAG

Модульная система для сравнения различных стратегий RAG (Retrieval-Augmented Generation) на базе медицинских документов на русском языке.

## Особенности

- 🔌 **Модульная архитектура** - Strategy Pattern для легкой смены компонентов
- 🤖 **Агентная система** - RAG pipeline с orchestrator и synthesis agent
- 🌐 **Веб-интерфейс** - Чат на FastAPI с красивым UI
- 🏠 **Локальные модели** - Работает с LM Studio (без отправки данных в облако)
- 🇷🇺 **Поддержка русского языка** - Оптимизация для кириллицы
- 📊 **Сравнение стратегий** - Простое переключение между конфигурациями

## Технологический стек

### Backend
- **Python 3.10+**
- **FastAPI** - async веб-фреймворк
- **ChromaDB** - векторная база данных
- **Chonkie** - основной чанкинг (fallback: LangChain)
- **Pydantic** - валидация конфигурации
- **Loguru** - структурированное логирование

### LLM & Embeddings
- **LM Studio** (http://127.0.0.1:1234)
  - Модель генерации: `qwen/qwen3-14b`
  - Модель эмбеддингов: `text-embedding-nomic-embed-text-v1.5`

### Frontend
- **Vanilla JavaScript** - без фреймворков
- **Modern CSS** - адаптивный дизайн

## Структура проекта

```
compareRAG/
├── data/                      # 10 медицинских PDF (4.9MB)
├── src/
│   ├── core/                  # Базовые абстракции (Strategy Pattern)
│   ├── document_processing/   # Загрузка и обработка PDF
│   ├── chunking/              # Стратегии чанкинга
│   ├── embeddings/            # Генерация эмбеддингов
│   ├── vector_stores/         # ChromaDB интеграция
│   ├── retrieval/             # Семантический поиск
│   ├── llm/                   # LM Studio клиент и промпты
│   ├── agents/                # Агентная архитектура
│   └── utils/                 # Логирование и исключения
├── web/                       # FastAPI приложение
│   ├── app.py                 # Основное приложение
│   ├── api/chat.py            # Chat endpoints
│   ├── templates/             # HTML шаблоны
│   └── static/                # CSS и JavaScript
├── scripts/                   # Скрипты запуска
│   └── ingest_documents.py    # Индексация документов
├── configs/                   # YAML конфигурации
│   ├── default.yaml           # Базовая конфигурация
│   └── strategies/            # Стратегии для сравнения
│       ├── baseline.yaml      # chunk_size=512
│       ├── large_chunks.yaml  # chunk_size=1024
│       └── small_chunks.yaml  # chunk_size=256
└── chroma_db/                 # База векторов (создается автоматически)
```

## Установка

### 1. Предварительные требования

- **Python 3.10+**
- **LM Studio** - https://lmstudio.ai/
  - Загрузите модель генерации: `qwen/qwen3-14b`
  - Загрузите модель эмбеддингов: `text-embedding-nomic-embed-text-v1.5`
  - Запустите локальный сервер на http://127.0.0.1:1234

### 2. Клонирование и установка зависимостей

```bash
# Перейдите в директорию проекта
cd compareRAG

# Создайте виртуальное окружение (если еще не создано)
python -m venv venv

# Активируйте окружение
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Установите зависимости (если еще не установлены)
pip install -r requirements.txt
```

### 3. Проверка установки LM Studio

Убедитесь, что LM Studio запущен и доступен:

```bash
curl http://127.0.0.1:1234/v1/models
```

Должны увидеть список доступных моделей.

## Использование

### Шаг 1: Индексация документов

Перед первым запуском нужно проиндексировать PDF документы:

```bash
# Базовая стратегия (chunk_size=512)
python scripts/ingest_documents.py --config configs/strategies/baseline.yaml

# Или с другими параметрами
python scripts/ingest_documents.py --config configs/strategies/large_chunks.yaml
python scripts/ingest_documents.py --config configs/strategies/small_chunks.yaml

# Доступные опции:
# --config         Путь к YAML конфигурации (default: configs/default.yaml)
# --data-dir       Путь к PDF файлам (default: ./data)
# --reset          Очистить коллекцию перед индексацией
# --batch-size     Размер батча для эмбеддингов (default: 32)
# --log-level      Уровень логов (DEBUG, INFO, WARNING, ERROR)
```

Ожидаемый вывод:

```
============================================================
Запуск индексации документов
============================================================
Этап 1/3: Загрузка и чанкинг документов
Этап 2/3: Генерация эмбеддингов
Этап 3/3: Сохранение в векторное хранилище
✅ Индексация завершена успешно!
Обработано документов: 10
Создано чанков: ~300-400 (зависит от стратегии)
```

### Шаг 2: Запуск веб-интерфейса

```bash
# Запуск FastAPI сервера
cd web
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Или через Python
python app.py
```

Откройте браузер: **http://localhost:8000**

### Шаг 3: Использование чата

1. Откройте http://localhost:8000
2. Дождитесь подключения к системе (зеленый индикатор)
3. Задайте вопрос на русском языке, например:
   - "Какие методы лечения описаны в документах?"
   - "Расскажи о медикаментозной токсикодермии"
   - "Какие исследования проводились с пациентами?"

Параметры запроса:
- **Количество документов (top_k)**: 1-20 (сколько чанков использовать)
- **Температура**: 0.0-2.0 (креативность генерации)

## Конфигурация

### Основной файл: `configs/default.yaml`

```yaml
# LM Studio
lm_studio:
  url: "http://127.0.0.1:1234"
  llm_model: "qwen/qwen3-14b"
  embedding_model: "text-embedding-nomic-embed-text-v1.5"
  temperature: 0.7
  max_tokens: 2000

# ChromaDB
chromadb:
  path: "./chroma_db"
  collection_name: "medical_docs"
  distance_function: "cosine"

# Чанкинг
chunking:
  strategy: "chonkie"  # или "langchain"
  chunk_size: 512
  chunk_overlap: 50

# Поиск
retrieval:
  strategy: "semantic"
  top_k: 5
  score_threshold: 0.0
```

### Сравнение стратегий чанкинга

Проект поддерживает быстрое переключение между стратегиями:

| Стратегия | Размер чанка | Описание |
|-----------|-------------|----------|
| **baseline.yaml** | 512 | Сбалансированный вариант |
| **large_chunks.yaml** | 1024 | Больше контекста, меньше чанков |
| **small_chunks.yaml** | 256 | Точный поиск, больше чанков |

Для сравнения:

```bash
# 1. Индексируйте с baseline
python scripts/ingest_documents.py --config configs/strategies/baseline.yaml --reset

# 2. Протестируйте в веб-интерфейсе

# 3. Переиндексируйте с large_chunks
python scripts/ingest_documents.py --config configs/strategies/large_chunks.yaml --reset

# 4. Сравните результаты
```

## API Endpoints

### POST `/api/chat`

Отправка вопроса в RAG систему.

**Request:**
```json
{
  "query": "Какие исследования описаны?",
  "top_k": 5,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "query": "Какие исследования описаны?",
  "answer": "В документах описаны следующие исследования...",
  "sources": ["document1.pdf", "document2.pdf"],
  "retrieval_time": 0.5,
  "generation_time": 2.3,
  "total_time": 2.8,
  "num_chunks_found": 5
}
```

### GET `/health`

Проверка состояния системы.

**Response:**
```json
{
  "status": "ok",
  "message": "Система работает",
  "statistics": {
    "retriever_type": "semantic",
    "top_k": 5,
    "vector_store": {
      "count": 350,
      "collection_name": "medical_docs"
    },
    "llm_model": "qwen/qwen3-14b"
  }
}
```

### GET `/api/stats`

Статистика RAG системы.

## Логирование

Логи сохраняются в директорию `logs/`:

- `app.log` - основные логи (ротация 500 MB)
- `errors.log` - только ошибки
- `metrics.jsonl` - метрики производительности (JSON Lines)

Уровни логирования: DEBUG, INFO, WARNING, ERROR

```bash
# Изменить уровень логирования
python scripts/ingest_documents.py --log-level DEBUG
```

## Решение проблем

### LM Studio недоступен

```
ERROR: LM Studio недоступен по адресу http://127.0.0.1:1234
```

**Решение:**
1. Убедитесь, что LM Studio запущен
2. Проверьте, что обе модели загружены
3. Проверьте порт: Settings → Server → Port 1234

### Векторное хранилище пусто

```
WARNING: Векторное хранилище пусто!
```

**Решение:**
Запустите индексацию:
```bash
python scripts/ingest_documents.py
```

### Ошибка импорта Chonkie

```
WARNING: Chonkie недоступен, переключение на LangChain fallback
```

**Решение:**
Это нормально - система автоматически переключится на LangChain.
Если хотите использовать Chonkie:
```bash
pip install chonkie
```

### Проблемы с кодировкой в Windows

Если видите �� вместо русских символов:

**Решение:**
1. Убедитесь что терминал использует UTF-8
2. В PowerShell: `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`

## Архитектурные решения

### Strategy Pattern

Все компоненты используют паттерн Стратегия:
- **Чанкинг**: `ChonkieChunker` ↔ `LangChainChunker`
- **Эмбеддинги**: `LMStudioEmbedder` (можно добавить другие)
- **Векторные БД**: `ChromaStore` (можно добавить FAISS, Pinecone)
- **Поиск**: `SemanticRetriever` (можно добавить Hybrid, Re-ranking)

### RAG Pipeline

```
Запрос пользователя
    ↓
[Orchestrator]
    ↓
[SemanticRetriever] → Поиск в ChromaDB (top_k чанков)
    ↓
[SynthesisAgent] → Генерация ответа через LM Studio
    ↓
Ответ с источниками
```

### Конфигурация через YAML

Все параметры вынесены в YAML файлы для легкого переключения между стратегиями без изменения кода.

## Расширение функциональности

### Добавление новой стратегии чанкинга

1. Создайте класс, наследующий `BaseChunker`:

```python
# src/chunking/my_chunker.py
from src.core.base_chunker import BaseChunker, Chunk

class MyChunker(BaseChunker):
    def chunk(self, document) -> List[Chunk]:
        # Ваша логика чанкинга
        pass
```

2. Зарегистрируйте в factory:

```python
# src/chunking/factory.py
CHUNKER_REGISTRY["my_chunker"] = MyChunker
```

3. Используйте в конфигурации:

```yaml
chunking:
  strategy: "my_chunker"
```

## Дальнейшее развитие

- [ ] Система evaluation (RAGAS метрики)
- [ ] Hybrid retrieval (semantic + keyword)
- [ ] Re-ranking результатов
- [ ] Query analyzer agent
- [ ] Streaming ответов в чате
- [ ] Сохранение истории диалогов
- [ ] Экспорт результатов сравнения

## Лицензия

MIT

## Автор

Проект создан для сравнения различных стратегий RAG на медицинских документах.

---

**Примечание**: Данные в `data/` - медицинские статьи на русском языке. Используйте ответы системы только в образовательных целях.
