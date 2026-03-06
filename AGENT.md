# AGENT.md — compareRAG

Модульная система сравнения стратегий RAG на 10 медицинских PDF (русский язык).
Всё локально через LM Studio — облачные API не используются.

---

## Предварительные условия

- **LM Studio** запущен на `http://127.0.0.1:1234` с двумя моделями:
  - `qwen/qwen3-14b` — генерация ответов
  - `text-embedding-nomic-embed-text-v1.5` — эмбеддинги (768-dim)
- Python 3.10+, виртуальное окружение: `venv\Scripts\activate`

## Быстрый старт

```bash
venv\Scripts\activate
python scripts/ingest_documents.py --config configs/strategies/baseline.yaml
cd web && python app.py                           # → http://localhost:8000
python scripts/run_evaluation.py                  # → logs/evaluation_<date>.log
```

---

## Стек

| Слой | Что |
|---|---|
| Чанкинг | Chonkie (primary) / LangChain (fallback) |
| Эмбеддинги | LM Studio `/v1/embeddings` (OpenAI-compatible) |
| Векторная БД | ChromaDB, SQLite backend, cosine distance |
| LLM | LM Studio `/v1/chat/completions` (qwen3-14b) |
| Web | FastAPI + Jinja2 + Vanilla JS |
| Config | Pydantic + YAML |
| Logging | Loguru (rotation 500MB, retention 10d, JSON metrics) |

---

## Структура проекта

```
compareRAG/
├── configs/
│   ├── default.yaml                  # базовая конфигурация
│   └── strategies/
│       ├── baseline.yaml             # chunk_size=512,  overlap=50,  collection=medical_docs_baseline
│       ├── large_chunks.yaml         # chunk_size=1024, overlap=100, collection=medical_docs_large
│       └── small_chunks.yaml         # chunk_size=256,  overlap=25,  collection=medical_docs_small
├── data/                             # 10 медицинских PDF (русский язык, 4.9 MB)
├── src/
│   ├── core/
│   │   ├── base_chunker.py           # Document, Chunk dataclasses; BaseChunker ABC
│   │   ├── base_embedder.py          # BaseEmbedder ABC (batch support)
│   │   ├── base_vector_store.py      # SearchResult dataclass; BaseVectorStore ABC
│   │   ├── base_retriever.py         # RetrievalResult dataclass; BaseRetriever ABC
│   │   └── config.py                 # RAGConfig (Pydantic nested model), load_config(yaml_path)
│   ├── document_processing/
│   │   ├── pdf_loader.py             # PDFLoader: PyPDF, русский язык, page markers [Страница N]
│   │   ├── text_cleaner.py           # Unicode-нормализация, clean_for_embedding()
│   │   └── metadata_extractor.py     # Извлечение title, author, subject из PDF
│   ├── chunking/
│   │   ├── chonkie_chunker.py        # TokenChunker — основной чанкер
│   │   ├── langchain_chunker.py      # RecursiveCharacterTextSplitter — fallback
│   │   └── factory.py                # ChunkingStrategyFactory.create(strategy, **kwargs)
│   ├── embeddings/
│   │   ├── lm_studio_embedder.py     # HTTP POST /v1/embeddings, batch=32, retry×3 expo backoff
│   │   └── factory.py                # create_embedder_from_config(config)
│   ├── vector_stores/
│   │   ├── chroma_store.py           # ChromaDB: persistence, metadata filtering, collection mgmt
│   │   └── factory.py                # create_vector_store_from_config(config)
│   ├── retrieval/
│   │   ├── semantic_retriever.py     # Cosine sim search, top-k, score_threshold
│   │   ├── bm25_retriever.py         # BM25Okapi + pymorphy2 лемматизация, индекс из ChromaDB
│   │   ├── hybrid_retriever.py       # SemanticRetriever + BM25Retriever → RRF fusion
│   │   └── factory.py                # create_retriever_from_config(config, embedder, store)
│   ├── llm/
│   │   ├── lm_studio_client.py       # HTTP POST /v1/chat/completions, retry×3, timeout=120s
│   │   └── prompts.py                # SYSTEM_PROMPT, SYNTHESIS_PROMPT, ANSWER_EVALUATION_PROMPT (RU)
│   ├── agents/
│   │   ├── base_agent.py             # AgentResult(success, data, error, metadata); BaseAgent ABC
│   │   ├── synthesis_agent.py        # Генерация ответа: контекст из chunks → LLM → текст
│   │   └── orchestrator.py           # SimpleRAGOrchestrator.query() — координация всего pipeline
│   ├── evaluation/
│   │   ├── questions.py              # 10 EvalQuestion (question, expected_sources=имя PDF)
│   │   ├── retrieval_metrics.py      # precision_at_k, recall_at_k, MRR, ndcg_at_k, compute_all()
│   │   └── generation_metrics.py     # LLMJudge: 4 критерия (1–5), парсит JSON из ответа LLM
│   └── utils/
│       ├── logging_config.py         # Loguru: console + file rotation + JSON metrics sink
│       └── exceptions.py             # 11 кастомных исключений (DocumentProcessingError, ...)
├── scripts/
│   ├── ingest_documents.py           # CLI: PDF → chunks → embeddings → ChromaDB
│   └── run_evaluation.py             # CLI: 10 вопросов × full pipeline → метрики в лог
├── web/
│   ├── app.py                        # FastAPI, lifespan init всех компонентов, CORS
│   ├── api/
│   │   └── chat.py                   # POST /api/chat, GET /health, GET /api/stats
│   ├── templates/
│   │   └── index.html                # Чат UI (Jinja2 template)
│   └── static/
│       ├── css/styles.css
│       └── js/chat.js                # Fetch API, auto-resize, health check при старте
├── chroma_db/                        # Персистент ChromaDB (SQLite), не в git
├── logs/                             # Логи: rag_*.log, errors_*.log, evaluation_*.log, metrics_*.json
└── requirements.txt
```

---

## Поток данных

```
Запрос пользователя
  → POST /api/chat                          (web/api/chat.py)
    → SimpleRAGOrchestrator.query()         (src/agents/orchestrator.py)
        ├── [SemanticRetriever | HybridRetriever].retrieve()
        │     Semantic: LMStudioEmbedder.embed() → ChromaStore.search() → top-k
        │     Hybrid:   Semantic(top_k×3) + BM25(top_k×3) → RRF fusion → top-k
        └── SynthesisAgent.execute()        (src/agents/synthesis_agent.py)
              └── LMStudioClient.generate() → текст ответа
    ← {query, answer, sources, scores, retrieval_time, generation_time}
```

---

## Конфигурация

Всё через YAML + Pydantic. Загрузка:

```python
from src.core.config import load_config

config = load_config("configs/strategies/baseline.yaml")
config.lm_studio.url              # http://127.0.0.1:1234
config.lm_studio.llm_model        # qwen/qwen3-14b
config.chunking.chunk_size        # 512
config.chunking.strategy          # chonkie
config.chromadb.collection_name   # medical_docs_baseline
config.retrieval.top_k            # 5
```

Каждая стратегия — отдельная ChromaDB коллекция. При смене стратегии нужна переиндексация с `--reset`.

## Фабрики компонентов

```python
from src.chunking.factory import ChunkingStrategyFactory
from src.embeddings.factory import create_embedder_from_config
from src.vector_stores.factory import create_vector_store_from_config
from src.retrieval.factory import create_retriever_from_config

chunker       = ChunkingStrategyFactory.create("chonkie", chunk_size=512)
embedder      = create_embedder_from_config(config)
vector_store  = create_vector_store_from_config(config)
retriever     = create_retriever_from_config(config, embedder, vector_store)
```

---

## API

| Метод | Путь | Request | Response |
|---|---|---|---|
| POST | `/api/chat` | `{query, top_k, temperature}` | `{query, answer, sources, num_chunks_found, retrieval_time, generation_time, total_time}` |
| GET | `/health` | — | `{status, message, statistics}` |
| GET | `/api/stats` | — | Статистика: модели, кол-во документов |

---

## Evaluation

**Запуск:**
```bash
python scripts/run_evaluation.py --config configs/strategies/baseline.yaml --top-k 5
```

**Как работает:** 10 вопросов из `questions.py` прогоняются через полный RAG pipeline. Retrieval-метрики считаются по source-level: `expected_sources` (имя PDF) сравнивается с `file_name` из retrieved chunks через set-пересечение — **точное строковое совпадение**. Generation-метрики — LLM-as-Judge через тот же LM Studio (4 критерия, шкала 1–5).

**Результаты (2026-02-05):**

| Стратегия | chunk_size | P@5  | R@5  | MRR  | NDCG@5 | Correctness | Completeness |
|-----------|-----------|------|------|------|--------|-------------|--------------|
| baseline  | 512       | 0.68 | 1.00 | 0.81 | 0.84   | 4.9         | 4.4          |
| large     | 1024      | 0.64 | 1.00 | 0.93 | 0.90   | 5.0         | 4.8          |
| small     | 256       | 0.74 | 1.00 | 0.88 | 0.90   | 4.9         | 4.2          |

Relevance=5.0 и Coherence=5.0 во всех стратегиях.

---

## Файлы в data/ (имена для expected_sources)

```
klinicheskiy-polimorfizm-pri-narusheniyah-v-rabote-pischevaritelnoy-sistemy-i-metody-ih-korrektsii-u-detey-rannego-vozrasta-rodivshihsya-s-ekstremalno-nizkoy-i-ochen-nizkoy-massoy-tela-v-katamneze.pdf
mediko-sotsialnoe-issledovanie-patsientov-s-medikamentoznoy-toksikodermiey.pdf
metodika-primeneniya-propriotseptivnyh-korrektorov.pdf
opyt-organizatsii-obespecheniya-lekarstvennymi-sredstvami-i-meditsinskimi-izdeliyami-mezhdunarodnyh-sportivnyh-sorevnovaniy-na-primere-zimney-universiady-2019.pdf
otnoshenie-k-fenomenu-childfree-studentov-meditsinskogo-vuza.pdf
provedenie-sravnitelnoy-kliniko-farmakologicheskoy-otsenki-farmakoterapii-preparatami-botulinicheskogo-toksina-tipa-a-u-detey-s-detskim-tserebralnym-paralichom-s-vyrazhennym-sindromom-spastichnosti.pdf
sravnitelnyy-analiz-vliyaniya-psihologicheskih-harakteristik-kachestva-zhizni-i-sotsialno-ekonomicheskogo-statusa-na-uspevaemost-u-rossiyskih-i-kitayskih-studentov-meditsinskih-spetsialnostey-sopredelnyh-territoriy.pdf
sravnitelnyy-kontentnyy-analiz-normativnyh-dokumentov-reglamentiruyuschih-provedenie-rentgenovskoy-mammografii-v-rossiyskoy-federatsii.pdf
vliyanie-aminoguanidina-na-kataraktogenez-v-usloviyah-eksperimentalnogo-saharnogo-diabeta.pdf
vozmozhnosti-otsenki-riska-u-patsientov-s-ostrym-koronarnym-sindromom-starshe-75-let.pdf
```

Эти имена используются как `expected_sources` в `src/evaluation/questions.py` и как `file_name` в метаданных ChromaDB. Любое несовпадение символ-в-символ обнуляет retrieval-метрики для этого вопроса.

---

## Известные проблемы

**2026-02-04 — опечатка в questions.py (вопрос 7):**
`expected_sources` содержал `uspevaenost` вместо `uspevaemost`. Метрики показывали 0 несмотря на корректный поиск. Исправлено в `src/evaluation/questions.py:60`.

---

## Стратегии поиска (retrieval.strategy)

| Значение | Что делает | Когда использовать |
|----------|-----------|-------------------|
| `"semantic"` | Только векторный поиск (cosine sim) | По умолчанию |
| `"hybrid"` | Semantic + BM25 → RRF fusion | Лучший recall на точных терминах |

Переключение — одна строка в YAML, перезапуск не требует переиндексации:
```yaml
retrieval:
  strategy: "hybrid"  # или "semantic"
  rrf_k: 60           # константа RRF (только для hybrid)
  bm25_k1: 1.5        # BM25 k1 (только для hybrid)
  bm25_b: 0.75        # BM25 b  (только для hybrid)
```

BM25 индекс строится автоматически при старте из уже проиндексированных чанков ChromaDB.
Новые зависимости: `rank-bm25`, `pymorphy2`, `pymorphy2-dicts-ru`.

---

## Важные детали для модификаций

- **Русский язык** повсюду: промпты (`prompts.py`), PDF-текст, UI. Всё UTF-8.
- **LM Studio API** — OpenAI-compatible. Клиенты в `lm_studio_embedder.py` и `lm_studio_client.py`. Оба с retry×3 + exponential backoff.
- **ChromaDB коллекции** — одна на стратегию (`medical_docs_baseline`, `_large`, `_small`). Переиндексация требует `--reset`.
- **Strategy pattern**: чтобы добавить новый чанкер/эмбеддер/ретриевер — реализовать ABC из `core/`, добавить в соответствующий `factory.py`.
- **Evaluation ground truth** — файлы PDF из `data/`. При добавлении/переименовании файлов обновить `questions.py` синхронно.
- **Web startup**: компоненты инициализируются в lifespan `web/app.py`, не при импорте модулей.
- **ingest_documents.py args**: `--config`, `--data-dir`, `--reset`, `--batch-size` (default 32), `--log-level`.
- **run_evaluation.py args**: `--config`, `--top-k` (default 5).
