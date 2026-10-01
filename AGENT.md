# AGENT.md — compareRAG

Модульная система сравнения RAG-стратегий на трёх доменах (медицинские статьи, Гражданский кодекс РФ, документация PostgreSQL). 30 вопросов на домен, ground-truth ответы.

Эмбеддинги и индексация — локально через LM Studio. LLM генерация и LLM-as-Judge — через API (polza.ai по умолчанию).

---

## Предварительные условия

- **LM Studio** запущен на `http://127.0.0.1:1234` с загруженной моделью эмбеддингов. Поддерживаемые модели:
  - `text-embedding-bge-m3` (рекомендуется)
  - `text-embedding-multilingual-e5-large-instruct`
  - `text-embedding-qwen3-embedding-0.6b`
  - `text-embedding-nomic-embed-text-v1.5` (default в конфигах)
- **API ключ** для LLM генерации (polza.ai или совместимый OpenAI endpoint)
- **Python 3.10+**, виртуальное окружение: `venv\Scripts\activate`
- Опционально для reranking: интернет для первой загрузки моделей из HuggingFace в `~/.cache/huggingface/`

## Быстрый старт

```bash
venv\Scripts\activate

# Индексация (отдельно на каждый домен)
python scripts/ingest_documents.py --config configs/strategies/medical/sentence_baseline.yaml --data-dir data/Medical_articles --embedding-model text-embedding-bge-m3 --reset
python scripts/ingest_documents.py --config configs/strategies/civil_code/sentence_baseline.yaml --data-dir data/Civil_Code --embedding-model text-embedding-bge-m3 --reset
python scripts/ingest_documents.py --config configs/strategies/postgresql/baseline.yaml --data-dir data/PostrgreSQL --embedding-model text-embedding-multilingual-e5-large-instruct --reset

# Оценка
python scripts/run_evaluation.py --config configs/strategies/medical/sentence_baseline.yaml \
  --domain medical --top-k 10 \
  --embedding-model text-embedding-bge-m3 --retrieval semantic \
  --llm-url https://polza.ai/api --llm-model qwen/qwen3-14b --api-key YOUR_KEY

# Web UI (опционально, использует baseline.yaml)
cd web && python app.py   # → http://localhost:8000
```

---

## Стек

| Слой | Реализация |
|---|---|
| Чанкинг | Chonkie (`TokenChunker`, `SentenceChunker`, `RecursiveChunker`, `SemanticChunker`) + LangChain (`RecursiveCharacterTextSplitter`) |
| Эмбеддинги | LM Studio `/v1/embeddings` (OpenAI-compatible), батч 32, retry×3 |
| Векторная БД | ChromaDB (SQLite backend, cosine distance) |
| Lexical | BM25Okapi + pymorphy2 (русская лемматизация) |
| LLM | OpenAI-compatible `/v1/chat/completions` (polza.ai по умолчанию, qwen/qwen3-14b) |
| Reranker | sentence-transformers `CrossEncoder` (bge, jina) + transformers `AutoModelForCausalLM` (qwen3-reranker) |
| Web | FastAPI + Jinja2 + Vanilla JS |
| Config | Pydantic + YAML |
| Logging | Loguru: console + per-component files + experiment logs |

---

## Структура проекта

```
compareRAG/
├── configs/strategies/
│   ├── medical/                      # data/Medical_articles/ (10 PDF)
│   ├── civil_code/                   # data/Civil_Code/ (gkodeksrf.pdf, ~600 стр.)
│   └── postgresql/                   # data/PostrgreSQL/ (postgres_part1_2.pdf, ~600 стр.)
│       # Каждая папка содержит:
│       # baseline.yaml          — chonkie 512 (default)
│       # small_chunks.yaml      — chonkie 256
│       # large_chunks.yaml      — chonkie 1024
│       # sentence_baseline.yaml — sentence 512
│       # sentence_small.yaml    — sentence 256
│       # sentence_large.yaml    — sentence 1024
│       # recursive_baseline.yaml, recursive_small.yaml, recursive_large.yaml
│       # langchain_baseline.yaml
│       # medical/ дополнительно: semantic_chunker_{baseline,small,large}.yaml
├── data/
│   ├── Medical_articles/             # 10 русскоязычных мед. PDF
│   ├── Civil_Code/                   # gkodeksrf.pdf
│   └── PostrgreSQL/                  # postgres_part1_2.pdf
├── src/
│   ├── core/
│   │   ├── base_chunker.py           # Document, Chunk; BaseChunker ABC
│   │   ├── base_embedder.py          # BaseEmbedder ABC
│   │   ├── base_vector_store.py      # SearchResult; BaseVectorStore ABC
│   │   ├── base_retriever.py         # RetrievalResult; BaseRetriever ABC
│   │   └── config.py                 # RAGConfig (Pydantic), load_config(), embedding_model_id
│   ├── document_processing/
│   │   ├── pdf_loader.py             # PyPDF, page markers [Страница N]
│   │   ├── text_cleaner.py           # Unicode-нормализация
│   │   └── metadata_extractor.py
│   ├── chunking/
│   │   ├── chonkie_chunker.py        # TokenChunker (chonkie) — потокенное
│   │   ├── sentence_chunker.py       # Chonkie SentenceChunker — по предложениям, RU-разделители
│   │   ├── semantic_chunker.py       # Chonkie SemanticChunker — локальный potion-base-32M, threshold 0.5
│   │   ├── recursive_chunker.py      # Chonkie RecursiveChunker — иерархическое
│   │   ├── langchain_chunker.py      # RecursiveCharacterTextSplitter — fallback
│   │   └── factory.py                # ChunkingStrategyFactory.create(strategy, **kwargs)
│   ├── embeddings/
│   │   ├── lm_studio_embedder.py     # POST /v1/embeddings, batch=32, retry×3, EOS-token support
│   │   └── factory.py                # create_embedder_from_config(config)
│   ├── vector_stores/
│   │   ├── chroma_store.py           # ChromaDB persistence + metadata filtering
│   │   └── factory.py                # create_vector_store_from_config(config)
│   ├── retrieval/
│   │   ├── semantic_retriever.py     # Cosine sim search; принимает optional reranker
│   │   ├── bm25_retriever.py         # BM25Okapi + pymorphy2, индекс строится из ChromaDB
│   │   ├── hybrid_retriever.py       # Semantic + BM25 → RRF fusion (k=60); optional reranker
│   │   ├── reranker.py               # CrossEncoder (bge/jina) + Qwen3 causal LM yes/no scoring
│   │   └── factory.py                # create_retriever_from_config(config, embedder, store, reranker)
│   ├── llm/
│   │   ├── lm_studio_client.py       # POST /v1/chat/completions; поддерживает Bearer API key, remote URLs
│   │   └── prompts.py                # SYSTEM_PROMPT, SYNTHESIS_PROMPT, ANSWER_EVALUATION_PROMPT (RU)
│   ├── agents/
│   │   ├── base_agent.py             # AgentResult; BaseAgent ABC
│   │   ├── synthesis_agent.py        # chunks → LLM → текст
│   │   └── orchestrator.py           # SimpleRAGOrchestrator.query()
│   ├── evaluation/
│   │   ├── questions.py              # EVAL_QUESTIONS — медицинский домен (30 вопросов)
│   │   ├── questions_civil_code.py   # 30 вопросов по ГК РФ
│   │   ├── questions_postgresql.py   # 30 вопросов по PostgreSQL
│   │   ├── ground_truth.json         # 30 эталонных ответов (medical)
│   │   ├── ground_truth_civil_code.json
│   │   ├── ground_truth_postgresql.json
│   │   ├── retrieval_metrics.py      # MAP@k, Recall@k, MRR, NDCG@k (page-based + file-based fallback)
│   │   └── generation_metrics.py     # LLMJudge: 4 критерия 1–5 (Correctness, Faithfulness, Relevancy, Completeness)
│   └── utils/
│       ├── logging_config.py         # Loguru: console + rag_*.log + errors_*.log + metrics_*.json
│       └── exceptions.py
├── scripts/
│   ├── ingest_documents.py           # CLI: PDF → chunks → embeddings → ChromaDB
│   ├── run_evaluation.py             # CLI: 30 вопросов × full pipeline → метрики + experiment logs
│   ├── run_batch.py                  # Батч 54 комбинации (3 домена × 3 чанкинга × 3 эмбеддинга × 2 retrieval)
│   ├── run_topk_test.py              # Тест top_k=1,3,5,10 на 3 лучших конфигах = 12 eval
│   ├── run_chunksize_test.py         # Тест chunk_size=256,512,1024 × 3 чанкинга × 3 домена = 27 eval
│   ├── run_rerank_test.py            # Тест 3 reranker (bge/jina/qwen3) × 3 лучших конфига = 9 eval
│   ├── collect_results.py            # Агрегация experiments/ → results_<domain>.txt (с фильтрами)
│   ├── generate_civil_code_questions.py / generate_postgresql_questions.py  # Генератор GT
│   └── fix_civil_code_pages.py
├── web/
│   ├── app.py                        # FastAPI, lifespan init
│   ├── api/chat.py                   # POST /api/chat, GET /health, GET /api/stats
│   └── templates/static/             # UI
├── experiments/                      # Логи запусков (см. ниже)
├── chroma_db/                        # Персистент ChromaDB (не в git)
├── logs/                             # rag_*.log, errors_*.log, evaluation_*.log, metrics_*.json
├── batch_logs/                       # Логи каждой задачи batch-скриптов
├── reranking/, size/, topk/          # Агрегированные результаты экспериментов
├── *_progress.json                   # Прогресс batch-скриптов (batch, topk, chunksize, rerank)
└── requirements.txt
```

---

## Поток данных

```
Запрос пользователя
  → orchestrator.query()
    ├── retriever.retrieve(query, k=top_k)
    │     Semantic: embed(query) → ChromaStore.similarity_search()
    │     Hybrid:   Semantic(candidate_k) + BM25(candidate_k) → RRF fusion (k=60)
    │     Reranker (опционально): берёт top_n кандидатов → score(query, doc) → top_k
    │       — CrossEncoder для bge/jina-reranker (sentence-transformers)
    │       — Qwen3 causal LM (P("yes") из логитов) для qwen3-reranker
    └── SynthesisAgent → LMStudioClient (chat completions API) → ответ
  ← {answer, chunks, scores, retrieval_time, generation_time}
```

---

## Конфигурация (RAGConfig, Pydantic)

```yaml
lm_studio:
  url: "http://127.0.0.1:1234"            # для эмбеддингов
  llm_url: "https://polza.ai/api"          # для LLM (опционально, overrides url)
  embedder_url: "http://127.0.0.1:1234"    # для embedder (опционально)
  llm_model: "qwen/qwen3-14b"
  embedding_model: "text-embedding-nomic-embed-text-v1.5"
  temperature: 0.7
  max_tokens: 2000
  timeout: 300
  api_key: null                            # передаётся как Bearer для polza.ai
  add_eos_token: false                     # нужно для qwen3-embedding
  eos_token: "</s>"

chromadb:
  path: "./chroma_db"
  collection_name: "civil_code_sentence_512"   # суффикс эмбеддинга добавляется автоматически
  distance_function: "cosine"

chunking:
  strategy: "sentence"          # chonkie | sentence | semantic_chunker | recursive | langchain
  chunk_size: 512
  chunk_overlap: 0              # для sentence/semantic — 0; для chonkie/recursive — chunk_size/10

retrieval:
  strategy: "hybrid"            # semantic | hybrid
  top_k: 10
  score_threshold: 0.0
  rrf_k: 60                     # RRF константа (hybrid)
  bm25_k1: 1.5                  # BM25 (hybrid)
  bm25_b: 0.75

data_dir: "./data/Civil_Code"
results_dir: "./results/civil_code_sentence_baseline"
```

Каждая комбинация (strategy, chunk_size, embedding) → отдельная коллекция ChromaDB вида `{base_name}_{embedding_safe_id}` (например `civil_code_sentence_512_bge`). Переключение чанкинга/эмбеддинга требует переиндексации с `--reset`.

---

## Стратегии чанкинга

| Значение | Класс | Подход |
|---|---|---|
| `chonkie` | `ChonkieChunker` (Chonkie `TokenChunker`) | По токенам, фиксированный размер. Основной. |
| `sentence` | `SentenceChunkerWrapper` (Chonkie `SentenceChunker`) | По предложениям, накапливает до `chunk_size` токенов. Русские разделители. Использует `chunk.start_index` для определения страницы. |
| `semantic_chunker` | `SemanticChunkerWrapper` (Chonkie `SemanticChunker`) | По семантическим границам через локальный `minishlab/potion-base-32M` (не LM Studio), threshold=0.5. |
| `recursive` | `RecursiveChunker` (Chonkie) | Иерархическое: пробует разделители от крупных к мелким. |
| `langchain` | `LangChainChunker` (`RecursiveCharacterTextSplitter`) | Fallback. По символам, не токенам. |

---

## Стратегии поиска

| Значение | Что делает |
|---|---|
| `semantic` | Только векторный поиск (cosine sim). |
| `hybrid` | Semantic (candidate_k = top_k×3) + BM25 (candidate_k) → RRF fusion. BM25-индекс строится при старте из ChromaDB. |

---

## Reranking (опционально)

После retrieval можно перейти через reranker, чтобы переупорядочить top_n кандидатов и оставить top_k.

**Поддерживаемые модели** (HuggingFace ID, скачиваются автоматически):

| HF ID | Backend | Время на пару (CPU) |
|---|---|---|
| `BAAI/bge-reranker-v2-m3` | `CrossEncoder` (sentence-transformers) | ~0.3-0.5 с |
| `jinaai/jina-reranker-v2-base-multilingual` | `CrossEncoder` (sentence-transformers, trust_remote_code) | ~0.5-1 с |
| `Qwen/Qwen3-Reranker-0.6B` | `AutoModelForCausalLM` (transformers), P("yes") из логитов | ~3-5 с |

**Использование:**
```bash
python scripts/run_evaluation.py --config ... --reranker-model BAAI/bge-reranker-v2-m3 --rerank-top-n 20
```

Можно использовать имена LM Studio (`text-embedding-bge-reranker-v2-m3`) — они автоматически резолвятся в HF ID.

Логи переранжированных запусков пишутся в подпапку `experiments/{domain}/{retrieval}/{embedding}/{chunking}/{chunk_size}/rerank_{short}/`.

---

## Evaluation

**Вопросы:** 30 на домен в `src/evaluation/questions{,_civil_code,_postgresql}.py` с полями:
- `question`, `expected_sources` (имена PDF), `expected_pages` (`{file: [pages]}`)
- `reference_answer` — подгружается из `ground_truth*.json` при запуске

**Retrieval-метрики** (`compute_all` в `retrieval_metrics.py`): MAP@k, Recall@k, MRR, NDCG@k.
- **Page-based** (основной): сравнение `(file_name, page_number)` retrieved chunks с `expected_pages`.
- **File-based fallback** (если у вопроса нет `expected_pages`): только по именам PDF.

**Generation-метрики** (`LLMJudge` в `generation_metrics.py`): LLM-as-Judge, 4 критерия 1-5:
- Answer Correctness (соответствие reference_answer)
- Faithfulness (опора только на контекст)
- Answer Relevancy (релевантность вопросу)
- Completeness (полнота)

**Параллельность:** `--workers N` (default 4, max 8). Каждый воркер обрабатывает свой вопрос полностью (retrieval + generation + judge).

---

## Experiment logging

Каждый запуск `run_evaluation.py` создаёт **два лога**:

```
experiments/{domain}/{retrieval}/{embedding_folder}/{chunking}/{chunk_size}/[rerank_{short}/]{full|results}/{prefix}_{type}_{date}_{nn}.log
```

- **full**: вся отладочная информация (header + per-question chunks/scores/answer + summary).
- **results**: header + summary (без per-question деталей).

**Embedding folder/abbrev mapping:**
| LM Studio имя | folder | abbrev |
|---|---|---|
| `text-embedding-bge-m3` | `bge-m3` | `bge` |
| `text-embedding-qwen3-embedding-0.6b` | `qwen3-06b` | `qwen3` |
| `text-embedding-nomic-embed-text-v1.5` | `nomic` | `nomic` |
| `text-embedding-multilingual-e5-large-instruct` | `e5-large` | `e5` |

**Loguru binding:**
- `logger.bind(eval_full=True)` → только full
- `logger.bind(eval_full=True, eval_results=True)` → оба
- хелперы: `eval_log(msg)` (full), `eval_log(msg, results=True)` (оба)

---

## Batch-скрипты

Все батч-скрипты используют subprocess + `*_progress.json` для возобновления после сбоя.

| Скрипт | Что | Прогресс-файл |
|---|---|---|
| `run_batch.py` | 54 комбинации: 3 домена × 3 чанкинга × 3 эмбеддинга × 2 retrieval (ingestion + eval, обе фазы) | `batch_progress.json` |
| `run_topk_test.py` | 3 лучших конфига × top_k=[1,3,5,10] = 12 eval | `topk_progress.json` |
| `run_chunksize_test.py` | 3 домена × 3 чанкинга × [256,512,1024] = 27 ingest + 27 eval (BGE-M3 + hybrid зафиксированы) | `chunksize_progress.json` |
| `run_rerank_test.py` | 3 лучших конфига × 3 reranker = 9 eval (поддерживает `--stream` для real-time вывода) | `rerank_progress.json` |

**Общие флаги:** `--api-key` (обязателен), `--dry-run`, `--reset-progress`, `--only-domain`, `--workers`, `--timeout`. Логи каждой задачи → `batch_logs/`.

**Лучшие конфигурации** (в `run_topk_test.py` и `run_rerank_test.py`):
| Домен | Чанкинг | Retrieval | Эмбеддинг |
|---|---|---|---|
| civil_code | sentence | hybrid | bge-m3 |
| medical | sentence | semantic | bge-m3 |
| postgresql | chonkie | hybrid | e5-large |

---

## Сбор результатов

```bash
# Все эксперименты:
python scripts/collect_results.py

# Только новые (rerank):
python scripts/collect_results.py --rerank-only --output-dir reranking

# Фильтр по chunksize-тесту:
python scripts/collect_results.py --retrieval hybrid --embedding bge-m3 \
  --chunking chonkie sentence recursive --chunk-size 256 512 1024 \
  --output-dir size
```

Группирует логи по результирующей папке, берёт самый свежий `.log` из каждой, пишет в `results_<domain>.txt`.

**Флаги:** `--retrieval`, `--embedding`, `--chunking`, `--chunk-size`, `--rerank-only`, `--no-rerank`, `--rerank <folder...>`, `--output-dir`.

---

## CLI: `scripts/run_evaluation.py`

| Флаг | Default | Что |
|---|---|---|
| `--config` | `configs/strategies/baseline.yaml` | YAML конфиг |
| `--domain` | `medical` | `medical` / `civil_code` / `postgresql` |
| `--top-k` | `10` | Финальный top-k для LLM |
| `--embedding-model` | (из конфига) | Переопределяет; автоматически суффиксирует collection_name |
| `--retrieval` | (из конфига) | `semantic` / `hybrid` |
| `--reranker-model` | — | HF ID или LM Studio имя reranker'а |
| `--rerank-top-n` | `20` | Сколько кандидатов извлекать ДО переранжирования |
| `--llm-url` | — | Например `https://polza.ai/api` |
| `--embedder-url` | — | По умолчанию из конфига (LM Studio) |
| `--llm-model` | — | Например `qwen/qwen3-14b` |
| `--api-key` | — | Bearer для LLM API |
| `--add-eos-token` / `--eos-token` | False / `</s>` | EOS для qwen3-embedding |
| `--workers` | `4` | Параллельные потоки |
| `--timeout` | (из конфига) | LLM timeout (сек) |
| `--verbose` | False | Подробный вывод в консоль |

## CLI: `scripts/ingest_documents.py`

| Флаг | Default | Что |
|---|---|---|
| `--config` | — | YAML конфиг |
| `--data-dir` | (из конфига) | Папка с PDF |
| `--embedding-model` | — | Переопределяет |
| `--reset` | False | Очистить коллекцию перед индексацией |
| `--batch-size` | `32` | Размер батча эмбеддинга |
| `--log-level` | `INFO` |  |

---

## Известные проблемы и нюансы

- **LM Studio не имеет `/v1/rerank`** — reranker-модели в LM Studio через `/v1/embeddings` возвращают обычные эмбеддинги (бесполезно для cross-encoder). Reranking реализован через sentence-transformers напрямую (HuggingFace download).
- **Jina-reranker несовместим с transformers ≥ 5.x**: импорт `create_position_ids_from_input_ids` падает. В `src/retrieval/reranker.py` monkey-patch возвращает функцию обратно перед загрузкой модели.
- **Loguru UnicodeEncodeError на Windows cp1251**: при `--stream` (run_rerank_test.py) логи с эмодзи (`✅`) ломают консольный вывод. На прогресс не влияет; файловые логи пишутся правильно (UTF-8). Воркараунд: `set PYTHONIOENCODING=utf-8 && chcp 65001`.
- **Reranker на CPU очень медленный**: bge ~7 мин на конфиг, jina ~12 мин, qwen3 ~40 мин (это 30 вопросов × 20 пар × forward LM).
- **PDF page mapping** для civil_code исправлен в `scripts/fix_civil_code_pages.py` (была расхождение между PyPDF-номерами и реальными страницами кодекса).
- **EOS token для qwen3-embedding**: модель ожидает EOS в конце текста. Включается через `--add-eos-token`.
- **Опечатка questions.py вопрос 7** (исправлено февраль 2026): было `uspevaenost` вместо `uspevaemost` в expected_sources.

---

## Важные детали для модификаций

- **Strategy pattern везде**: новый чанкер → реализовать `BaseChunker`, добавить в `src/chunking/factory.py`. Аналогично embedder/vector_store/retriever.
- **Новый reranker**: добавить класс в `src/retrieval/reranker.py`, обновить `create_reranker()` factory. Класс должен иметь `score(query, documents) → List[float]`.
- **Новый домен**: создать `src/evaluation/questions_<domain>.py`, `ground_truth_<domain>.json`, добавить в `DOMAIN_QUESTIONS` в `run_evaluation.py` и в `_GT_FILES`. Создать `configs/strategies/<domain>/*.yaml`.
- **PDF page metadata**: метаданные чанков должны содержать `page_number` (int) и `file_name` (str basename). Это критично для page-based retrieval-метрик.
- **Коллекции изолированы по эмбеддингу**: `{config.chromadb.collection_name}_{safe_model_id}`. Менять эмбеддинг без переиндексации нельзя.
- **Web UI** использует только `baseline.yaml` без CLI-оверрайдов — для экспериментов запускать через `run_evaluation.py`.
- **LLM URL routing**: `llm_url` имеет приоритет над `url` для LLM. Аналогично `embedder_url` для эмбеддингов. Если оба не заданы — используется `url`.
