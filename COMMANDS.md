# Команды запуска compareRAG

> Все команды выполняются из корня проекта (`compareRAG/`) с активированным venv.

```bash
venv\Scripts\activate
```

---

## Индексация документов

### Медицинские статьи
```bash
python scripts/ingest_documents.py --config configs/strategies/medical/baseline.yaml --data-dir data/Medical_articles --reset

python scripts/ingest_documents.py --config configs/strategies/medical/large_chunks.yaml --data-dir data/Medical_articles --reset

python scripts/ingest_documents.py --config configs/strategies/medical/small_chunks.yaml --data-dir data/Medical_articles --reset
```

### Гражданский кодекс РФ
```bash
python scripts/ingest_documents.py --config configs/strategies/civil_code/baseline.yaml --data-dir data/Civil_Code --reset
```

### Документация PostgreSQL
```bash
python scripts/ingest_documents.py --config configs/strategies/postgresql/baseline.yaml --data-dir data/PostrgreSQL --reset
```

### С указанием модели эмбеддингов
```bash
# bge-m3
python scripts/ingest_documents.py --config configs/strategies/medical/baseline.yaml --data-dir data/Medical_articles --embedding-model text-embedding-bge-m3 --reset

# qwen3-embedding (требует EOS-токен)
python scripts/ingest_documents.py --config configs/strategies/medical/baseline.yaml --data-dir data/Medical_articles --embedding-model text-embedding-qwen3-embedding-0.6b --add-eos-token --eos-token "<|endoftext|>" --reset
```

> `--reset` очищает коллекцию перед индексацией (обязателен при смене модели или переиндексации).
> При `--embedding-model` имя коллекции автоматически получает суффикс: `medical_docs_baseline_bge`.

### Все аргументы ingest_documents.py
| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `--config` | `configs/default.yaml` | Путь к YAML конфигурации |
| `--data-dir` | `./data` | Папка с PDF файлами |
| `--reset` | `False` | Очистить коллекцию перед индексацией |
| `--batch-size` | `32` | Размер батча для эмбеддингов |
| `--log-level` | `INFO` | Уровень логов: DEBUG / INFO / WARNING / ERROR |
| `--embedding-model` | из конфига | Переопределить модель эмбеддингов |
| `--add-eos-token` | `False` | Добавить EOS-токен (для qwen3-embedding) |
| `--eos-token` | `</s>` | EOS-токен (для qwen3-embedding: `<\|endoftext\|>`) |

---

## Оценка (Evaluation)

> **Важно:** флаги `--embedding-model`, `--add-eos-token`, `--eos-token` должны **точно совпадать** с теми,
> что использовались при индексации — иначе evaluation будет искать в другой коллекции или
> генерировать несовместимые векторы.

### Медицинские статьи
```bash
python scripts/run_evaluation.py --config configs/strategies/medical/baseline.yaml --domain medical --top-k 10

python scripts/run_evaluation.py --config configs/strategies/medical/large_chunks.yaml --domain medical --top-k 10

python scripts/run_evaluation.py --config configs/strategies/medical/small_chunks.yaml --domain medical --top-k 10
```

### Гражданский кодекс РФ
```bash
python scripts/run_evaluation.py --config configs/strategies/civil_code/baseline.yaml --domain civil_code --top-k 10
```

### Документация PostgreSQL
```bash
python scripts/run_evaluation.py --config configs/strategies/postgresql/baseline.yaml --domain postgresql --top-k 10
```

### С указанием модели эмбеддингов
```bash
python scripts/run_evaluation.py --config configs/strategies/medical/baseline.yaml --domain medical --embedding-model text-embedding-bge-m3 --top-k 10

python scripts/run_evaluation.py --config configs/strategies/medical/baseline.yaml --domain medical --embedding-model text-embedding-qwen3-embedding-0.6b --add-eos-token --eos-token "<|endoftext|>" --top-k 10
```

> Результаты записываются в `experiments/{retrieval}/{emb_folder}/{chunk_strategy}/{chunk_size}/`.

### С удалённым LLM API (Polza AI) + локальными эмбеддингами
```bash
# Эмбеддинги — локальный LM Studio (bge-m3), LLM — Polza AI (qwen3-14b)
python scripts/run_evaluation.py --config configs/strategies/postgresql/baseline.yaml --domain postgresql --top-k 10 --embedding-model text-embedding-bge-m3 --embedder-url http://127.0.0.1:1234 --llm-url https://polza.ai/api --llm-model qwen/qwen3-14b --api-key <ключ> --timeout 600

# То же для medical
python scripts/run_evaluation.py --config configs/strategies/medical/baseline.yaml --domain medical --top-k 10 --embedding-model text-embedding-bge-m3 --embedder-url http://127.0.0.1:1234 --llm-url https://polza.ai/api --llm-model qwen/qwen3-14b --api-key <ключ> --timeout 600
```

> `--embedder-url` указывает куда идут эмбеддинги (локально), `--llm-url` — куда генерация (remote).
> `--api-key` передаётся в заголовок `Authorization: Bearer <ключ>`.

### Все аргументы run_evaluation.py
| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `--config` | `configs/strategies/baseline.yaml` | Путь к YAML конфигурации |
| `--domain` | `medical` | Домен вопросов: `medical` / `civil_code` / `postgresql` |
| `--top-k` | `10` | Количество чанков для поиска |
| `--embedding-model` | из конфига | Переопределить модель эмбеддингов |
| `--add-eos-token` | `False` | Добавить EOS-токен |
| `--eos-token` | `</s>` | EOS-токен |
| `--verbose` | `False` | Подробный вывод чанков и ответов в консоль |
| `--url` | из конфига | Общий URL для LLM и эмбеддингов |
| `--llm-url` | из конфига | URL для LLM генерации (переопределяет `--url`) |
| `--embedder-url` | из конфига | URL для эмбеддингов (переопределяет `--url`) |
| `--llm-model` | из конфига | Модель генерации (напр. `qwen/qwen3-14b`) |
| `--api-key` | из конфига | API-ключ для LLM (Bearer token) |
| `--workers` | `4` | Кол-во параллельных потоков (1 = последовательно) |
| `--timeout` | `120` | Таймаут LLM-запросов в секундах |

---

## Генерация вопросов и ground truth

### Гражданский кодекс РФ
```bash
python scripts/generate_civil_code_questions.py --api-key <ключ>
python scripts/generate_civil_code_questions.py  # ключ из POLZA_AI_API_KEY
python scripts/generate_civil_code_questions.py --resume  # продолжить с checkpoint
```

### Документация PostgreSQL
```bash
python scripts/generate_postgresql_questions.py --api-key <ключ>
python scripts/generate_postgresql_questions.py  # ключ из POLZA_AI_API_KEY
python scripts/generate_postgresql_questions.py --resume  # продолжить с checkpoint
```

> Скрипты генерируют 20 лёгких (1 страница) + 10 средних (3 страницы) вопросов.
> Выход: `src/evaluation/questions_*.py` + `src/evaluation/ground_truth_*.json`.
> API: Polza AI (`qwen/qwen3-235b-a22b`), ключ через `--api-key` или `POLZA_AI_API_KEY`.

---

## Веб-приложение

### Запуск с семантическим поиском (по умолчанию)
```bash
cd web && python app.py
```
Открыть: http://localhost:8000

### Переключить на гибридный поиск (Semantic + BM25)
1. В `configs/default.yaml` изменить:
   ```yaml
   retrieval:
     strategy: "hybrid"   # было "semantic"
   ```
2. Запустить как обычно:
   ```bash
   cd web && python app.py
   ```
   В логах при старте появится:
   ```
   BM25Retriever инициализирован: 1234 документов проиндексировано
   HybridRetriever инициализирован: top_k=10, rrf_k=60
   ```

> BM25 индекс строится из уже загруженных в ChromaDB чанков — переиндексация не нужна.

---

## Типичный рабочий цикл

```bash
# 1. Активировать окружение
venv\Scripts\activate

# 2. Убедиться что LM Studio запущен на http://127.0.0.1:1234
#    с моделями: text-embedding-nomic-embed-text-v1.5 и qwen/qwen3-14b

# 3. Индексировать документы (один раз или при смене модели/чанкинга)
python scripts/ingest_documents.py --config configs/strategies/medical/baseline.yaml --data-dir data/Medical_articles --reset

# 4. Запустить веб-интерфейс
cd web && python app.py

# 5. Оценить качество (в другом терминале)
python scripts/run_evaluation.py --config configs/strategies/medical/baseline.yaml --domain medical --top-k 10
```

---

## Имена коллекций ChromaDB

При использовании `--embedding-model` суффикс добавляется автоматически:

| Домен | Конфиг + модель | Коллекция |
|-------|----------------|-----------|
| medical | `medical/baseline` + `nomic` (из конфига) | `medical_docs_baseline_nomic` |
| medical | `medical/baseline` + `bge-m3` | `medical_docs_baseline_bge` |
| medical | `medical/baseline` + `qwen3-embedding-0.6b` | `medical_docs_baseline_qwen3` |
| medical | `medical/large_chunks` + `bge-m3` | `medical_docs_large_bge` |
| medical | `medical/small_chunks` + `bge-m3` | `medical_docs_small_bge` |
| civil_code | `civil_code/baseline` (nomic из конфига) | `civil_code_baseline_nomic` |
| postgresql | `postgresql/baseline` (nomic из конфига) | `postgresql_baseline_nomic` |
