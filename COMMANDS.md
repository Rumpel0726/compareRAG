# Команды запуска compareRAG

> Все команды выполняются из корня проекта (`compareRAG/`) с активированным venv.

```bash
venv\Scripts\activate
```

---

## Индексация документов

### Базовый запуск (модель из конфига)
```bash
python scripts/ingest_documents.py \
    --config configs/strategies/baseline.yaml
```

### С указанием модели эмбеддингов (используется на практике)
```bash
python scripts/ingest_documents.py \
    --config configs/strategies/baseline.yaml \
    --embedding-model text-embedding-bge-m3 \
    --reset
```

> `--reset` очищает коллекцию перед индексацией (обязателен при смене модели или переиндексации).
> При `--embedding-model` имя коллекции автоматически получает суффикс: `medical_docs_baseline_bge`.

### Все три стратегии
```bash
python scripts/ingest_documents.py \
    --config configs/strategies/baseline.yaml \
    --embedding-model text-embedding-bge-m3 \
    --reset

python scripts/ingest_documents.py \
    --config configs/strategies/large_chunks.yaml \
    --embedding-model text-embedding-bge-m3 \
    --reset

python scripts/ingest_documents.py \
    --config configs/strategies/small_chunks.yaml \
    --embedding-model text-embedding-bge-m3 \
    --reset
```

### С другой моделью эмбеддингов
```bash
# nomic (из конфига по умолчанию)
python scripts/ingest_documents.py \
    --config configs/strategies/baseline.yaml \
    --reset

# qwen3-embedding (требует EOS-токен)
python scripts/ingest_documents.py \
    --config configs/strategies/baseline.yaml \
    --embedding-model text-embedding-qwen3-embedding-0.6b \
    --add-eos-token \
    --eos-token "<|endoftext|>" \
    --reset
```

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

### Базовый запуск
```bash
python scripts/run_evaluation.py \
    --config configs/strategies/baseline.yaml \
    --embedding-model text-embedding-bge-m3
```

### С указанием top-k
```bash
python scripts/run_evaluation.py \
    --config configs/strategies/baseline.yaml \
    --embedding-model text-embedding-bge-m3 \
    --top-k 10
```

### С qwen3-embedding (EOS-токен обязателен — как при индексации)
```bash
python scripts/run_evaluation.py \
    --config configs/strategies/baseline.yaml \
    --embedding-model text-embedding-qwen3-embedding-0.6b \
    --add-eos-token \
    --eos-token "<|endoftext|>" \
    --top-k 10
```

### Сравнение всех трёх стратегий
```bash
python scripts/run_evaluation.py \
    --config configs/strategies/baseline.yaml \
    --embedding-model text-embedding-bge-m3 \
    --top-k 10

python scripts/run_evaluation.py \
    --config configs/strategies/large_chunks.yaml \
    --embedding-model text-embedding-bge-m3 \
    --top-k 10

python scripts/run_evaluation.py \
    --config configs/strategies/small_chunks.yaml \
    --embedding-model text-embedding-bge-m3 \
    --top-k 10
```

> Результаты записываются в `logs/evaluation_<YYYY-MM-DD>.log`.

### Все аргументы run_evaluation.py
| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `--config` | `configs/strategies/baseline.yaml` | Путь к YAML конфигурации |
| `--top-k` | `10` | Количество чанков для поиска |
| `--embedding-model` | из конфига | Переопределить модель эмбеддингов |
| `--add-eos-token` | `False` | Добавить EOS-токен |
| `--eos-token` | `</s>` | EOS-токен |

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
#    с моделями: text-embedding-bge-m3 и qwen/qwen3-14b

# 3. Индексировать документы (один раз или при смене модели/чанкинга)
python scripts/ingest_documents.py \
    --config configs/strategies/baseline.yaml \
    --embedding-model text-embedding-bge-m3 \
    --reset

# 4. Запустить веб-интерфейс
cd web && python app.py

# 5. Оценить качество (в другом терминале)
python scripts/run_evaluation.py \
    --config configs/strategies/baseline.yaml \
    --embedding-model text-embedding-bge-m3 \
    --top-k 10
```

---

## Имена коллекций ChromaDB

При использовании `--embedding-model` суффикс добавляется автоматически:

| Конфиг + модель | Коллекция |
|----------------|-----------|
| `baseline` + `bge-m3` | `medical_docs_baseline_bge` |
| `large_chunks` + `bge-m3` | `medical_docs_large_bge` |
| `small_chunks` + `bge-m3` | `medical_docs_small_bge` |
| `baseline` + `nomic` (из конфига) | `medical_docs_baseline_nomic` |
| `baseline` + `qwen3-embedding-0.6b` | `medical_docs_baseline_qwen3` |
