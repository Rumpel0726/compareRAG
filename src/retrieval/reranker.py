# -*- coding: utf-8 -*-
"""
Reranker - переранжирование результатов поиска через cross-encoder модели.

Поддерживает три модели:
- BAAI/bge-reranker-v2-m3 (стандартный cross-encoder)
- jinaai/jina-reranker-v2-base-multilingual (cross-encoder, trust_remote_code)
- Qwen/Qwen3-Reranker-0.6B (causal LM с yes/no scoring)
"""

from typing import List, Optional, Tuple
from loguru import logger

from ..core.base_chunker import Chunk


# Маппинг короткого имени LM Studio → HuggingFace ID
LMSTUDIO_TO_HF = {
    "text-embedding-bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "text-embedding-jina-reranker-v2-base-multilingual": "jinaai/jina-reranker-v2-base-multilingual",
    "qwen3-reranker-0.6b": "Qwen/Qwen3-Reranker-0.6B",
}

# Короткие имена для путей в experiments/
RERANKER_SHORT = {
    "BAAI/bge-reranker-v2-m3": "bge-rerank-v2",
    "jinaai/jina-reranker-v2-base-multilingual": "jina-rerank-v2",
    "Qwen/Qwen3-Reranker-0.6B": "qwen3-rerank-06b",
}


def resolve_reranker_id(model: str) -> str:
    """Резолвит имя модели: принимает либо LM Studio имя, либо HF ID."""
    return LMSTUDIO_TO_HF.get(model, model)


def reranker_short_name(model: str) -> str:
    """Возвращает короткое имя для использования в путях."""
    hf_id = resolve_reranker_id(model)
    if hf_id in RERANKER_SHORT:
        return RERANKER_SHORT[hf_id]
    # Fallback: нижний регистр, замена / на _
    return hf_id.lower().replace("/", "_").replace("-", "_")


def _patch_transformers_for_jina():
    """
    Jina-reranker-v2 использует кастомный modeling_xlm_roberta.py, который
    импортирует `create_position_ids_from_input_ids` — в transformers 5.x этой
    функции больше нет. Возвращаем её обратно как монки-патч.
    """
    try:
        from transformers.models.xlm_roberta import modeling_xlm_roberta as mxr
        if hasattr(mxr, "create_position_ids_from_input_ids"):
            return
        import torch

        def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
            mask = input_ids.ne(padding_idx).int()
            incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
            return incremental_indices.long() + padding_idx

        mxr.create_position_ids_from_input_ids = create_position_ids_from_input_ids
        logger.info("Monkey-patch для jina: добавлена create_position_ids_from_input_ids")
    except Exception as e:
        logger.warning(f"Не удалось применить monkey-patch для jina: {e}")


class CrossEncoderReranker:
    """
    Reranker на основе sentence-transformers CrossEncoder.
    Подходит для bge-reranker и jina-reranker.
    """

    def __init__(self, model_id: str):
        # Для jina-reranker нужен monkey-patch transformers
        if "jina" in model_id.lower():
            _patch_transformers_for_jina()

        from sentence_transformers import CrossEncoder

        logger.info(f"Загрузка CrossEncoder: {model_id}")
        # trust_remote_code нужен для jina-reranker-v2
        self.model_id = model_id
        self.model = CrossEncoder(model_id, trust_remote_code=True)
        logger.info(f"CrossEncoder загружен: {model_id}")

    def score(self, query: str, documents: List[str]) -> List[float]:
        """Возвращает relevance scores для пар (query, doc)."""
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]


class Qwen3CausalReranker:
    """
    Reranker для Qwen3-Reranker — генеративная LM, которая отвечает yes/no.
    Score = P(yes) из логитов первого сгенерированного токена.
    """

    PROMPT_TEMPLATE = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".<|im_end|>\n'
        "<|im_start|>user\n"
        "<Instruct>: Given a search query, retrieve relevant passages that answer the query\n"
        "<Query>: {query}\n"
        "<Document>: {document}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info(f"Загрузка Qwen3-Reranker: {model_id}")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).eval()
        self.torch = torch

        # ID токенов "yes" / "no" — определяются с учётом особенностей токенизатора
        self.yes_id = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.no_id = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        logger.info(
            f"Qwen3-Reranker загружен: yes_id={self.yes_id}, no_id={self.no_id}"
        )

    def score(self, query: str, documents: List[str]) -> List[float]:
        """Возвращает P(yes) для каждой пары (query, doc)."""
        if not documents:
            return []

        scores = []
        with self.torch.no_grad():
            for doc in documents:
                prompt = self.PROMPT_TEMPLATE.format(query=query, document=doc)
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
                outputs = self.model(**inputs)
                last_logits = outputs.logits[0, -1, :]
                yes_logit = last_logits[self.yes_id]
                no_logit = last_logits[self.no_id]
                # Softmax по двум токенам
                stacked = self.torch.stack([no_logit, yes_logit])
                probs = self.torch.softmax(stacked, dim=0)
                scores.append(float(probs[1]))
        return scores


def create_reranker(model: str):
    """Factory: создаёт нужный класс по имени модели."""
    hf_id = resolve_reranker_id(model)
    if "qwen3-reranker" in hf_id.lower() or "Qwen3-Reranker" in hf_id:
        return Qwen3CausalReranker(hf_id)
    return CrossEncoderReranker(hf_id)


class Reranker:
    """
    Универсальная обёртка с методом rerank_chunks().

    Принимает чанки и их исходные scores, возвращает top-k чанков
    отсортированных по rerank score (исходные scores заменяются).
    """

    def __init__(self, model: str, top_n: int = 20):
        """
        Args:
            model: имя модели (LM Studio имя или HF ID)
            top_n: сколько кандидатов брать из retriever ДО переранжирования
        """
        self.model_name = model
        self.top_n = top_n
        self._backend = create_reranker(model)
        logger.info(
            f"Reranker готов: model={model}, top_n={top_n}, "
            f"backend={type(self._backend).__name__}"
        )

    def rerank_chunks(
        self,
        query: str,
        chunks: List[Chunk],
        scores: List[float],
        k: int,
    ) -> Tuple[List[Chunk], List[float]]:
        """
        Переранжирует чанки. Возвращает top-k чанков с новыми scores.
        """
        if not chunks:
            return chunks, scores
        documents = [c.text for c in chunks]
        rerank_scores = self._backend.score(query, documents)

        # Сортируем по убыванию score
        indexed = sorted(
            enumerate(rerank_scores), key=lambda x: x[1], reverse=True
        )[:k]
        reranked_chunks = [chunks[i] for i, _ in indexed]
        reranked_scores = [s for _, s in indexed]
        return reranked_chunks, reranked_scores
