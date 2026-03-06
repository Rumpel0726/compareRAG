# -*- coding: utf-8 -*-
"""
Метрики этапа Retrieval: MAP, R@k, MRR, NDCG@k.

Все функции работают с page-based бинарной релевантностью.
Проверяется соответствие (file_name, page_number) из retrieved чанков
с ground truth expected_pages.
"""

import math
from typing import List, Dict, Any


def average_precision(
    retrieved_chunks: List[Dict[str, Any]],
    expected_pages: Dict[str, List[int]],
    k: int
) -> float:
    """
    Average Precision (AP) — составляющая MAP (page-level).

    AP = (1/R) × Σ_{i=1}^{k} P(i) × rel(i)

    R    = суммарное кол-во релевантных страниц в expected_pages.
    P(i) = накопленная точность на позиции i.
    rel(i) = 1 если чанк на позиции i релевантен, иначе 0.

    Args:
        retrieved_chunks: Список словарей чанков с metadata
        expected_pages: Словарь {file_name: [page_numbers]}
        k: количество top-k позиций

    Returns:
        AP score для одного запроса
    """
    total_relevant = sum(len(pages) for pages in expected_pages.values())
    if total_relevant == 0 or k == 0:
        return 0.0

    seen_pages: set = set()
    relevant_so_far = 0
    ap_sum = 0.0

    for i, chunk in enumerate(retrieved_chunks[:k], start=1):
        file_name = chunk.get("file_name") or chunk.get("metadata", {}).get("file_name")
        page_number = chunk.get("metadata", {}).get("page_number")

        is_relevant = 0
        if file_name and page_number:
            page_key = (file_name, page_number)
            if page_key not in seen_pages:
                seen_pages.add(page_key)
                expected = expected_pages.get(file_name, [])
                if page_number in expected:
                    is_relevant = 1

        if is_relevant:
            relevant_so_far += 1
            ap_sum += relevant_so_far / i

    return ap_sum / total_relevant


def recall_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    expected_pages: Dict[str, List[int]]
) -> float:
    """
    Recall@k (page-level) — доля найденных ожидаемых страниц.

    Args:
        retrieved_chunks: Список словарей чанков
        expected_pages: Словарь {file_name: [page_numbers]}

    Returns:
        Recall score
    """
    # Подсчет общего количества ожидаемых страниц
    total_expected = sum(len(pages) for pages in expected_pages.values())
    if total_expected == 0:
        return 0.0

    # Множество найденных (file_name, page_number)
    found_pages = set()
    for chunk in retrieved_chunks:
        file_name = chunk.get("file_name") or chunk.get("metadata", {}).get("file_name")
        page_number = chunk.get("metadata", {}).get("page_number")

        if file_name and page_number:
            expected = expected_pages.get(file_name, [])
            if page_number in expected:
                found_pages.add((file_name, page_number))

    return len(found_pages) / total_expected


def mean_reciprocal_rank(
    retrieved_chunks: List[Dict[str, Any]],
    expected_pages: Dict[str, List[int]]
) -> float:
    """
    MRR — 1 / rank первого релевантного чанка (page-level).

    Args:
        retrieved_chunks: Список словарей чанков в порядке ранжирования
        expected_pages: Словарь {file_name: [page_numbers]}

    Returns:
        MRR score
    """
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        file_name = chunk.get("file_name") or chunk.get("metadata", {}).get("file_name")
        page_number = chunk.get("metadata", {}).get("page_number")

        if file_name and page_number:
            expected = expected_pages.get(file_name, [])
            if page_number in expected:
                return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    expected_pages: Dict[str, List[int]],
    k: int
) -> float:
    """
    NDCG@k с binary relevance (page-level).

    DCG@k  = Σ rel_i / log2(i + 2),  i = 0..k-1
    IDCG@k = DCG при идеальном ранжировании (все relevant сначала)
    NDCG   = DCG / IDCG.  Если IDCG == 0 → NDCG = 0.

    Args:
        retrieved_chunks: Список словарей чанков в порядке ранжирования
        expected_pages: Словарь {file_name: [page_numbers]}
        k: количество top-k позиций

    Returns:
        NDCG@k score
    """
    # Вектор релевантности
    relevance = []
    for chunk in retrieved_chunks[:k]:
        file_name = chunk.get("file_name") or chunk.get("metadata", {}).get("file_name")
        page_number = chunk.get("metadata", {}).get("page_number")

        is_relevant = 0
        if file_name and page_number:
            expected = expected_pages.get(file_name, [])
            if page_number in expected:
                is_relevant = 1

        relevance.append(is_relevant)

    # DCG
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))

    # IDCG
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def compute_all(
    retrieved_chunks: List[Dict[str, Any]],
    expected_pages: Dict[str, List[int]],
    k: int
) -> Dict[str, float]:
    """
    Вычисляет все retrieval-метрики за один вызов (page-level).

    Args:
        retrieved_chunks: Список словарей чанков
        expected_pages: Словарь {file_name: [page_numbers]}
        k: количество top-k позиций

    Returns:
        {"map": ..., "recall_at_k": ..., "mrr": ..., "ndcg_at_k": ...}
    """
    return {
        "map":         round(average_precision(retrieved_chunks, expected_pages, k), 4),
        "recall_at_k": round(recall_at_k(retrieved_chunks, expected_pages), 4),
        "mrr":         round(mean_reciprocal_rank(retrieved_chunks, expected_pages), 4),
        "ndcg_at_k":   round(ndcg_at_k(retrieved_chunks, expected_pages, k), 4),
    }
