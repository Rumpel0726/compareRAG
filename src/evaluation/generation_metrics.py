# -*- coding: utf-8 -*-
"""
LLM-as-a-Judge для оценки этапа генерации.

Использует тот же LMStudioClient, что и основная система,
для оценки ответа по 4 критериям на шкале 1–5.
"""

import json
import re
from typing import Optional, Dict
from loguru import logger

from ..llm.lm_studio_client import LMStudioClient
from ..llm.prompts import (
    create_answer_evaluation_prompt,
    create_answer_evaluation_academic_prompt,
)


class LLMJudge:
    """
    Оценивает качество ответа RAG системы через LLM.

    Критерии (1–5):
        correctness  — факты соответствуют эталону (или контексту, если эталона нет)
        relevance    — ответ отвечает на вопрос
        completeness — все аспекты вопроса охвачены
        coherence    — логичность и связность изложения
    """

    def __init__(self, llm_client: LMStudioClient):
        self.llm_client = llm_client

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
        reference_answer: Optional[str] = None,
    ) -> Optional[Dict[str, int]]:
        """
        Оценивает ответ по 4 критериям.

        Если передан reference_answer — сравнивает с эталонным ответом.
        Иначе — сравнивает с контекстом (устаревший режим).

        Returns:
            {"correctness": int, "relevance": int, "completeness": int, "coherence": int}
            или None при ошибке парсинга / генерации.
        """
        if reference_answer:
            prompt = create_answer_evaluation_academic_prompt(
                query=query, answer=answer, context=context, reference_answer=reference_answer
            )
            expected_keys = ("answer_correctness", "faithfulness", "answer_relevancy", "completeness")
        else:
            prompt = create_answer_evaluation_prompt(
                query=query, answer=answer, context=context
            )
            expected_keys = ("correctness", "relevance", "completeness", "coherence")

        messages = [
            {"role": "system", "content": "Ты — эксперт по оценке качества текста. Отвечай ТОЛЬКО валидным JSON без дополнительных комментариев."},
            {"role": "user", "content": prompt}
        ]

        try:
            raw_response = self.llm_client.generate(
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            return self._parse_scores(raw_response, expected_keys)

        except Exception as e:
            logger.warning(f"LLMJudge ошибка генерации: {e}")
            return None

    def _parse_scores(self, raw: str, expected_keys: tuple) -> Optional[Dict[str, int]]:
        """
        Извлекает JSON из ответа LLM.

        Обрабатывает обёртки ```json ... ```,
        а также случай когда JSON окружён произвольным текстом.
        """
        # Попытка 1: markdown-блок ```json ... ```
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Попытка 2: вырезать от первого { до последнего }
            start = raw.find("{")
            end = raw.rfind("}")
            json_str = raw[start:end + 1] if start != -1 and end > start else raw

        try:
            data = json.loads(json_str)
            scores = {}
            for key in expected_keys:
                val = data.get(key)
                if val is None:
                    logger.warning(f"LLMJudge: отсутствует поле '{key}' в ответе")
                    return None
                # Clamp в диапазон 1–5
                scores[key] = max(1, min(5, int(val)))
            return scores

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"LLMJudge: не удалось спарсить JSON: {e}\nRaw: {raw[:300]}")
            return None
