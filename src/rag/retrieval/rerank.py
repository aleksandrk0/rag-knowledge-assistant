"""Реранк кандидатов кросс-энкодером.

Гибридный поиск даёт top_k кандидатов (recall), кросс-энкодер пересортирует
их по точной паре (запрос, кусок) и оставляет top_n (precision) — это убирает
мусор из контекста LLM и поднимает faithfulness. Дорого, поэтому только на
финальном узком наборе. NoopReranker — фоллбэк без модели (для CI/демо).
"""
from __future__ import annotations

from ..types import Scored


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder  # ленивый импорт

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, scored: list[Scored], top_n: int) -> list[Scored]:
        if not scored:
            return []
        pairs = [(query, s.chunk.text) for s in scored]
        scores = self._model.predict(pairs, show_progress_bar=False)
        order = sorted(range(len(scored)), key=lambda i: -scores[i])[:top_n]
        return [Scored(scored[i].chunk, float(scores[i])) for i in order]


class NoopReranker:
    """Без модели: отдаёт первые top_n как есть (порядок гибрида)."""

    def rerank(self, query: str, scored: list[Scored], top_n: int) -> list[Scored]:
        return scored[:top_n]
