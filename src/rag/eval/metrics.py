"""Метрики качества RAG.

Поиск (по документам, на дедуплицированном ранжировании источников):
  recall_at_k     — попал ли нужный источник в top-k.
  reciprocal_rank — 1/позиция первого верного источника (основа MRR).

Ответ:
  lexical_faithfulness — доля слов ответа, покрытых контекстом (грубый прокси
                         «опоры на источники»; в проде заменяется LLM-judge/RAGAS).
"""
from __future__ import annotations

import re

_WORD = re.compile(r"\w+", re.UNICODE)


def dedup_sources(sources: list[str]) -> list[str]:
    """Список источников по порядку рангов без повторов (несколько кусков из
    одного документа схлопываются в первую позицию этого документа)."""
    return list(dict.fromkeys(sources))


def recall_at_k(sources: list[str], gold: str, k: int) -> float:
    return 1.0 if gold in dedup_sources(sources)[:k] else 0.0


def reciprocal_rank(sources: list[str], gold: str) -> float:
    deduped = dedup_sources(sources)
    for i, src in enumerate(deduped):
        if src == gold:
            return 1.0 / (i + 1)
    return 0.0


def lexical_faithfulness(answer: str, contexts: list[str]) -> float:
    a = set(_WORD.findall(answer.lower()))
    if not a:
        return 0.0
    c: set[str] = set()
    for ctx in contexts:
        c |= set(_WORD.findall(ctx.lower()))
    return round(len(a & c) / len(a), 3)
