"""Слияние ранжирований через RRF (Reciprocal Rank Fusion).

RRF складывает несколько списков по позиции, не по «сырым» оценкам — поэтому
несравнимые шкалы (косинус 0..1 и BM25 0..N) корректно объединяются без
ручной нормализации. Формула: score(d) = sum_i 1 / (k + rank_i(d)).
k=60 — стандарт из статьи Cormack et al., 2009.
"""
from __future__ import annotations

from ..types import Chunk, Scored


def rrf_fuse(rankings: list[list[Scored]], k: int = 60, top_k: int = 20) -> list[Scored]:
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, scored in enumerate(ranking):
            cid = scored.chunk.id
            by_id[cid] = scored.chunk
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    return [Scored(by_id[cid], s) for cid, s in ordered]
