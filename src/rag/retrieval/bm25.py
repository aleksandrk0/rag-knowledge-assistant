"""Лексический поиск BM25 (Okapi). Ловит точные термины/числа/коды,
на которых чистый векторный поиск промахивается. Половина гибрида.
"""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from ..types import Chunk, Scored

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks])

    def search(self, query: str, top_k: int) -> list[Scored]:
        scores = self._bm25.get_scores(_tokenize(query))
        idx = np.argsort(-scores)[:top_k]
        return [Scored(self.chunks[i], float(scores[i])) for i in idx]
