"""Векторное хранилище в памяти (numpy, косинус по нормированным векторам).

Дефолт — чтобы демо запускалось одной командой без внешних сервисов.
Для прода есть адаптер Qdrant (store/qdrant_store.py), интерфейс совпадает.
"""
from __future__ import annotations

import numpy as np

from ..types import Chunk, Scored


class MemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vecs: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks и vectors разной длины")
        self._chunks.extend(chunks)
        self._vecs = vectors if self._vecs is None else np.vstack([self._vecs, vectors])

    def search(self, query_vec: np.ndarray, top_k: int) -> list[Scored]:
        if self._vecs is None or not self._chunks:
            return []
        sims = self._vecs @ query_vec  # косинус: векторы уже нормированы
        idx = np.argsort(-sims)[:top_k]
        return [Scored(self._chunks[i], float(sims[i])) for i in idx]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)
