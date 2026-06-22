"""Адаптер Qdrant (прод-вектор-БД). Интерфейс совпадает с MemoryVectorStore.

Включается через RAG_VECTOR_STORE=qdrant + docker-compose up qdrant.
Импорт qdrant_client ленивый, чтобы базовое демо не требовало этой зависимости.
"""
from __future__ import annotations

import uuid

import numpy as np

from ..types import Chunk, Scored

_COLLECTION = "rag_chunks"


class QdrantVectorStore:
    def __init__(self, url: str, dim: int, collection: str = _COLLECTION) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = QdrantClient(url=url)
        self._collection = collection
        self._client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec.tolist(),
                payload={"id": c.id, "text": c.text, "source": c.source},
            )
            for c, vec in zip(chunks, vectors, strict=False)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[Scored]:
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=query_vec.tolist(),
            limit=top_k,
        )
        out: list[Scored] = []
        for h in hits:
            p = h.payload or {}
            out.append(Scored(Chunk(p["id"], p["text"], p["source"]), float(h.score)))
        return out
