"""Сборка пайплайна из настроек. Общая точка для demo/api/eval, чтобы
выбор энкодера/хранилища/реранка жил в одном месте.
"""
from __future__ import annotations

from .embeddings.encoder import E5Encoder
from .pipeline import RAGPipeline
from .retrieval.rerank import CrossEncoderReranker, NoopReranker


def build_pipeline(settings) -> RAGPipeline:
    encoder = E5Encoder(settings.embedding_model)

    store = None
    if settings.vector_store == "qdrant":
        from .store.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore(settings.qdrant_url, dim=encoder.dim)

    reranker = (
        CrossEncoderReranker(settings.rerank_model)
        if settings.use_rerank
        else NoopReranker()
    )
    return RAGPipeline(settings, encoder=encoder, reranker=reranker, store=store)
