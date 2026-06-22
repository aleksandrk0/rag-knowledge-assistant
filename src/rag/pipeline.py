"""Оркестрация RAG: ingest() и answer().

Поток answer():
  embed(query) -> [vector_search ‖ bm25] -> RRF -> rerank -> LLM
Каждый шаг таймится; ответ несёт sources и timings_ms (наблюдаемость в проде).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .embeddings.encoder import EncoderProtocol
from .generation.llm import LLMClient
from .ingestion.chunking import chunk_text
from .observability import timed
from .retrieval.bm25 import BM25Index
from .retrieval.hybrid import rrf_fuse
from .retrieval.rerank import NoopReranker
from .store.memory import MemoryVectorStore
from .types import Chunk, Scored


@dataclass(slots=True)
class Answer:
    text: str
    sources: list[str]
    contexts: list[str] = field(default_factory=list)  # тексты кусков для оценки faithfulness
    timings_ms: dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, settings, encoder: EncoderProtocol, reranker=None, llm=None, store=None):
        self.settings = settings
        self.encoder = encoder
        self.store = store or MemoryVectorStore()
        self.bm25: BM25Index | None = None
        self.reranker = reranker or NoopReranker()
        self.llm = llm or LLMClient(settings)

    def ingest(self, docs: list[tuple[str, str]]) -> int:
        """docs: список (источник, текст). Возвращает число кусков."""
        chunks: list[Chunk] = []
        for source, text in docs:
            pieces = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
            for i, piece in enumerate(pieces):
                chunks.append(Chunk(id=f"{source}#{i}", text=piece, source=source))
        if not chunks:
            raise ValueError("Нет кусков для индексации (пустые документы?)")
        vecs = self.encoder.encode_passages([c.text for c in chunks])
        self.store.add(chunks, vecs)
        self.bm25 = BM25Index(self.store.all_chunks())
        return len(chunks)

    def retrieve(self, question: str, mode: str = "hybrid_rerank") -> list[Scored]:
        """Ранжированные куски для одного из режимов поиска. Используется в
        ablation-оценке для сравнения vector / bm25 / hybrid / hybrid_rerank.
        """
        if self.bm25 is None:
            raise RuntimeError("Сначала вызовите ingest()")
        vec = self.store.search(self.encoder.encode_query(question), self.settings.top_k)
        if mode == "vector":
            return vec
        bm = self.bm25.search(question, self.settings.top_k)
        if mode == "bm25":
            return bm
        fused = rrf_fuse([vec, bm], self.settings.rrf_k, self.settings.top_k)
        if mode == "hybrid":
            return fused
        if mode == "hybrid_rerank":
            return self.reranker.rerank(question, fused, self.settings.top_n)
        raise ValueError(f"Неизвестный режим: {mode}")

    def answer(self, question: str) -> Answer:
        if self.bm25 is None:
            raise RuntimeError("Сначала вызовите ingest()")
        t: dict[str, float] = {}
        with timed(t, "embed"):
            qv = self.encoder.encode_query(question)
        with timed(t, "retrieve"):
            vec_hits = self.store.search(qv, self.settings.top_k)
            bm_hits = self.bm25.search(question, self.settings.top_k)
            fused = rrf_fuse([vec_hits, bm_hits], self.settings.rrf_k, self.settings.top_k)
        with timed(t, "rerank"):
            top = self.reranker.rerank(question, fused, self.settings.top_n)
        with timed(t, "generate"):
            text = self.llm.generate(question, [s.chunk for s in top])
        sources = list(dict.fromkeys(s.chunk.source for s in top))
        return Answer(text=text, sources=sources,
                      contexts=[s.chunk.text for s in top], timings_ms=t)
