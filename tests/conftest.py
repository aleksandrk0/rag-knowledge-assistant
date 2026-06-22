"""Фикстуры тестов. FakeEncoder — детерминированный энкодер на хешировании слов:
не качает модель, работает в CI за миллисекунды. Реальный E5 проверяется отдельно
(не в CI), а логика пайплайна — здесь, на фейке.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config.settings import Settings  # noqa: E402
from rag.generation.llm import LLMClient  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.retrieval.rerank import NoopReranker  # noqa: E402

_WORD = re.compile(r"\w+", re.UNICODE)


class FakeEncoder:
    dim = 64

    def _embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _WORD.findall(text.lower()):
            idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
            v[idx] += 1.0
        norm = float(np.linalg.norm(v))
        return (v / norm).astype(np.float32) if norm else v

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._embed(t) for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._embed(text)


DOCS = [
    ("retrieval.md", "Гибридный поиск объединяет векторный и лексический поиск BM25. "
                     "RRF объединяет списки результатов по позиции."),
    ("quality.md", "Реранкер оставляет самые точные куски и поднимает faithfulness. "
                   "Перекрытие лечит потерю контекста на границах."),
]


@pytest.fixture
def settings():
    return Settings(llm_provider="fallback", use_rerank=False)


@pytest.fixture
def pipeline(settings):
    return RAGPipeline(settings, encoder=FakeEncoder(),
                       reranker=NoopReranker(), llm=LLMClient(settings))


@pytest.fixture
def ingested(pipeline):
    pipeline.ingest(DOCS)
    return pipeline
