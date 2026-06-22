"""Эмбеддинги. По умолчанию multilingual-E5 (русский+английский).

E5 ОБЯЗАТЕЛЬНО требует префиксы 'query:' и 'passage:' — без них качество
проседает. Это частая ошибка; вынесено в отдельные методы encode_query/
encode_passages, чтобы её нельзя было допустить на стороне вызова.

Encoder спрятан за Protocol — в тестах/CI подставляется FakeEncoder без
скачивания модели (см. tests/conftest.py).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EncoderProtocol(Protocol):
    dim: int

    def encode_passages(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class E5Encoder:
    """Реальный энкодер на sentence-transformers. Модель качается при 1-м старте."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small") -> None:
        from sentence_transformers import SentenceTransformer  # ленивый импорт

        self._model = SentenceTransformer(model_name)
        try:
            self.dim = int(self._model.get_embedding_dimension())
        except AttributeError:  # старые версии sentence-transformers
            self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"passage: {t}" for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([f"query: {text}"])[0]

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
