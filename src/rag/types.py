"""Базовые типы данных. Один источник истины для всех модулей."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Chunk:
    """Кусок документа после чанкинга."""
    id: str
    text: str
    source: str


@dataclass(slots=True)
class Scored:
    """Кусок с оценкой релевантности от ретривера/реранкера."""
    chunk: Chunk
    score: float
