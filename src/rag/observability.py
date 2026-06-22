"""Минимальная наблюдаемость: тайминги шагов пайплайна.

В проде сюда подключается Langfuse/OpenTelemetry; здесь — лёгкая обёртка,
чтобы каждый ответ нёс p50/p95-метрики по шагам (embed/retrieve/rerank/generate).
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger("rag")


@contextmanager
def timed(store: dict[str, float], name: str):
    """Замеряет время блока в мс и кладёт в store[name]."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = round((time.perf_counter() - t0) * 1000, 1)
        store[name] = dt
        logger.debug("step=%s ms=%s", name, dt)
