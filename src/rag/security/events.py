"""Событие безопасности — единица наблюдаемости для аудита и алертов."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SecurityEvent:
    stage: str       # "context" (вход) | "output" (выход)
    kind: str        # имя детектора
    severity: str    # "high" | "medium"
    source: str      # источник куска или "answer"
    detail: str      # фрагмент-улика
