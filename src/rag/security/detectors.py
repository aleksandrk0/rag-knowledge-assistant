"""Детекторы инъекций: сканируют (санитизированный) текст по сигнатурам."""
from __future__ import annotations

from .events import SecurityEvent
from .patterns import INJECTION_PATTERNS


def scan_text(text: str, source: str, stage: str = "context") -> list[SecurityEvent]:
    events: list[SecurityEvent] = []
    for name, pattern in INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            events.append(
                SecurityEvent(stage, name, "high", source, match.group(0)[:80])
            )
    return events


def is_malicious(events: list[SecurityEvent]) -> bool:
    return any(e.severity == "high" for e in events)
