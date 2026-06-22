"""Контроль выхода модели: утечка системного промпта и эксфильтрация по URL."""
from __future__ import annotations

from .events import SecurityEvent
from .patterns import URL_PATTERN


def guard_output(answer: str, canary: str | None = None) -> tuple[str, list[SecurityEvent]]:
    events: list[SecurityEvent] = []
    safe = answer

    # OWASP LLM02: утечка системного промпта через канарейку.
    if canary and canary in safe:
        events.append(
            SecurityEvent("output", "system_prompt_leak", "high", "answer", "canary token leaked")
        )
        safe = safe.replace(canary, "[REDACTED]")

    # OWASP LLM01/LLM05: эксфильтрация данных через URL в ответе.
    urls = URL_PATTERN.findall(safe)
    for url in urls:
        events.append(
            SecurityEvent("output", "exfiltration_url", "high", "answer", url[:80])
        )
    if urls:
        safe = URL_PATTERN.sub("[BLOCKED-URL]", safe)

    return safe, events
