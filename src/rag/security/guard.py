"""SecurityGuard — оркестратор защиты в глубину.

Вход (screen_context): санитизация невидимых символов -> детект инъекций ->
карантин отравленных кусков ДО попадания в контекст модели.
Выход (guard_output): контроль утечки системного промпта и URL-эксфильтрации.
"""
from __future__ import annotations

from ..types import Chunk
from .detectors import is_malicious, scan_text
from .events import SecurityEvent
from .output_guard import guard_output as _guard_output
from .sanitize import has_smuggling, sanitize

DEFAULT_CANARY = "CANARY-9b3f7e2a-do-not-reveal"


class SecurityGuard:
    def __init__(self, canary: str = DEFAULT_CANARY, quarantine: bool = True) -> None:
        self.canary = canary
        self.quarantine = quarantine

    def screen_context(self, chunks: list[Chunk]) -> tuple[list[Chunk], list[SecurityEvent]]:
        clean: list[Chunk] = []
        events: list[SecurityEvent] = []
        for c in chunks:
            local: list[SecurityEvent] = []
            if has_smuggling(c.text):
                local.append(
                    SecurityEvent("context", "unicode_smuggling", "high", c.source,
                                  "invisible/bidi characters")
                )
            sanitized = sanitize(c.text)
            local.extend(scan_text(sanitized, c.source, "context"))
            events.extend(local)
            if self.quarantine and is_malicious(local):
                continue  # отравленный кусок не доходит до модели
            clean.append(Chunk(c.id, sanitized, c.source))
        return clean, events

    def guard_output(self, answer: str) -> tuple[str, list[SecurityEvent]]:
        return _guard_output(answer, self.canary)
