"""Загрузка документов. Поддержка .md/.txt из коробки; .pdf — опционально."""
from __future__ import annotations

from pathlib import Path

_TEXT_SUFFIXES = {".md", ".txt"}


def load_dir(path: str | Path) -> list[tuple[str, str]]:
    """Возвращает список (имя_источника, текст) из директории (рекурсивно)."""
    docs: list[tuple[str, str]] = []
    for f in sorted(Path(path).glob("**/*")):
        if f.is_file() and f.suffix.lower() in _TEXT_SUFFIXES:
            docs.append((f.name, f.read_text(encoding="utf-8")))
    return docs
