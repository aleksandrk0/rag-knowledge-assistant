"""Санитизация недоверенного текста перед попаданием в контекст модели."""
from __future__ import annotations

import unicodedata

from .patterns import SMUGGLING_CHARS


def has_smuggling(text: str) -> bool:
    """Есть ли невидимые/bidi/tag-символы (скрытая инъекция)."""
    return bool(SMUGGLING_CHARS.search(text))


def sanitize(text: str) -> str:
    """Удаляет невидимые символы и нормализует Unicode (NFKC).
    Снимает смуглинг-кодирование, при котором инструкция спрятана между токенами.
    """
    stripped = SMUGGLING_CHARS.sub("", text)
    return unicodedata.normalize("NFKC", stripped)
