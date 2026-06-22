"""Чанкинг: режем документ на куски ~chunk_size символов с перекрытием.

Структурно-осознанный: сначала по абзацам (пустая строка), длинные абзацы — по
предложениям. Перекрытие (overlap) переносит хвост предыдущего куска в начало
следующего — это лечит «потерю контекста на границах», классический баг RAG.
"""
from __future__ import annotations

import re

_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    paras = [p.strip() for p in _PARA.split(text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 1 <= chunk_size:
            cur = f"{cur}\n{p}".strip()
        else:
            if cur:
                chunks.append(cur)
            if len(p) <= chunk_size:
                cur = p
            else:
                chunks.extend(_split_by_sentences(p, chunk_size))
                cur = ""
    if cur:
        chunks.append(cur)
    return _apply_overlap(chunks, overlap)


def _split_by_sentences(text: str, size: int) -> list[str]:
    out: list[str] = []
    cur = ""
    for s in _SENT.split(text):
        if len(cur) + len(s) + 1 <= size:
            cur = f"{cur} {s}".strip()
        else:
            if cur:
                out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    return out


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:]
        out.append(f"{tail} {chunks[i]}".strip())
    return out
