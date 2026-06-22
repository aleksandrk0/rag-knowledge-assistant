"""Промпты. Главный анти-галлюцинационный приём: жёсткая инструкция
отвечать ТОЛЬКО по контексту и явный «escape hatch» — «Не нашёл в базе знаний».
"""
from __future__ import annotations

from ..types import Chunk

SYSTEM_PROMPT = (
    "Ты ассистент по базе знаний. Отвечай СТРОГО по приведённому контексту. "
    "Если ответа в контексте нет — ответь ровно: «Не нашёл в базе знаний». "
    "Не выдумывай фактов. В конце укажи источники в формате [источник]."
)

NO_ANSWER = "Не нашёл в базе знаний."


def build_user_prompt(question: str, contexts: list[Chunk]) -> str:
    ctx = "\n\n".join(f"[{c.source}] {c.text}" for c in contexts)
    return f"Контекст:\n{ctx}\n\nВопрос: {question}\n\nОтвет:"
