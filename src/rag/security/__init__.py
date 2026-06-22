"""LLM-безопасность для RAG: защита от непрямых промпт-инъекций и эксфильтрации.

Покрывает риски OWASP LLM Top-10: LLM01 (Prompt Injection), LLM02 (Sensitive
Information Disclosure), LLM05 (Improper Output Handling), LLM08 (Vector &
Embedding Weaknesses — отравление базы знаний).
"""
from .guard import SecurityGuard

__all__ = ["SecurityGuard"]
