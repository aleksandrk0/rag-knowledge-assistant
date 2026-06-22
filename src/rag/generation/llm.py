"""Генератор ответа. Четыре провайдера, выбор через RAG_LLM_PROVIDER:

  fallback (дефолт) — ЧЕСТНЫЙ экстрактивный ответ из предложений контекста, без
                      ключей и сети. Не «фейк-LLM»: ничего не выдумывает.
  openai            — любой OpenAI-совместимый API: OpenAI, а также ЛОКАЛЬНЫЕ
                      Ollama / vLLM / LM Studio (base_url=http://localhost:.../v1).
  local             — локальная модель через transformers (на RTX 4090), без сети.
  gigachat          — GigaChat: OAuth-токен, затем chat/completions.

Все импорты тяжёлых зависимостей ленивые — базовый режим их не требует.
"""
from __future__ import annotations

import re

from ..types import Chunk
from .prompts import NO_ANSWER, SYSTEM_PROMPT, build_user_prompt

_WORD = re.compile(r"\w+", re.UNICODE)
_SENT = re.compile(r"(?<=[.!?])\s+")

# Стоп-слова: без них экстрактивный фоллбэк не цепляется за вопросительные слова
# и корректно отказывает на вопросах вне базы знаний.
_STOP = {
    "что", "как", "какой", "какая", "какое", "какие", "зачем", "почему", "чем",
    "где", "когда", "сколько", "кто", "это", "этот", "тот", "такое", "ли", "же",
    "бы", "и", "в", "во", "на", "с", "со", "по", "к", "ко", "о", "об", "от", "до",
    "за", "из", "у", "а", "но", "или", "не", "ни", "для", "при", "про", "над",
    "под", "между", "сегодня", "вообще", "ещё", "уже",
}


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


class LLMClient:
    def __init__(self, settings) -> None:
        self.settings = settings
        self._local_pipe = None  # ленивая инициализация transformers
        self._gigachat_token: str | None = None

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        if not contexts:
            return NO_ANSWER
        provider = self.settings.llm_provider
        if provider == "fallback":
            return self._extractive(question, contexts)
        if provider == "local":
            return self._local(question, contexts)
        if provider == "gigachat":
            return self._gigachat(question, contexts)
        return self._openai_compatible(question, contexts)

    # --- OpenAI-совместимый (OpenAI / Ollama / vLLM / LM Studio) ---
    def _openai_compatible(self, question: str, contexts: list[Chunk]) -> str:
        import httpx

        resp = httpx.post(
            f"{self.settings.llm_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.settings.llm_api_key}"},
            json=self._chat_payload(self.settings.llm_model, question, contexts),
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    # --- GigaChat: сначала OAuth-токен, затем chat ---
    def _gigachat(self, question: str, contexts: list[Chunk]) -> str:
        import httpx

        if self._gigachat_token is None:
            self._gigachat_token = self._fetch_gigachat_token()
        resp = httpx.post(
            f"{self.settings.gigachat_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._gigachat_token}"},
            json=self._chat_payload(self.settings.llm_model, question, contexts),
            verify=self.settings.gigachat_verify_ssl,
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _fetch_gigachat_token(self) -> str:
        import uuid

        import httpx

        resp = httpx.post(
            self.settings.gigachat_auth_url,
            headers={
                "Authorization": f"Basic {self.settings.gigachat_auth_key}",
                "RqUID": str(uuid.uuid4()),
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"scope": self.settings.gigachat_scope},
            verify=self.settings.gigachat_verify_ssl,
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    # --- Локальная модель через transformers (RTX 4090) ---
    def _local(self, question: str, contexts: list[Chunk]) -> str:
        pipe = self._get_local_pipe()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, contexts)},
        ]
        out = pipe(messages, max_new_tokens=256, do_sample=False)
        return out[0]["generated_text"][-1]["content"].strip()

    def _get_local_pipe(self):
        if self._local_pipe is None:
            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() else -1  # GPU при наличии, иначе CPU
            self._local_pipe = pipeline(
                "text-generation",
                model=self.settings.local_model,
                torch_dtype="auto",
                device=device,
            )
        return self._local_pipe

    @staticmethod
    def _chat_payload(model: str, question: str, contexts: list[Chunk]) -> dict:
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, contexts)},
            ],
            "temperature": 0.0,
        }

    # --- Экстрактивный фоллбэк (без ключей) ---
    def _extractive(self, question: str, contexts: list[Chunk]) -> str:
        q = _content_words(question)
        scored: list[tuple[int, str, str]] = []
        for c in contexts:
            for sent in _SENT.split(c.text):
                overlap = len(q & set(_WORD.findall(sent.lower())))
                if overlap:
                    scored.append((overlap, sent.strip(), c.source))
        if not scored:
            return NO_ANSWER
        scored.sort(key=lambda x: -x[0])
        top = scored[:2]
        answer = " ".join(s for _, s, _ in top)
        srcs = " ".join(dict.fromkeys(f"[{src}]" for _, _, src in top))
        return f"{answer} {srcs}"
