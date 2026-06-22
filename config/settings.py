"""Конфигурация через pydantic-settings: значения берутся из переменных
окружения с префиксом RAG_ или из .env. Один источник истины для параметров.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # --- эмбеддинги / реранк ---
    embedding_model: str = "intfloat/multilingual-e5-small"
    # мультиязычный реранкер: английский ms-marco деградирует на русском корпусе
    rerank_model: str = "DiTy/cross-encoder-russian-msmarco"
    use_rerank: bool = False  # требует загрузки кросс-энкодера; включается явно

    # --- чанкинг ---
    chunk_size: int = 500
    chunk_overlap: int = 80

    # --- ретрив ---
    top_k: int = 20  # кандидатов от каждого ретривера и после RRF
    top_n: int = 5   # после реранка -> в LLM
    rrf_k: int = 60

    # --- векторное хранилище ---
    vector_store: str = "memory"  # "memory" | "qdrant"
    qdrant_url: str = "http://localhost:6333"

    # --- LLM ---
    # fallback  — экстрактивный, без ключей (дефолт)
    # openai    — любой OpenAI-совместимый API (OpenAI, Ollama, vLLM, LM Studio)
    # local     — локальная модель через transformers (на RTX 4090)
    # gigachat  — GigaChat (OAuth + chat)
    llm_provider: str = "fallback"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"

    # локальная генерация (provider=local)
    local_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # GigaChat (provider=gigachat)
    gigachat_auth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    gigachat_base_url: str = "https://gigachat.devices.sberbank.ru/api/v1"
    gigachat_auth_key: str | None = None  # Authorization key (Base64) из кабинета
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_verify_ssl: bool = True  # False, если нет цепочки сертификатов Минцифры


settings = Settings()
