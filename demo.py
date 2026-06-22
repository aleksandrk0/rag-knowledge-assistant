"""End-to-end демо: индексирует data/sample и отвечает на вопросы.

Запуск без ключей и внешних сервисов:
    python demo.py
По умолчанию: E5-эмбеддинги + BM25 + RRF + экстрактивный фоллбэк-генератор.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")  # Windows-консоль: кириллица в stdout

from config.settings import settings  # noqa: E402
from rag.factory import build_pipeline  # noqa: E402
from rag.ingestion.loaders import load_dir  # noqa: E402

QUESTIONS = [
    "Какое стандартное значение константы k в RRF?",
    "Какие префиксы обязательно требует модель E5?",
    "В каком формате QLoRA квантует базовую модель?",
    "Что стоит на первом месте в OWASP LLM Top-10?",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore", "sentence_transformers", "transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    docs = load_dir(ROOT / "data" / "sample")
    print(f"Загружено документов: {len(docs)}")

    pipe = build_pipeline(settings)
    n = pipe.ingest(docs)
    print(f"Проиндексировано кусков: {n}\n")

    for q in QUESTIONS:
        ans = pipe.answer(q)
        print(f"❓ {q}")
        print(f"💬 {ans.text}")
        print(f"📎 источники: {', '.join(ans.sources)}")
        print(f"⏱  {ans.timings_ms}\n")


if __name__ == "__main__":
    main()
