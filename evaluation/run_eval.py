"""Ablation-оценка качества поиска + проверка отказа на вопросах вне базы.

Сравнивает четыре режима ретрива — vector / bm25 / hybrid / hybrid+rerank — по
recall@1, recall@3 и MRR на золотом наборе. Дополнительно разбивает по типу
вопроса (lexical / semantic), чтобы показать, что гибрид забирает лучшее из
обоих каналов, а реранк поднимает позицию верного документа.

Запуск:  python evaluation/run_eval.py
Цифрами из вывода заполняется раздел «Бенчмарки» в README.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")  # Windows-консоль: кириллица в stdout

from config.settings import settings  # noqa: E402
from rag.embeddings.encoder import E5Encoder  # noqa: E402
from rag.eval.metrics import lexical_faithfulness, recall_at_k, reciprocal_rank  # noqa: E402
from rag.generation.prompts import NO_ANSWER  # noqa: E402
from rag.ingestion.loaders import load_dir  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.retrieval.rerank import CrossEncoderReranker  # noqa: E402

MODES = ["vector", "bm25", "hybrid", "hybrid_rerank"]


def avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def load_qa() -> list[dict]:
    path = Path(__file__).parent / "qa_dataset.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def sources_for(pipe: RAGPipeline, question: str, mode: str) -> list[str]:
    return [s.chunk.source for s in pipe.retrieve(question, mode=mode)]


def main() -> None:
    docs = load_dir(ROOT / "data" / "sample")
    encoder = E5Encoder(settings.embedding_model)
    reranker = CrossEncoderReranker(settings.rerank_model)
    pipe = RAGPipeline(settings, encoder=encoder, reranker=reranker)
    n_chunks = pipe.ingest(docs)

    qa = load_qa()
    in_scope = [q for q in qa if q["gold_source"]]
    oos = [q for q in qa if not q["gold_source"]]
    print(f"Документов: {len(docs)}  кусков: {n_chunks}  "
          f"вопросов: {len(in_scope)} в базе + {len(oos)} вне базы\n")

    # --- Таблица 1: ablation по режимам ---
    stats = {m: {"r@1": [], "r@3": [], "mrr": []} for m in MODES}
    for item in in_scope:
        for m in MODES:
            srcs = sources_for(pipe, item["question"], m)
            stats[m]["r@1"].append(recall_at_k(srcs, item["gold_source"], 1))
            stats[m]["r@3"].append(recall_at_k(srcs, item["gold_source"], 3))
            stats[m]["mrr"].append(reciprocal_rank(srcs, item["gold_source"]))

    print("Режим поиска       recall@1  recall@3   MRR")
    print("-" * 48)
    for m in MODES:
        print(f"{m:<18} {avg(stats[m]['r@1']):>7.2f} {avg(stats[m]['r@3']):>9.2f} "
              f"{avg(stats[m]['mrr']):>6.2f}")

    # --- Таблица 2: recall@1 по типу вопроса ---
    print("\nrecall@1 по типу вопроса:")
    print("Режим поиска        lexical  semantic")
    print("-" * 40)
    for m in MODES:
        lex = [recall_at_k(sources_for(pipe, q["question"], m), q["gold_source"], 1)
               for q in in_scope if q.get("type") == "lexical"]
        sem = [recall_at_k(sources_for(pipe, q["question"], m), q["gold_source"], 1)
               for q in in_scope if q.get("type") == "semantic"]
        print(f"{m:<18} {avg(lex):>8.2f} {avg(sem):>9.2f}")

    # --- Генерация: отказ на вопросах вне базы + faithfulness против контекста ---
    abstain = sum(1 for q in oos if pipe.answer(q["question"]).text == NO_ANSWER)
    faith = []
    for q in in_scope:
        ans = pipe.answer(q["question"])
        faith.append(lexical_faithfulness(ans.text, ans.contexts))  # ответ против найденного
    print(f"\nLLM-провайдер: {settings.llm_provider}")
    print(f"Отказ на вопросах вне базы: {abstain}/{len(oos)}")
    print(f"faithfulness(lex) в базе:   {avg(faith):.2f}")


if __name__ == "__main__":
    main()
