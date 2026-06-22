import pytest


def test_answer_returns_relevant_source(ingested):
    ans = ingested.answer("Что делает RRF?")
    assert ans.text
    assert "retrieval.md" in ans.sources
    assert {"embed", "retrieve", "rerank", "generate"} <= set(ans.timings_ms)


def test_answer_before_ingest_raises(pipeline):
    with pytest.raises(RuntimeError):
        pipeline.answer("любой вопрос")


def test_no_context_returns_no_answer_phrase(ingested):
    # вопрос без пересечений со словарём -> экстрактивный фоллбэк честно молчит
    ans = ingested.answer("xyzqwerty неизвестное_слово_которого_нет")
    assert ans.text == "Не нашёл в базе знаний."
