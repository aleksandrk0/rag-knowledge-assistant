from rag.ingestion.chunking import chunk_text


def test_empty_returns_nothing():
    assert chunk_text("", chunk_size=100, overlap=10) == []


def test_long_text_splits_into_multiple():
    text = "\n\n".join(f"Абзац номер {i} с некоторым содержимым." for i in range(30))
    chunks = chunk_text(text, chunk_size=120, overlap=0)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_overlap_carries_tail():
    text = "\n\n".join(["Первый абзац достаточно длинный для разбиения.",
                        "Второй абзац идёт следом за первым."])
    chunks = chunk_text(text, chunk_size=40, overlap=12)
    assert len(chunks) >= 2
    # инвариант перекрытия: кусок i начинается с хвоста куска i-1
    tail = chunks[0][-12:]
    assert chunks[1].startswith(tail)
