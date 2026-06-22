from rag.retrieval.bm25 import BM25Index
from rag.types import Chunk


def test_bm25_finds_exact_term():
    # Корпус из нескольких документов: редкий термин даёт положительный IDF
    # (на N=2 IDF Okapi обнуляется — это артефакт малого корпуса, не баг).
    chunks = [
        Chunk("a#0", "векторный поиск находит документы по смыслу", "a"),
        Chunk("b#0", "лексический поиск находит точный артикул XYZ123", "b"),
        Chunk("c#0", "семантический поиск по смыслу запроса", "c"),
        Chunk("d#0", "гибридный поиск объединяет два подхода", "d"),
        Chunk("e#0", "реранкер пересортировывает кандидатов", "e"),
    ]
    res = BM25Index(chunks).search("XYZ123", top_k=3)
    assert res[0].chunk.id == "b#0"


def test_bm25_returns_at_most_top_k():
    chunks = [Chunk(f"c#{i}", f"текст номер {i}", "c") for i in range(5)]
    res = BM25Index(chunks).search("текст", top_k=3)
    assert len(res) == 3
