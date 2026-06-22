from rag.retrieval.hybrid import rrf_fuse
from rag.types import Chunk, Scored


def _c(cid):
    return Chunk(cid, f"text {cid}", "s")


def test_rrf_rewards_agreement_across_lists():
    a, b, c = _c("a"), _c("b"), _c("c")
    # a и b высоко в обоих списках, c — внизу. Шкалы разные (косинус vs BM25).
    list_vec = [Scored(a, 0.9), Scored(b, 0.8), Scored(c, 0.1)]
    list_bm25 = [Scored(b, 12.0), Scored(a, 9.0), Scored(c, 0.5)]
    fused = rrf_fuse([list_vec, list_bm25], k=60, top_k=3)
    ids = [s.chunk.id for s in fused]
    assert ids[-1] == "c"
    assert set(ids[:2]) == {"a", "b"}


def test_rrf_deduplicates():
    a = _c("a")
    fused = rrf_fuse([[Scored(a, 0.9)], [Scored(a, 5.0)]], k=60, top_k=10)
    assert len(fused) == 1
    assert fused[0].chunk.id == "a"
