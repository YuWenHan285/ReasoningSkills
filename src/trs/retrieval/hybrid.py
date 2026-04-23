from __future__ import annotations

from collections import defaultdict

from trs.retrieval.bm25 import BM25Index
from trs.retrieval.dense import DenseIndex


def _minmax(pairs: list[tuple[int, float]]) -> dict[int, float]:
    if not pairs:
        return {}
    scores = [s for _, s in pairs]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-8:
        return {i: 1.0 for i, _ in pairs}
    return {i: (s - lo) / (hi - lo) for i, s in pairs}


def hybrid_search(
    *,
    query: str,
    bm25: BM25Index,
    dense: DenseIndex,
    top_k: int,
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
    candidate_k: int | None = None,
) -> list[tuple[int, float]]:
    candidate_k = candidate_k or max(top_k * 3, 20)
    bm25_pairs = bm25.search(query, candidate_k)
    dense_pairs = dense.search_local(query, candidate_k)
    b_norm = _minmax(bm25_pairs)
    d_norm = _minmax(dense_pairs)
    merged: dict[int, float] = defaultdict(float)
    for idx, score in b_norm.items():
        merged[idx] += bm25_weight * score
    for idx, score in d_norm.items():
        merged[idx] += dense_weight * score
    return sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]
