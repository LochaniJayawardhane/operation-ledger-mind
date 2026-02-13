"""
Reciprocal Rank Fusion (RRF)

Combines dense and BM25 retrieval result lists into a single
fused ranking.

    RRF_score(doc) = Σ  1 / (k + rank_i)   for each ranking i

Reference: Cormack, Clarke & Büttcher (2009) – "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning methods".
"""

from __future__ import annotations

from typing import Any, Dict, List


def rrf_fusion(
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    *,
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Fuse two ranked result lists using RRF.

    Parameters
    ----------
    dense_results : list
        Results from dense (vector) retrieval – ordered by relevance.
    bm25_results : list
        Results from BM25 (keyword) retrieval – ordered by relevance.
    k : int
        RRF constant; higher → more weight to lower-ranked docs.
        Default 60 (from the original paper).

    Returns
    -------
    list
        Fused results sorted by descending RRF score. Each dict is
        augmented with ``rrf_score``, ``dense_rank``, and ``bm25_rank``.
    """
    # Map id → result dict  (first-seen copy wins)
    id_to_result: Dict[str, Dict[str, Any]] = {}
    rrf_scores: Dict[str, float] = {}

    # --- Dense ranking contribution ---
    for rank, res in enumerate(dense_results, start=1):
        doc_id = res["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in id_to_result:
            id_to_result[doc_id] = {**res, "dense_rank": rank, "bm25_rank": None}
        else:
            id_to_result[doc_id]["dense_rank"] = rank

    # --- BM25 ranking contribution ---
    for rank, res in enumerate(bm25_results, start=1):
        doc_id = res["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in id_to_result:
            id_to_result[doc_id] = {**res, "dense_rank": None, "bm25_rank": rank}
        else:
            id_to_result[doc_id]["bm25_rank"] = rank

    # --- Build fused list ---
    fused: List[Dict[str, Any]] = []
    for doc_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        entry = id_to_result[doc_id]
        entry["rrf_score"] = score
        fused.append(entry)

    return fused
