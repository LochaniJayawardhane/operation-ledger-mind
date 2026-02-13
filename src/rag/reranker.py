"""
Cross-Encoder Reranker

Takes the fused candidate list and rescores each (query, doc) pair
with a cross-encoder model for higher-precision relevance ordering.

Unlike bi-encoders (which encode query and document separately),
cross-encoders see the query **and** document together – this gives
higher accuracy but is too slow to run on the full corpus, so we
only apply it to the top-M fused candidates.
"""

from __future__ import annotations

from typing import Any, Dict, List

from sentence_transformers import CrossEncoder


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    cross_encoder: CrossEncoder,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Rerank *candidates* using the cross-encoder.

    Parameters
    ----------
    query : str
        The user question.
    candidates : list
        Fused result dicts (must have a ``"content"`` key).
    cross_encoder : CrossEncoder
        A loaded ``sentence_transformers.CrossEncoder`` model.
    top_k : int
        Number of top results to return after reranking.

    Returns
    -------
    list
        Top-K candidates sorted by ``rerank_score`` (descending).
    """
    if not candidates:
        return []

    # Build (query, document) pairs for the cross-encoder
    pairs = [[query, cand["content"]] for cand in candidates]

    # Cross-encoder returns a relevance score per pair
    scores = cross_encoder.predict(pairs)

    # Attach scores
    for cand, score in zip(candidates, scores):
        cand["rerank_score"] = float(score)

    # Sort descending by rerank score and take top-k
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]
