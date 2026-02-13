"""
Librarian Inference – ``query_librarian(question)``

End-to-end Advanced Hybrid RAG:

    Question → ensure_index → Dense + BM25 → RRF fusion → Cross-encoder rerank → Generate answer

Returns a dict with the answer, source chunks, scores, and pipeline stats.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder, SentenceTransformer

from src.utils.config_loader import load_config
from src.rag.weaviate_store import connect_weaviate, ensure_collection
from src.rag.index_builder import ensure_index_built
from src.rag.retrieval import dense_search, bm25_search
from src.rag.fusion import rrf_fusion
from src.rag.reranker import rerank
from src.rag.generation import generate_answer


# ---------------------------------------------------------------------------
# Module-level caches (loaded once, reused across calls)
# ---------------------------------------------------------------------------
_CACHE: Dict[str, Any] = {}


def _get_embedder(config: Dict[str, Any]) -> SentenceTransformer:
    model_name = config["rag"]["embeddings"]["model"]
    key = f"embedder:{model_name}"
    if key not in _CACHE:
        _CACHE[key] = SentenceTransformer(model_name)
    return _CACHE[key]


def _get_reranker(config: Dict[str, Any]) -> CrossEncoder:
    model_name = config["rag"]["refinement"]["reranker"]["model"]
    key = f"reranker:{model_name}"
    if key not in _CACHE:
        _CACHE[key] = CrossEncoder(model_name)
    return _CACHE[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query_librarian(
    question: str,
    *,
    config_path: str | Path = "config/config.yaml",
    generator_mode: str | None = None,
    rebuild_index: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Ask a question and get an answer grounded in the annual report.

    Parameters
    ----------
    question : str
        The user question.
    config_path : str | Path
        Path to config.yaml.
    generator_mode : str | None
        Override the generator: ``"openai"`` | ``"intern_finetuned"`` | ``"intern_base"``.
        Defaults to ``config['rag']['inference']['generator_mode']``.
    rebuild_index : bool
        If True, drop and rebuild the Weaviate index from scratch.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        ``answer``   – the generated answer string
        ``sources``  – list of top source chunks with scores and metadata
        ``stats``    – pipeline statistics (counts, latency)
    """
    t0 = time.perf_counter()

    # --- Load config ---
    config = load_config(config_path)

    # --- Models ---
    embedder = _get_embedder(config)
    cross_encoder = _get_reranker(config)

    # --- Weaviate client ---
    client = connect_weaviate(config)

    try:
        # --- Ensure index is built ---
        ensure_index_built(
            config,
            client=client,
            embedder=embedder,
            force_rebuild=rebuild_index,
            verbose=verbose,
        )

        # --- Stage 1: Dense retrieval ---
        top_n = int(config["rag"]["retrieval"].get("top_k", 20))
        t_ret = time.perf_counter()

        dense_results = dense_search(
            question, client=client, embedder=embedder, config=config, top_n=top_n,
        )

        # --- Stage 2: BM25 retrieval ---
        bm25_results = bm25_search(
            question, client=client, config=config, top_n=top_n,
        )

        t_ret_end = time.perf_counter()

        if verbose:
            print(f"🔍 Dense: {len(dense_results)} results | BM25: {len(bm25_results)} results")

        # --- Stage 3: RRF fusion ---
        rrf_k = int(config["rag"]["refinement"]["rrf"].get("k", 60))
        fused = rrf_fusion(dense_results, bm25_results, k=rrf_k)

        if verbose:
            print(f"🔀 RRF fusion: {len(fused)} unique candidates")

        # --- Stage 4: Cross-encoder reranking ---
        rerank_top_k = int(config["rag"]["refinement"]["reranker"].get("top_k", 5))
        reranked = rerank(
            question,
            fused[:top_n],  # only rerank the top-N fused candidates
            cross_encoder=cross_encoder,
            top_k=rerank_top_k,
        )

        if verbose:
            print(f"🏆 Reranked to top-{len(reranked)}")

        # --- Stage 5: Build context from top reranked chunks ---
        context = "\n\n".join(
            f"[Chunk {r['meta'].get('chunk_id', '?')}] {r['content']}"
            for r in reranked
        )

        # --- Stage 6: Generate answer ---
        mode = generator_mode or config["rag"]["inference"].get("generator_mode", "openai")
        t_gen = time.perf_counter()

        answer = generate_answer(
            question, context, config,
            mode=mode,
            config_path=config_path,
        )

        t_gen_end = time.perf_counter()

        if verbose:
            print(f"💬 Answer generated via '{mode}'")

        # --- Build sources list ---
        sources = []
        for r in reranked:
            sources.append({
                "chunk_id":       r["meta"].get("chunk_id"),
                "page_start":     r["meta"].get("page_start"),
                "page_end":       r["meta"].get("page_end"),
                "source_section": r["meta"].get("source_section"),
                "scores": {
                    "dense":  r.get("score"),
                    "rrf":    r.get("rrf_score"),
                    "rerank": r.get("rerank_score"),
                    "dense_rank": r.get("dense_rank"),
                    "bm25_rank":  r.get("bm25_rank"),
                },
                "preview": r["content"][:200] + "…" if len(r["content"]) > 200 else r["content"],
            })

        t_end = time.perf_counter()

        return {
            "answer": answer,
            "generator_mode": mode,
            "sources": sources,
            "stats": {
                "dense_top_n":    len(dense_results),
                "bm25_top_n":     len(bm25_results),
                "fused_n":        len(fused),
                "reranked_k":     len(reranked),
                "retrieval_ms":   round((t_ret_end - t_ret) * 1000, 1),
                "generation_ms":  round((t_gen_end - t_gen) * 1000, 1),
                "total_ms":       round((t_end - t0) * 1000, 1),
            },
        }

    finally:
        client.close()
