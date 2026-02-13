"""
Retrieval Module – Dense (nearVector) + BM25 against Weaviate

Each function returns a list of normalised result dicts:
    {
        "id":       str,        # Weaviate UUID
        "content":  str,        # chunk text
        "meta":     dict,       # chunk_id, source_section, page_start, …
        "score":    float,      # raw score from the retriever
    }
"""

from __future__ import annotations

from typing import Any, Dict, List

import weaviate
from sentence_transformers import SentenceTransformer
from weaviate.classes.query import MetadataQuery


# ---------------------------------------------------------------------------
# Internal: normalise a Weaviate result object into our shared dict format
# ---------------------------------------------------------------------------

def _normalise(obj, score: float | None = None) -> Dict[str, Any]:
    props = obj.properties
    return {
        "id":      str(obj.uuid),
        "content": props.get("content", ""),
        "meta": {
            "chunk_id":       props.get("chunk_id"),
            "doc_name":       props.get("doc_name"),
            "report_year":    props.get("report_year"),
            "source_section": props.get("source_section"),
            "page_start":     props.get("page_start"),
            "page_end":       props.get("page_end"),
            "start_char":     props.get("start_char"),
            "end_char":       props.get("end_char"),
        },
        "score": score,
    }


# ---------------------------------------------------------------------------
# Dense (vector) search
# ---------------------------------------------------------------------------

def dense_search(
    question: str,
    *,
    client: weaviate.WeaviateClient,
    embedder: SentenceTransformer,
    config: Dict[str, Any],
    top_n: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Embed *question* with the same SentenceTransformer used at index time,
    then run a ``near_vector`` query against Weaviate.
    """
    top_n = top_n or int(config["rag"]["retrieval"].get("top_k", 20))
    class_name = config["rag"]["vector_db"].get("class_name", "FinancialDocument")

    query_vec = embedder.encode(question, normalize_embeddings=True).tolist()

    collection = client.collections.get(class_name)
    response = collection.query.near_vector(
        near_vector=query_vec,
        limit=top_n,
        return_metadata=MetadataQuery(distance=True),
    )

    results: List[Dict[str, Any]] = []
    for obj in response.objects:
        # Weaviate returns *distance*; convert to similarity (1 - distance)
        dist = obj.metadata.distance if obj.metadata.distance is not None else 0.0
        results.append(_normalise(obj, score=1.0 - dist))

    return results


# ---------------------------------------------------------------------------
# BM25 (keyword / sparse) search
# ---------------------------------------------------------------------------

def bm25_search(
    question: str,
    *,
    client: weaviate.WeaviateClient,
    config: Dict[str, Any],
    top_n: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Run a Weaviate BM25 keyword query over the ``content`` property.
    """
    top_n = top_n or int(config["rag"]["retrieval"].get("top_k", 20))
    class_name = config["rag"]["vector_db"].get("class_name", "FinancialDocument")
    text_key = config["rag"]["vector_db"].get("text_key", "content")

    collection = client.collections.get(class_name)
    response = collection.query.bm25(
        query=question,
        query_properties=[text_key],
        limit=top_n,
        return_metadata=MetadataQuery(score=True),
    )

    results: List[Dict[str, Any]] = []
    for obj in response.objects:
        sc = obj.metadata.score if obj.metadata.score is not None else 0.0
        results.append(_normalise(obj, score=sc))

    return results
