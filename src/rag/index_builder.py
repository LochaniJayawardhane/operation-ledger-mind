"""
Index Builder – PDF → Chunks → Embeddings → Weaviate

Loads the annual report, chunks it using existing ingestion utilities,
embeds each chunk with SentenceTransformers, and upserts into Weaviate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from weaviate.classes.data import DataObject

from src.ingestion.pdf_loader import load_pdf, clean_text
from src.ingestion.chunker import chunk_text
from src.rag.weaviate_store import (
    connect_weaviate,
    ensure_collection,
    deterministic_uuid,
)


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _embed_chunks(
    chunks: List[Dict[str, Any]],
    config: Dict[str, Any],
    embedder: SentenceTransformer | None = None,
) -> tuple[List[Dict[str, Any]], list]:
    """
    Embed the ``text`` field of each chunk and return (chunks, vectors).
    """
    emb_cfg = config["rag"]["embeddings"]
    if embedder is None:
        embedder = SentenceTransformer(emb_cfg["model"])

    texts = [c["text"] for c in chunks]
    batch_size = int(emb_cfg.get("batch_size", 32))
    normalize = bool(emb_cfg.get("normalize", True))

    vectors = embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    )
    return chunks, vectors.tolist()


# ---------------------------------------------------------------------------
# Core indexing
# ---------------------------------------------------------------------------

def build_index(
    config: Dict[str, Any],
    *,
    client=None,
    embedder: SentenceTransformer | None = None,
    verbose: bool = True,
) -> int:
    """
    End-to-end: load PDF → clean → chunk → embed → upsert into Weaviate.

    Returns the number of objects inserted.
    """
    # --- Resolve PDF path ---
    raw_data = Path(config["environment"]["paths"]["raw_data"])
    doc_name = config["project"]["document"]
    pdf_path = raw_data / doc_name
    report_year = int(config["project"].get("report_year", 0))

    if not pdf_path.exists():
        # Fallback: grab any PDF in the directory
        pdfs = list(raw_data.glob("*.pdf"))
        if pdfs:
            pdf_path = pdfs[0]
            doc_name = pdf_path.name
        else:
            raise FileNotFoundError(f"No PDF found in {raw_data}")

    if verbose:
        print(f"📄 Loading PDF: {pdf_path}")

    # --- Load & clean ---
    ing_cfg = config.get("data_factory", {}).get("ingestion", {})
    raw_text = load_pdf(pdf_path)
    cleaned = clean_text(
        raw_text,
        remove_headers=ing_cfg.get("remove_headers", True),
        remove_footers=ing_cfg.get("remove_footers", True),
        normalize_whitespace=ing_cfg.get("normalize_whitespace", True),
    )

    if verbose:
        print(f"✂️  Cleaned text: {len(cleaned):,} chars")

    # --- Chunk ---
    ch_cfg = config.get("data_factory", {}).get("chunking", {})
    chunks = chunk_text(
        cleaned,
        chunk_size=int(ch_cfg.get("chunk_size", 1500)),
        overlap=float(ch_cfg.get("overlap", 0.15)),
        strategy=ch_cfg.get("strategy", "semantic"),
    )

    if verbose:
        print(f"🧩 Chunks: {len(chunks)}")

    # --- Embed ---
    chunks, vectors = _embed_chunks(chunks, config, embedder=embedder)

    if verbose:
        print(f"📐 Embeddings: {len(vectors)} vectors of dim {len(vectors[0])}")

    # --- Weaviate upsert ---
    own_client = False
    if client is None:
        client = connect_weaviate(config)
        own_client = True

    try:
        ensure_collection(client, config)
        class_name = config["rag"]["vector_db"].get("class_name", "FinancialDocument")
        collection = client.collections.get(class_name)

        data_objects: List[DataObject] = []
        for chunk, vec in zip(chunks, vectors):
            meta = chunk.get("metadata", {})
            props = {
                "content":        chunk["text"],
                "chunk_id":       int(chunk["chunk_id"]),
                "doc_name":       doc_name,
                "report_year":    report_year,
                "source_section": meta.get("source_section") or "",
                "page_start":     int(meta.get("page_start") or 0),
                "page_end":       int(meta.get("page_end") or 0),
                "start_char":     int(chunk.get("start_char", 0)),
                "end_char":       int(chunk.get("end_char", 0)),
            }
            uid = deterministic_uuid(doc_name, chunk["chunk_id"])
            data_objects.append(DataObject(properties=props, vector=vec, uuid=uid))

        # Batch insert
        if verbose:
            print(f"⬆️  Upserting {len(data_objects)} objects into Weaviate …")

        with collection.batch.fixed_size(batch_size=100) as batch:
            for obj in tqdm(data_objects, desc="Indexing", disable=not verbose):
                batch.add_object(
                    properties=obj.properties,
                    vector=obj.vector,
                    uuid=obj.uuid,
                )

        count = collection.aggregate.over_all().total_count
        if verbose:
            print(f"✅ Weaviate collection '{class_name}' now has {count} objects")
        return count

    finally:
        if own_client:
            client.close()


# ---------------------------------------------------------------------------
# Idempotent guard
# ---------------------------------------------------------------------------

def ensure_index_built(
    config: Dict[str, Any],
    *,
    client=None,
    embedder: SentenceTransformer | None = None,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> int:
    """
    Build the index only when the collection is empty (or if *force_rebuild*).

    Returns current object count.
    """
    own_client = False
    if client is None:
        client = connect_weaviate(config)
        own_client = True

    try:
        ensure_collection(client, config)
        class_name = config["rag"]["vector_db"].get("class_name", "FinancialDocument")
        collection = client.collections.get(class_name)
        count = collection.aggregate.over_all().total_count

        if count > 0 and not force_rebuild:
            if verbose:
                print(f"📚 Index already has {count} objects – skipping rebuild")
            return count

        if force_rebuild and count > 0:
            if verbose:
                print(f"🗑️  Dropping existing collection ({count} objects) …")
            client.collections.delete(class_name)
            ensure_collection(client, config)

        return build_index(config, client=client, embedder=embedder, verbose=verbose)

    finally:
        if own_client:
            client.close()
