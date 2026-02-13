"""
Weaviate Client & Schema Module

Handles connecting to Weaviate (embedded / cloud / local) and
creating the FinancialDocument collection with custom vectors.
"""

from __future__ import annotations

import uuid as _uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import weaviate
from weaviate.classes.config import Configure, Property, DataType


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def connect_weaviate(config: Dict[str, Any]) -> weaviate.WeaviateClient:
    """
    Open a Weaviate client based on ``config['rag']['vector_db']``.

    Supported modes:
      - ``embedded``  – spins up an embedded Weaviate process (default)
      - ``cloud``     – connects to Weaviate Cloud (WCS)
      - ``local``     – connects to a local Docker instance
    """
    vdb = config["rag"]["vector_db"]
    mode = vdb.get("mode", "embedded").lower()

    if mode == "embedded":
        embedded_cfg = vdb.get("embedded", {})
        persistence_dir = str(
            Path(embedded_cfg.get("persistence_dir", ".weaviate")).resolve()
        )
        client = weaviate.connect_to_embedded(
            persistence_data_path=persistence_dir,
        )

    elif mode == "cloud":
        import os
        cluster_url = os.environ.get("WEAVIATE_URL", "")
        api_key = os.environ.get("WEAVIATE_API_KEY", "")
        if not cluster_url:
            raise RuntimeError("WEAVIATE_URL env var required for cloud mode")
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=cluster_url,
            auth_credentials=weaviate.auth.AuthApiKey(api_key) if api_key else None,
        )

    elif mode == "local":
        host = vdb.get("host", "localhost")
        port = int(vdb.get("port", 8080))
        grpc_port = int(vdb.get("grpc_port", 50051))
        client = weaviate.connect_to_local(
            host=host, port=port, grpc_port=grpc_port,
        )

    else:
        raise ValueError(f"Unknown rag.vector_db.mode: {mode!r}")

    return client


# ---------------------------------------------------------------------------
# Schema / collection helpers
# ---------------------------------------------------------------------------

_PROPERTIES = [
    Property(name="content",        data_type=DataType.TEXT),
    Property(name="chunk_id",       data_type=DataType.INT),
    Property(name="doc_name",       data_type=DataType.TEXT),
    Property(name="report_year",    data_type=DataType.INT),
    Property(name="source_section", data_type=DataType.TEXT),
    Property(name="page_start",     data_type=DataType.INT),
    Property(name="page_end",       data_type=DataType.INT),
    Property(name="start_char",     data_type=DataType.INT),
    Property(name="end_char",       data_type=DataType.INT),
]


def ensure_collection(
    client: weaviate.WeaviateClient,
    config: Dict[str, Any],
) -> None:
    """
    Create the target collection if it does not already exist.

    Uses ``Configure.Vectorizer.none()`` so we supply our own
    SentenceTransformer vectors at insert time.
    """
    class_name = config["rag"]["vector_db"].get("class_name", "FinancialDocument")

    if client.collections.exists(class_name):
        return

    client.collections.create(
        name=class_name,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=_PROPERTIES,
    )


# ---------------------------------------------------------------------------
# Deterministic UUID helper
# ---------------------------------------------------------------------------

_NAMESPACE_UUID = _uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def deterministic_uuid(doc_name: str, chunk_id: int) -> str:
    """UUID5 from ``doc_name:chunk_id`` for idempotent upserts."""
    return str(_uuid.uuid5(_NAMESPACE_UUID, f"{doc_name}:{chunk_id}"))
