"""
CLI runner for ``query_librarian``.

Usage::

    uv run python -m src.rag.cli "What was the total revenue in FY2024?"
    uv run python -m src.rag.cli --mode intern_finetuned "What does Form 10-K disclose?"
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="The Librarian – Advanced Hybrid RAG CLI",
    )
    parser.add_argument("question", type=str, help="Question to ask the Librarian")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["openai", "intern_finetuned", "intern_base"],
        help="Generator mode (default: from config)",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force-rebuild the Weaviate index before querying",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output raw JSON instead of formatted text",
    )
    args = parser.parse_args()

    from src.rag.librarian_inference import query_librarian

    result = query_librarian(
        args.question,
        config_path=args.config,
        generator_mode=args.mode,
        rebuild_index=args.rebuild,
        verbose=True,
    )

    if args.json_output:
        print(json.dumps(result, indent=2, default=str))
        return

    # Pretty-print
    print("\n" + "=" * 70)
    print(f"📖 Question: {args.question}")
    print(f"🤖 Generator: {result['generator_mode']}")
    print("=" * 70)
    print(f"\n{result['answer']}\n")
    print("-" * 70)
    print("📚 Top Sources:")
    for i, src in enumerate(result["sources"], 1):
        scores = src["scores"]
        print(
            f"  [{i}] chunk {src['chunk_id']} "
            f"(pp. {src['page_start']}–{src['page_end']}) "
            f"rerank={scores.get('rerank', 'N/A'):.3f}"
            if scores.get("rerank") is not None
            else f"  [{i}] chunk {src['chunk_id']}"
        )
        print(f"      {src['preview'][:120]}…" if len(src["preview"]) > 120 else f"      {src['preview']}")
    print("-" * 70)
    stats = result["stats"]
    print(
        f"⏱  Retrieval: {stats['retrieval_ms']}ms | "
        f"Generation: {stats['generation_ms']}ms | "
        f"Total: {stats['total_ms']}ms"
    )


if __name__ == "__main__":
    main()
