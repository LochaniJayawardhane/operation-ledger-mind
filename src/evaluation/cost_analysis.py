"""
Cost Analysis Module – Bonus (Part 4)

Estimates monthly cloud cost for two strategies:
    1. **Intern** – Self-hosted fine-tuned model on AWS GPU instances.
    2. **Librarian** – OpenAI API (gpt-4o-mini) + lightweight compute.

Assumptions are driven by ``config['cost_analysis']``.
"""

from __future__ import annotations

from typing import Any, Dict


# ---------------------------------------------------------------------------
# OpenAI pricing (as of early 2025 – gpt-4o-mini)
# ---------------------------------------------------------------------------
_OPENAI_PRICING = {
    "gpt-4o-mini": {
        "input_per_1m_tokens": 0.15,   # USD per 1M input tokens
        "output_per_1m_tokens": 0.60,  # USD per 1M output tokens
    },
    "gpt-4o": {
        "input_per_1m_tokens": 2.50,
        "output_per_1m_tokens": 10.00,
    },
    "o3-mini": {
        "input_per_1m_tokens": 1.10,
        "output_per_1m_tokens": 4.40,
    },
}

# Embedding model pricing (sentence-transformers is free/self-hosted, but
# if using OpenAI embeddings in the future)
_EMBEDDING_PRICING = {
    "self_hosted": 0.0,  # sentence-transformers runs on the same instance
}

# Average tokens per query (estimated)
_AVG_INPUT_TOKENS_PER_QUERY = 1500   # prompt + retrieved context
_AVG_OUTPUT_TOKENS_PER_QUERY = 200   # generated answer


def estimate_monthly_cost(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Estimate monthly cloud cost for both the Intern and Librarian strategies.

    Reads from ``config['cost_analysis']`` for user/query volumes and
    infrastructure pricing.

    Returns
    -------
    dict with keys:
        ``queries_per_month``  – total monthly query volume
        ``intern``             – cost breakdown for the self-hosted Intern
        ``librarian``          – cost breakdown for the API-based Librarian
        ``summary``            – human-readable summary strings
    """
    ca = config.get("cost_analysis", {})

    users_per_day = int(ca.get("users_per_day", 500))
    queries_per_user = int(ca.get("queries_per_user", 10))
    days_per_month = int(ca.get("days_per_month", 30))
    queries_per_month = users_per_day * queries_per_user * days_per_month

    utilization = float(ca.get("assumptions", {}).get("utilization_percent", 60)) / 100.0

    # ------------------------------------------------------------------
    # Strategy 1: The Intern (self-hosted on AWS GPU)
    # ------------------------------------------------------------------
    intern_infra = ca.get("infrastructure", {}).get("intern", {})
    intern_instance = intern_infra.get("instance_type", "g4dn.xlarge")
    intern_hourly = float(intern_infra.get("hourly_cost_usd", 0.526))

    # Must run 24/7 to serve queries (or scale down during off-peak)
    hours_per_month = days_per_month * 24
    intern_compute = intern_hourly * hours_per_month

    # Effective cost accounting for utilization
    intern_effective = intern_compute  # instance runs regardless of utilization

    # ------------------------------------------------------------------
    # Strategy 2: The Librarian (OpenAI API + lightweight compute)
    # ------------------------------------------------------------------
    rag_llm_model = (
        config.get("rag", {})
        .get("inference", {})
        .get("answer_llm", {})
        .get("model", "gpt-4o-mini")
    )
    pricing = _OPENAI_PRICING.get(rag_llm_model, _OPENAI_PRICING["gpt-4o-mini"])

    total_input_tokens = queries_per_month * _AVG_INPUT_TOKENS_PER_QUERY
    total_output_tokens = queries_per_month * _AVG_OUTPUT_TOKENS_PER_QUERY

    api_input_cost = (total_input_tokens / 1_000_000) * pricing["input_per_1m_tokens"]
    api_output_cost = (total_output_tokens / 1_000_000) * pricing["output_per_1m_tokens"]
    api_total = api_input_cost + api_output_cost

    # Weaviate + embedding inference needs a lighter instance (e.g., c5.xlarge)
    # Estimated at ~$0.17/hr for a c5.xlarge (or free if using Weaviate Cloud)
    librarian_compute_hourly = 0.17
    librarian_compute = librarian_compute_hourly * hours_per_month

    librarian_total = api_total + librarian_compute

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    return {
        "queries_per_month": queries_per_month,
        "intern": {
            "instance_type": intern_instance,
            "hourly_cost_usd": intern_hourly,
            "hours_per_month": hours_per_month,
            "compute_cost_usd": round(intern_compute, 2),
            "api_cost_usd": 0.0,
            "total_monthly_usd": round(intern_effective, 2),
        },
        "librarian": {
            "llm_model": rag_llm_model,
            "input_tokens_total": total_input_tokens,
            "output_tokens_total": total_output_tokens,
            "api_input_cost_usd": round(api_input_cost, 2),
            "api_output_cost_usd": round(api_output_cost, 2),
            "api_total_usd": round(api_total, 2),
            "compute_instance": "c5.xlarge (Weaviate + embeddings)",
            "compute_cost_usd": round(librarian_compute, 2),
            "total_monthly_usd": round(librarian_total, 2),
        },
        "summary": {
            "intern_monthly": f"${intern_effective:,.2f}",
            "librarian_monthly": f"${librarian_total:,.2f}",
            "cheaper": "Librarian" if librarian_total < intern_effective else "Intern",
            "savings_pct": round(
                abs(intern_effective - librarian_total)
                / max(intern_effective, librarian_total)
                * 100,
                1,
            ),
            "disclaimer": ca.get("assumptions", {}).get(
                "disclaimer", "Estimated costs based on public AWS/OpenAI pricing."
            ),
        },
    }
