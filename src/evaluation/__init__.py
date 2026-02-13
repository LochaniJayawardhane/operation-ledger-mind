"""
Evaluation Arena – Part 4: The Showdown

Metrics, LLM-as-a-Judge scoring, latency measurement, and cost analysis
for comparing The Intern (fine-tuned) vs The Librarian (RAG).
"""

from src.evaluation.metrics import compute_rouge_l, llm_judge_score, measure_latency
from src.evaluation.cost_analysis import estimate_monthly_cost

__all__ = [
    "compute_rouge_l",
    "llm_judge_score",
    "measure_latency",
    "estimate_monthly_cost",
]
