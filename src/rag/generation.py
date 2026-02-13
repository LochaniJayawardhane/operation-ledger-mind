"""
Answer Generation Module – Multiple Generator Backends

Provides a unified interface for generating answers from retrieved context
using different LLM backends:

  - ``openai``           – OpenAI API via the existing LLMClient
  - ``intern_finetuned`` – Part 2 fine-tuned Llama-3 with LoRA adapters
  - ``intern_base``      – Same Llama-3 base model *without* LoRA adapters

This lets us compare how much fine-tuning impacts contextual understanding
when given the exact same retrieved chunks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Prompt template (shared across all generators)
# ---------------------------------------------------------------------------

_RAG_PROMPT_TEMPLATE = """You are a senior financial analyst. Answer the question using ONLY the provided context from the company's annual report.

Rules:
- Use ONLY the information in the context below.
- If the context does not contain the answer, say: "The information is not available in the provided context."
- Be precise, factual, and concise (2-6 sentences).
- Include specific numbers, dates, or entity names from the context when relevant.

Context:
{context}

Question: {question}

Answer:"""


def build_rag_prompt(question: str, context: str) -> str:
    """Format the shared RAG prompt."""
    return _RAG_PROMPT_TEMPLATE.format(context=context, question=question)


# ---------------------------------------------------------------------------
# OpenAI generator
# ---------------------------------------------------------------------------

def generate_openai(
    question: str,
    context: str,
    config: Dict[str, Any],
) -> str:
    """Generate an answer via the OpenAI API."""
    llm_cfg = config["rag"]["inference"]["answer_llm"]
    client = LLMClient(llm_cfg)
    prompt = build_rag_prompt(question, context)
    return client.generate(prompt)


# ---------------------------------------------------------------------------
# Intern fine-tuned generator
# ---------------------------------------------------------------------------

def generate_intern_finetuned(
    question: str,
    context: str,
    config: Dict[str, Any],
    *,
    config_path: str | Path = "config/config.yaml",
) -> str:
    """Generate an answer using the Part 2 fine-tuned Intern (LoRA adapters)."""
    from src.finetuning.intern_inference import query_intern

    adapter_dir = config.get("finetuning", {}).get("output_dir", "models/intern_adapter")

    return query_intern(
        question,
        chunk_text=context,
        config_path=config_path,
        adapter_dir=adapter_dir,
    )


# ---------------------------------------------------------------------------
# Base model generator (no LoRA – control group)
# ---------------------------------------------------------------------------

def generate_intern_base(
    question: str,
    context: str,
    config: Dict[str, Any],
    *,
    config_path: str | Path = "config/config.yaml",
) -> str:
    """
    Generate an answer using the base Llama-3 model *without* LoRA adapters.

    Uses the same prompt template and chat format as the fine-tuned version
    so that the comparison isolates the effect of fine-tuning.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from src.finetuning.data import SYSTEM_PROMPT, build_user_message
    from src.utils.config_loader import load_config

    cfg = load_config(config_path) if isinstance(config_path, (str, Path)) else config
    ft_cfg = cfg.get("finetuning", {}) or {}
    inf_cfg = ft_cfg.get("inference", {}) or {}
    q_cfg = ft_cfg.get("quantization", {}) or {}

    base_model_name = str(ft_cfg.get("base_model", "meta-llama/Meta-Llama-3-8B-Instruct"))
    device_map = cfg.get("providers", {}).get("huggingface", {}).get("device_map", "auto")
    trust = bool(cfg.get("providers", {}).get("huggingface", {}).get("trust_remote_code", True))

    # Quantisation config (same as Intern so the comparison is fair)
    compute_dtype_str = str(q_cfg.get("compute_dtype", "float16")).lower()
    compute_dtype = torch.float16 if "16" in compute_dtype_str and "b" not in compute_dtype_str else torch.bfloat16

    bnb_config = None
    if q_cfg.get("enabled", True):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=bool(q_cfg.get("load_in_4bit", True)),
            bnb_4bit_quant_type=str(q_cfg.get("quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(q_cfg.get("double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, trust_remote_code=trust)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        trust_remote_code=trust,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
    )
    model.eval()

    # Build prompt with same chat template
    user_msg = build_user_message(question, context)
    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    max_new_tokens = int(inf_cfg.get("max_new_tokens", 512))
    temperature = float(inf_cfg.get("temperature", 0.2))
    top_p = float(inf_cfg.get("top_p", 0.95))
    do_sample = temperature > 0

    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = gen[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

GENERATORS = {
    "openai":           generate_openai,
    "intern_finetuned": generate_intern_finetuned,
    "intern_base":      generate_intern_base,
}


def generate_answer(
    question: str,
    context: str,
    config: Dict[str, Any],
    *,
    mode: str | None = None,
    config_path: str | Path = "config/config.yaml",
) -> str:
    """
    Route to the correct generator based on *mode* (or config default).

    Parameters
    ----------
    mode : str | None
        One of ``"openai"``, ``"intern_finetuned"``, ``"intern_base"``.
        Falls back to ``config['rag']['inference']['generator_mode']``.
    """
    mode = mode or config["rag"]["inference"].get("generator_mode", "openai")
    gen_fn = GENERATORS.get(mode)
    if gen_fn is None:
        raise ValueError(f"Unknown generator_mode: {mode!r}. Choose from {list(GENERATORS)}")

    # The intern generators need config_path; openai doesn't
    if mode.startswith("intern"):
        return gen_fn(question, context, config, config_path=config_path)
    return gen_fn(question, context, config)
