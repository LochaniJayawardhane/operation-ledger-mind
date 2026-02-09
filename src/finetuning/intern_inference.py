from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import PeftModel

from src.finetuning.data import SYSTEM_PROMPT, build_user_message, resolve_finetune_paths
from src.utils.config_loader import load_config


_CACHED: Dict[Tuple[str, str], Tuple[Any, Any, Dict[str, Any]]] = {}


def _dtype_from_string(name: str) -> torch.dtype:
    v = (name or "").lower()
    if v in {"float16", "fp16"}:
        return torch.float16
    if v in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if v in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported compute_dtype: {name!r}. Use float16/bfloat16/float32.")


def _build_bnb_config(ft_cfg: Dict[str, Any]) -> Optional[BitsAndBytesConfig]:
    q = ft_cfg.get("quantization", {}) or {}
    if not q.get("enabled", True):
        return None
    compute_dtype = _dtype_from_string(str(q.get("compute_dtype", "float16")))
    return BitsAndBytesConfig(
        load_in_4bit=bool(q.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(q.get("quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(q.get("double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _load_model_and_tokenizer(config_path: str | Path, adapter_dir: str | Path | None):
    config = load_config(config_path)
    ft_cfg: Dict[str, Any] = config.get("finetuning", {}) or {}

    base_model = str(ft_cfg.get("base_model"))
    paths = resolve_finetune_paths(config)
    adapter_path = Path(adapter_dir) if adapter_dir else Path(paths.output_dir)

    cache_key = (base_model, str(adapter_path.resolve()))
    if cache_key in _CACHED:
        model, tokenizer, cfg = _CACHED[cache_key]
        return model, tokenizer, cfg

    device_map = config.get("providers", {}).get("huggingface", {}).get("device_map", "auto")
    trust_remote_code = bool(config.get("providers", {}).get("huggingface", {}).get("trust_remote_code", True))

    bnb = _build_bnb_config(ft_cfg)
    compute_dtype = _dtype_from_string(str(ft_cfg.get("quantization", {}).get("compute_dtype", "float16")))

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), use_fast=True, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        quantization_config=bnb,
        torch_dtype=compute_dtype,
    )

    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    _CACHED[cache_key] = (model, tokenizer, config)
    return model, tokenizer, config


@torch.no_grad()
def query_intern(
    question: str,
    *,
    chunk_text: Optional[str] = None,
    config_path: str | Path = "config/config.yaml",
    adapter_dir: str | Path | None = None,
) -> str:
    """
    Load base model + saved LoRA adapters (4-bit if configured) and generate an answer.
    """
    model, tokenizer, config = _load_model_and_tokenizer(config_path, adapter_dir)
    ft_cfg: Dict[str, Any] = config.get("finetuning", {}) or {}
    inf_cfg: Dict[str, Any] = ft_cfg.get("inference", {}) or {}

    user_msg = build_user_message(question, chunk_text)
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

    new_tokens = gen[0, input_ids.shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text

