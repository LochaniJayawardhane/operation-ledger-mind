from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from src.finetuning.data import resolve_finetune_paths, load_sft_datasets
from src.utils.config_loader import load_config


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


def train_intern(config_path: str | Path) -> Path:
    """
    Run QLoRA fine-tuning using TRL SFTTrainer and save LoRA adapters to output_dir.
    """
    config = load_config(config_path)
    ft_cfg: Dict[str, Any] = config.get("finetuning", {}) or {}
    tr_cfg: Dict[str, Any] = ft_cfg.get("training", {}) or {}

    paths = resolve_finetune_paths(config)
    output_dir = Path(paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = str(ft_cfg.get("base_model"))
    device_map = config.get("providers", {}).get("huggingface", {}).get("device_map", "auto")
    trust_remote_code = bool(config.get("providers", {}).get("huggingface", {}).get("trust_remote_code", True))

    bnb_config = _build_bnb_config(ft_cfg)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_seq_length = int(tr_cfg.get("max_seq_length", 2048))
    ds = load_sft_datasets(
        train_file=paths.train_file,
        eval_file=paths.eval_file,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        torch_dtype=_dtype_from_string(str(ft_cfg.get("quantization", {}).get("compute_dtype", "float16"))),
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    lora_cfg: Dict[str, Any] = ft_cfg.get("lora", {}) or {}
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        bias=str(lora_cfg.get("bias", "none")),
        task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
    )
    model = get_peft_model(model, peft_config)

    # Ensure >= min_steps regardless of dataset size/epochs.
    min_steps = int(tr_cfg.get("min_steps", 100))
    max_steps = max(min_steps, int(tr_cfg.get("max_steps", min_steps)))

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        num_train_epochs=float(tr_cfg.get("num_epochs", 1)),
        per_device_train_batch_size=int(tr_cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(tr_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(tr_cfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(tr_cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(tr_cfg.get("logging_steps", 10)),
        save_steps=int(tr_cfg.get("save_steps", 50)),
        max_seq_length=max_seq_length,
        packing=False,
        bf16=_dtype_from_string(str(ft_cfg.get("quantization", {}).get("compute_dtype", "float16"))) == torch.bfloat16,
        fp16=_dtype_from_string(str(ft_cfg.get("quantization", {}).get("compute_dtype", "float16"))) == torch.float16,
        gradient_checkpointing=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval"),
        tokenizer=tokenizer,
        dataset_text_field="text",
    )

    trainer.train()

    # Save adapter + tokenizer for inference.
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune The Intern (QLoRA + TRL SFTTrainer)")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    out_dir = train_intern(args.config)
    print(f"Saved adapters + tokenizer to: {out_dir}")


if __name__ == "__main__":
    main()

