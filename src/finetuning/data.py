from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset


SYSTEM_PROMPT = (
    "You are Uber's 2024 strategy intern. You write crisp, accurate, executive-ready answers.\n"
    "Rules:\n"
    "- Use ONLY the provided context.\n"
    "- If the context does not contain the answer, say: \"The information is not available in the provided text.\"\n"
    "- Be concise (2-6 sentences) and avoid speculation.\n"
)


@dataclass(frozen=True)
class FinetunePaths:
    train_file: Path
    eval_file: Optional[Path]
    output_dir: Path


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def resolve_finetune_paths(config: Dict[str, Any]) -> FinetunePaths:
    """
    Resolve train/eval/output paths from config with sensible fallbacks.
    """
    data_dir = _as_path(config["environment"]["paths"]["data_dir"])
    models_dir = _as_path(config["environment"]["paths"]["models_dir"])

    dataset_cfg = config.get("data_factory", {}).get("dataset", {})
    default_train = data_dir / "output" / str(dataset_cfg.get("train_file", "train.jsonl"))
    default_eval = data_dir / "output" / str(dataset_cfg.get("test_file", "golden_test_set.jsonl"))

    ft_cfg = config.get("finetuning", {})
    ft_data = ft_cfg.get("data", {})
    train_file = _as_path(ft_data.get("train_file", default_train))
    eval_file = _as_path(ft_data.get("eval_file", default_eval)) if ft_data.get("eval_file", default_eval) else None
    output_dir = _as_path(ft_cfg.get("output_dir", models_dir / "intern_adapter"))

    return FinetunePaths(train_file=train_file, eval_file=eval_file, output_dir=output_dir)


def build_user_message(question: str, chunk_text: Optional[str]) -> str:
    """
    Build the user content for chat-formatted SFT.
    """
    q = (question or "").strip()
    ctx = (chunk_text or "").strip()
    if ctx:
        return f"Question:\n{q}\n\nContext:\n{ctx}"
    return f"Question:\n{q}"


def _truncate_context_to_fit(
    tokenizer,
    *,
    question: str,
    answer: str,
    chunk_text: Optional[str],
    max_seq_length: int,
    safety_margin: int = 32,
) -> Tuple[str, Optional[str]]:
    """
    Token-budget truncation that preserves chat formatting and the full answer.
    """
    # Base length (no context)
    user_no_ctx = build_user_message(question, chunk_text=None)
    base_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_no_ctx},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
    )
    base_ids = tokenizer(base_text, add_special_tokens=False).input_ids

    remaining = max_seq_length - len(base_ids) - safety_margin
    if remaining <= 0 or not chunk_text:
        return user_no_ctx, None

    ctx_ids = tokenizer(chunk_text, add_special_tokens=False).input_ids
    if len(ctx_ids) > remaining:
        ctx_ids = ctx_ids[:remaining]
    ctx_trunc = tokenizer.decode(ctx_ids, skip_special_tokens=True).strip()
    return build_user_message(question, ctx_trunc), ctx_trunc


def format_sft_record(
    record: Dict[str, Any],
    tokenizer,
    *,
    max_seq_length: int,
) -> Dict[str, str]:
    """
    Convert a JSONL record (question/answer/chunk_text) into a single `text` field
    suitable for TRL SFTTrainer.
    """
    question = str(record.get("question", "")).strip()
    answer = str(record.get("answer", "")).strip()
    chunk_text = record.get("chunk_text")
    chunk_text = str(chunk_text) if chunk_text is not None else None

    user_msg, _ = _truncate_context_to_fit(
        tokenizer,
        question=question,
        answer=answer,
        chunk_text=chunk_text,
        max_seq_length=max_seq_length,
    )

    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
    )
    return {"text": text}


def load_sft_datasets(
    *,
    train_file: str | Path,
    eval_file: str | Path | None,
    tokenizer,
    max_seq_length: int,
) -> DatasetDict:
    """
    Load JSONL files and produce an SFT-ready DatasetDict with a single `text` column.
    """
    train_file = _as_path(train_file)
    eval_file = _as_path(eval_file) if eval_file else None

    if not train_file.exists():
        raise FileNotFoundError(f"Train JSONL not found: {train_file}")
    if eval_file and not eval_file.exists():
        eval_file = None

    data_files: Dict[str, str] = {"train": str(train_file)}
    if eval_file:
        data_files["eval"] = str(eval_file)

    raw = load_dataset("json", data_files=data_files)

    def _map_fn(examples: Dict[str, list]) -> Dict[str, list]:
        out_texts = []
        for i in range(len(examples["question"])):
            rec = {k: examples[k][i] for k in examples.keys()}
            out_texts.append(format_sft_record(rec, tokenizer, max_seq_length=max_seq_length)["text"])
        return {"text": out_texts}

    train_ds: Dataset = raw["train"].map(_map_fn, batched=True, remove_columns=raw["train"].column_names)
    if "eval" in raw:
        eval_ds: Dataset = raw["eval"].map(_map_fn, batched=True, remove_columns=raw["eval"].column_names)
        return DatasetDict(train=train_ds, eval=eval_ds)

    return DatasetDict(train=train_ds)

