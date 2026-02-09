# Fine-tune “The Intern” (QLoRA + TRL)

This repo fine-tunes a Llama-3-Instruct model on your generated JSONL Q/A dataset and saves **LoRA adapters** (not full weights).

## Prereqs

- **GPU recommended** (Colab T4 works). QLoRA uses `bitsandbytes`, which is typically easiest on Linux/Colab.
- **Hugging Face auth** (Llama-3 weights are gated):
  - `huggingface-cli login`

## Data

- Train: `data/output/train.jsonl`
- Eval: `data/output/golden_test_set.jsonl`

These paths are also set in `config/config.yaml` under `finetuning.data`.

## Run (script)

From the project root:

```bash
python -m src.finetuning.train_intern --config config/config.yaml
```

Adapters + tokenizer are saved to `finetuning.output_dir` (default: `models/intern_adapter`).

## Run (notebook)

Open `notebooks/02_finetuning_intern.ipynb` and run all cells. The notebook:

- Loads config
- Loads the JSONL dataset and formats it with the Llama-3 chat template
- Runs TRL `SFTTrainer` for **≥ 100 optimizer steps**
- Saves adapters
- Smoke-tests inference via `query_intern(...)`

## Inference

```python
from src.finetuning.intern_inference import query_intern

query_intern("What valuation model is used for market-based awards?")
```

