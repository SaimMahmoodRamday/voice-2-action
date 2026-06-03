# ML — Fine-tuning Voice2Action

QLoRA fine-tuning of Qwen2.5-3B-Instruct for Roman Urdu / Urdu / mixed-language task extraction.

## Setup

```bash
cd ml
pip install -r requirements.txt
```

A CUDA GPU is required for QLoRA (bitsandbytes 4-bit).

## Workflow

```bash
# 1. Convert raw examples into chat format
python scripts/prepare_dataset.py --input data/examples.jsonl --output data/train.jsonl

# 2. Train
python scripts/train_qlora.py --config configs/qlora.yaml
```

Adapter weights land in `outputs/qwen2.5-3b-voice2action/`.

## Dataset

`data/examples.jsonl` — one JSON object per line:

```json
{"input": "Roman Urdu transcript", "output": {"tasks": [...], "deadline": ..., "people": [...], "meetings": [...], "priority": ...}}
```

Add more examples here. The 10 starter examples are only enough to validate the pipeline runs; collect a real dataset before relying on the output.

## Serving the fine-tuned model

Two options for plugging the adapter back into the backend:

1. Merge LoRA into the base, export to GGUF, register with Ollama, point `OLLAMA_MODEL` at it.
2. Load adapter directly via transformers and swap `app/agent/llm.py` to call it locally.
