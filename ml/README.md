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
python ml/scripts/prepare_dataset.py --input ml/data/examples.jsonl --output ml/data/train.jsonl

# 2. Train
python ml/scripts/train_qlora.py --config ml/configs/qlora.yaml
```

Adapter weights land in `outputs/qwen2.5-3b-voice2action/`.

## Dataset

`data/examples.jsonl` — one JSON object per line:

```json
{"input": "Roman Urdu transcript", "output": {"tasks": [...], "deadline": ..., "people": [...], "meetings": [...]}}
```

Add more examples here. The 10 starter examples are only enough to validate the pipeline runs; collect a real dataset before relying on the output.

## Serving the fine-tuned model

The Ollama serving path (merge LoRA → GGUF → `ollama create`) is implemented.
See **[`serving/README.md`](serving/README.md)** for the full pipeline:

```bash
# Merge adapter into base, then convert straight to Q8_0 GGUF
python ml/scripts/merge_lora.py
python tools/llama.cpp/convert_hf_to_gguf.py \
    ml/outputs/qwen2.5-3b-voice2action-merged --outtype q8_0 \
    --outfile ml/serving/voice2action-q8_0.gguf

# Register with Ollama
cd ml/serving && ollama create voice2action -f Modelfile
```

The backend then uses it via `OLLAMA_MODEL=voice2action`.

> Alternative (not used here): load the adapter directly via transformers and
> swap `app/agent/llm.py` to call it locally instead of going through Ollama.
