"""Merge the fine-tuned LoRA adapter into the base model.

Produces a standalone full-precision (fp16) model that can be converted to GGUF
and served via Ollama. Runs on CPU to avoid VRAM limits (the 3B base in fp16 is
~6 GB, which won't fit alongside the adapter on a 6 GB card).

Usage:
    python ml/scripts/merge_lora.py
    python ml/scripts/merge_lora.py --adapter ml/outputs/qwen2.5-3b-voice2action \
        --out ml/outputs/qwen2.5-3b-voice2action-merged
"""
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--adapter", default="ml/outputs/qwen2.5-3b-voice2action")
    ap.add_argument("--out", default="ml/outputs/qwen2.5-3b-voice2action-merged")
    args = ap.parse_args()

    print(f"Loading base model: {args.base} (fp16, CPU)")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print(f"Attaching adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("Merging adapter into base weights (merge_and_unload)...")
    model = model.merge_and_unload()

    os.makedirs(args.out, exist_ok=True)
    print(f"Saving merged model to: {args.out}")
    model.save_pretrained(args.out, safe_serialization=True)

    # Save the tokenizer from the adapter dir so any added tokens are preserved.
    print("Saving tokenizer (from adapter dir, preserves added tokens)")
    tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tok.save_pretrained(args.out)

    print("Done. Merged model written to", args.out)


if __name__ == "__main__":
    main()
