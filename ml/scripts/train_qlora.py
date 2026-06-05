"""QLoRA fine-tuning entry point. Run after prepare_dataset.py.

Expects train.jsonl rows in conversational format:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
"""
import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ml/configs/qlora.yaml")
    ap.add_argument("--train_file", default="ml/data/train.jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)

    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["bnb_4bit_compute_dtype"]),
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )

    peft_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset("json", data_files=args.train_file, split="train")

    def formatting_func(examples):
        """Apply the model's chat template to each messages sequence.

        SFTTrainer receives batched examples, so examples["messages"] is a
        list of message lists.  We apply the tokenizer's built-in chat
        template (e.g. Qwen2.5's <|im_start|> format) so training tokens
        match inference exactly.
        """
        return [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            for msgs in examples["messages"]
        ]

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    sft_cfg = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        max_seq_length=cfg["max_seq_length"],
        bf16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=peft_cfg,
        formatting_func=formatting_func,
        args=sft_cfg,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"Saved adapter to {cfg['output_dir']}")


if __name__ == "__main__":
    main()

