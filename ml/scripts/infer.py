"""Inference test for the fine-tuned Voice2Action LoRA adapter.

Usage:
    python ml/scripts/infer.py
    python ml/scripts/infer.py --adapter ml/outputs/qwen2.5-3b-voice2action
    python ml/scripts/infer.py --prompt "Ahmed ko call karna hai kal tak"
"""
import argparse
import json
import textwrap

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── System prompt (must match training) ─────────────────────────────────────
SYSTEM = (
    "You extract structured task information from informal voice-note transcripts "
    "in Roman Urdu, Urdu, or mixed Urdu-English. Return ONLY a JSON object with keys "
    "tasks, deadline, people, meetings."
)

# ── Test cases ───────────────────────────────────────────────────────────────
# Mix of: Roman Urdu, English, mixed, single task, multi-task, meetings
TEST_CASES = [
    # Basic Roman Urdu
    {
        "input": "Ahmed ko call karna hai kal tak",
        "expect": {"tasks": ["Call Ahmed"], "deadline": "tomorrow", "people": ["Ahmed"]},
    },
    # Multi-task + deadline
    {
        "input": "report complete karni hai aur presentation finalize karni hai by EOD",
        "expect": {"tasks": ["Complete the report", "Finalize presentation"], "deadline": "end of day"},
    },
    # Meeting extraction
    {
        "input": "kal subah Sara ke sath meeting hai",
        "expect": {"meetings": ["Meeting with Sara"], "people": ["Sara"]},
    },
    # Pure English
    {
        "input": "Please send the proposal to Kamran by end of day",
        "expect": {"tasks": ["Send proposal to Kamran"], "deadline": "end of day", "people": ["Kamran"]},
    },
    # Mixed Urdu-English, multiple tasks
    {
        "input": "urgent hai PR merge karna hai aur deployment karna hai",
        "expect": {"tasks": ["Merge the PR", "Do deployment"]},
    },
    # Casual single task
    {
        "input": "koi jaldi nahi, bas library books return karne hain",
        "expect": {"tasks": ["Return the library books"]},
    },
    # People + meeting + task
    {
        "input": "jaldi se weekly meeting hai aur Sana se approval lena hai",
        "expect": {"meetings": ["Weekly meeting"], "people": ["Sana"], "tasks": ["Get approval from Sana"]},
    },
    # Novel / unseen style
    {
        "input": "boss ne kaha hai budget finalize karo aur Rizwan ko bhi inform karo",
        "expect": {"tasks": ["Finalize budget", "Inform Rizwan"], "people": ["Rizwan", "boss"]},
    },
]


def build_messages(user_input: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_input},
    ]


def run_inference(model, tokenizer, user_input: str, max_new_tokens: int = 256) -> str:
    messages = build_messages(user_input)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — deterministic for evaluation
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Strip the input prompt tokens from the output
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_json_safe(text: str) -> dict | None:
    """Try to extract a JSON object from model output."""
    import re
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def check_result(output_dict: dict | None, expected: dict) -> tuple[bool, list[str]]:
    """Spot-check key fields from expected against actual output."""
    if output_dict is None:
        return False, ["Output is not valid JSON"]

    issues = []
    required_keys = {"tasks", "deadline", "people", "meetings"}
    missing_keys = required_keys - output_dict.keys()
    if missing_keys:
        issues.append(f"Missing keys: {missing_keys}")

    for field, exp_val in expected.items():
        act_val = output_dict.get(field)
        if isinstance(exp_val, list):
            # Check that expected items appear (case-insensitive substring match)
            for item in exp_val:
                if not any(item.lower() in str(a).lower() for a in (act_val or [])):
                    issues.append(f"  [{field}] expected to contain '{item}', got {act_val}")
        elif isinstance(exp_val, str):
            if act_val is None or exp_val.lower() not in str(act_val).lower():
                issues.append(f"  [{field}] expected '{exp_val}', got '{act_val}'")

    passed = len(issues) == 0
    return passed, issues


def load_model(adapter_path: str):
    print(f"\nLoading base model from adapter config: Qwen/Qwen2.5-3B-Instruct")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print(f"Model loaded on: {next(model.parameters()).device}\n")
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--adapter",
        default="ml/outputs/qwen2.5-3b-voice2action",
        help="Path to the saved LoRA adapter directory",
    )
    ap.add_argument(
        "--prompt",
        default=None,
        help="Run a single custom prompt instead of the test suite",
    )
    args = ap.parse_args()

    model, tokenizer = load_model(args.adapter)

    # ── Single prompt mode ───────────────────────────────────────────────────
    if args.prompt:
        print(f"Input : {args.prompt}")
        raw = run_inference(model, tokenizer, args.prompt)
        print(f"Output: {raw}")
        parsed = parse_json_safe(raw)
        if parsed:
            print(f"Parsed: {json.dumps(parsed, ensure_ascii=False, indent=2)}")
        return

    # ── Full test suite ──────────────────────────────────────────────────────
    sep = "─" * 70
    passed_count = 0

    for i, case in enumerate(TEST_CASES, 1):
        print(sep)
        print(f"Test {i}/{len(TEST_CASES)}")
        print(f"Input   : {case['input']}")

        raw = run_inference(model, tokenizer, case["input"])
        parsed = parse_json_safe(raw)

        print(f"Raw out : {raw}")
        if parsed:
            print(f"Parsed  : {json.dumps(parsed, ensure_ascii=False)}")

        ok, issues = check_result(parsed, case["expect"])
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"Status  : {status}")
        if issues:
            for iss in issues:
                print(f"          {iss}")

        if ok:
            passed_count += 1

    print(sep)
    print(f"\nResult: {passed_count}/{len(TEST_CASES)} tests passed")
    if passed_count == len(TEST_CASES):
        print("All tests passed. Model is ready for integration.")
    elif passed_count >= len(TEST_CASES) * 0.75:
        print("Good quality. Minor issues may be acceptable.")
    else:
        print("Several tests failed. Consider re-checking training or data quality.")


if __name__ == "__main__":
    main()
