"""Convert raw examples.jsonl into instruction-tuning format for SFTTrainer."""
import argparse
import json
from pathlib import Path

SYSTEM = (
    "You extract structured task information from informal voice-note transcripts "
    "in Roman Urdu, Urdu, or mixed Urdu-English. Return ONLY a JSON object with keys "
    "tasks, deadline, people, meetings."
)


def to_chat(example: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": json.dumps(example["output"], ensure_ascii=False)},
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="ml/data/examples.jsonl")
    ap.add_argument("--output", default="ml/data/train.jsonl")
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fout.write(json.dumps(to_chat(json.loads(line)), ensure_ascii=False) + "\n")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
