"""Stratified, position-spread train/test split for examples.jsonl.

The dataset is ordered (original examples first, augmentation appended after), so
categories cluster by file position. To avoid a skewed test set we:
  1. bucket each row by a "kind" signature (task-count, has-deadline, has-meeting,
     language), so every kind is represented in test proportionally, and
  2. within each bucket pick test rows EVENLY SPACED across their original file
     positions (not contiguous) — so test draws from the top, middle and bottom.

Outputs (raw input/output rows):
  ml/data/examples_train.jsonl   -> goes through prepare_dataset.py -> train.jsonl
  ml/data/test.jsonl             -> held out, used only for evaluation
"""
import argparse
import json
import re
from collections import defaultdict

ROMAN = re.compile(r"\b(hai|hain|karna|karni|karne|karo|ko|kal|yaar|yar|bhej|"
                   r"nahi|lena|leni|laana|le ana|wala|jana|ana|mujhe|suno|sun)\b", re.I)


def lang_of(text: str) -> str:
    ascii_only = all(ord(c) < 128 for c in text)
    return "eng" if (ascii_only and not ROMAN.search(text)) else "rom"


def signature(row: dict):
    o = row["output"]
    nt = len(o.get("tasks") or [])
    tb = "0" if nt == 0 else "1" if nt == 1 else "2+"
    return (tb, bool(o.get("deadline")), bool(o.get("meetings")), lang_of(row["input"]))


def evenly_spaced(indices, k):
    """Pick k items spread across the (position-ordered) list."""
    if k <= 0:
        return []
    if k >= len(indices):
        return list(indices)
    n = len(indices)
    return [indices[round(i * (n - 1) / (k - 1))] for i in range(k)] if k > 1 else [indices[n // 2]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="ml/data/examples.jsonl")
    ap.add_argument("--train_out", default="ml/data/examples_train.jsonl")
    ap.add_argument("--test_out", default="ml/data/test.jsonl")
    ap.add_argument("--frac", type=float, default=0.10)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]

    # group original indices by signature (preserves file order within group)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[signature(r)].append(i)

    test_idx = set()
    for sig, idxs in groups.items():
        k = round(len(idxs) * args.frac)
        if len(idxs) >= 4 and k == 0:   # ensure non-tiny groups contribute one
            k = 1
        for j in evenly_spaced(idxs, k):
            test_idx.add(j)

    train = [r for i, r in enumerate(rows) if i not in test_idx]
    test = [r for i, r in enumerate(rows) if i in test_idx]

    for path, data in [(args.train_out, train), (args.test_out, test)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── report ────────────────────────────────────────────────────────────────
    print(f"total {len(rows)} -> train {len(train)} | test {len(test)} "
          f"({100*len(test)//len(rows)}%)")
    ti = sorted(test_idx)
    print(f"test index spread: min {ti[0]}, q1 {ti[len(ti)//4]}, "
          f"median {ti[len(ti)//2]}, q3 {ti[3*len(ti)//4]}, max {ti[-1]} (of {len(rows)})")

    def dist(data):
        d = defaultdict(int)
        for r in data:
            o = r["output"]
            nt = len(o.get("tasks") or [])
            d["tasks0" if nt == 0 else "tasks1" if nt == 1 else "tasks2+"] += 1
            if o.get("deadline"): d["deadline"] += 1
            if o.get("meetings"): d["meeting"] += 1
            d["eng" if lang_of(r["input"]) == "eng" else "rom"] += 1
        return {k: d[k] for k in sorted(d)}

    print("train dist:", dist(train))
    print("test  dist:", dist(test))


if __name__ == "__main__":
    main()
