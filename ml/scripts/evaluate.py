"""Evaluate fine-tuning vs prompting on the held-out test set.

Compares three conditions on ml/data/test.jsonl (never seen in training):
  1. base  zero-shot  (qwen2.5:3b, system prompt only)
  2. base  few-shot   (qwen2.5:3b, system prompt + 5 in-context examples)
  3. fine-tuned       (voice2action)

All conditions use the SAME system prompt (the one the model was trained on),
the raw transcript as the user turn, and greedy decoding (temperature 0) for
determinism. Few-shot additionally gets 5 worked examples; fine-tuned
additionally has the trained weights. This isolates "can prompting alone match
fine-tuning?".

Metrics (lenient set matching, since task text is free-form English):
  - JSON-valid %       : output parses to an object with the 4 expected keys
  - tasks / people / meetings F1
  - deadline accuracy  : both null, or lenient string match
  - exact-match %      : all fields correct

Usage:
    python ml/scripts/evaluate.py
    python ml/scripts/evaluate.py --limit 20      # quick smoke
"""
import argparse
import json
import re

import httpx

OLLAMA = "http://localhost:11434"
SYSTEM = (
    "You extract structured task information from informal voice-note transcripts "
    "in Roman Urdu, Urdu, or mixed Urdu-English. Return ONLY a JSON object with keys "
    "tasks, deadline, people, meetings."
)
KEYS = {"tasks", "deadline", "people", "meetings"}

# Few-shot examples (from the train distribution; NOT in test). Diverse on
# purpose: compound 2-task, deadline+person, meeting, negative, household.
FEWSHOT = [
    ("report banani hai phir Ali ko bhejni hai",
     {"tasks": ["Prepare the report", "Send report to Ali"], "deadline": None,
      "people": ["Ali"], "meetings": []}),
    ("Ali ko call karna hai friday tak",
     {"tasks": ["Call Ali"], "deadline": "Friday", "people": ["Ali"], "meetings": []}),
    ("kal client ke sath meeting hai",
     {"tasks": [], "deadline": "tomorrow", "people": ["client"],
      "meetings": ["Meeting with client"]}),
    ("yaar aaj bohat garmi hai",
     {"tasks": [], "deadline": None, "people": [], "meetings": []}),
    ("doodh le ana hai",
     {"tasks": ["Buy milk"], "deadline": None, "people": [], "meetings": []}),
]


def chat(client, model, transcript, few_shot=False):
    msgs = [{"role": "system", "content": SYSTEM}]
    if few_shot:
        for inp, out in FEWSHOT:
            msgs.append({"role": "user", "content": inp})
            msgs.append({"role": "assistant", "content": json.dumps(out, ensure_ascii=False)})
    msgs.append({"role": "user", "content": transcript})
    r = client.post(f"{OLLAMA}/api/chat", json={
        "model": model, "messages": msgs, "stream": False,
        "options": {"temperature": 0},
    })
    r.raise_for_status()
    return r.json()["message"]["content"]


def parse(text):
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def lenient(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return a == b
    if a == b or a in b or b in a:
        return True
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / len(ta | tb) >= 0.5 if (ta | tb) else False


def set_f1(pred, gold):
    pred, gold = list(pred or []), list(gold or [])
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    used, matched = set(), 0
    for p in pred:
        for j, g in enumerate(gold):
            if j not in used and lenient(p, g):
                used.add(j); matched += 1; break
    prec = matched / len(pred)
    rec = matched / len(gold)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def deadline_ok(pred, gold):
    if not pred and not gold:
        return True
    if not pred or not gold:
        return False
    return lenient(pred, gold)


def score(pred, gold):
    if pred is None:
        return {"json": 0, "tasks": 0.0, "people": 0.0, "meetings": 0.0,
                "deadline": 0, "exact": 0}
    jvalid = 1 if KEYS.issubset(pred.keys()) else 0
    tf = set_f1(pred.get("tasks"), gold.get("tasks"))
    pf = set_f1(pred.get("people"), gold.get("people"))
    mf = set_f1(pred.get("meetings"), gold.get("meetings"))
    dl = 1 if deadline_ok(pred.get("deadline"), gold.get("deadline")) else 0
    exact = 1 if (tf == 1.0 and pf == 1.0 and mf == 1.0 and dl == 1) else 0
    return {"json": jvalid, "tasks": tf, "people": pf, "meetings": mf,
            "deadline": dl, "exact": exact}


def run(client, label, model, rows, few_shot=False):
    agg = {"json": 0, "tasks": 0.0, "people": 0.0, "meetings": 0.0, "deadline": 0, "exact": 0}
    details = []
    for i, r in enumerate(rows, 1):
        try:
            raw = chat(client, model, r["input"], few_shot)
        except Exception as e:
            raw = f"<error: {e}>"
        pred = parse(raw)
        s = score(pred, r["output"])
        for k in agg:
            agg[k] += s[k]
        details.append({"input": r["input"], "gold": r["output"], "raw": raw, "score": s})
        if i % 25 == 0:
            print(f"    {label}: {i}/{len(rows)}")
    n = len(rows)
    return {k: agg[k] / n for k in agg}, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="ml/data/test.jsonl")
    ap.add_argument("--base", default="qwen2.5:3b")
    ap.add_argument("--ft", default="voice2action")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="ml/eval_results.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"evaluating on {len(rows)} held-out examples\n")

    conditions = [
        ("base zero-shot", args.base, False),
        ("base few-shot", args.base, True),
        ("fine-tuned", args.ft, False),
    ]
    results, all_details = {}, {}
    with httpx.Client(timeout=180.0) as client:
        for label, model, fs in conditions:
            print(f"  running: {label} ({model})")
            summary, details = run(client, label, model, rows, fs)
            results[label] = summary
            all_details[label] = details

    # ── table ─────────────────────────────────────────────────────────────────
    cols = [("JSON%", "json"), ("Task F1", "tasks"), ("People F1", "people"),
            ("Meet F1", "meetings"), ("Deadline", "deadline"), ("Exact%", "exact")]
    head = f"{'Condition':<16}" + "".join(f"{c[0]:>11}" for c in cols)
    print("\n" + head)
    print("-" * len(head))
    for label, _, _ in conditions:
        s = results[label]
        row = f"{label:<16}"
        for name, key in cols:
            v = s[key]
            row += f"{(v*100):>10.1f}" + ("%" if key in ("json", "exact", "deadline") else " ")
        print(row)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"n": len(rows), "summary": results, "details": all_details},
                  f, ensure_ascii=False, indent=2)
    print(f"\nfull per-example results written to {args.out}")


if __name__ == "__main__":
    main()
