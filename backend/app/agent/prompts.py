# IMPORTANT: this must stay identical to the system prompt used during
# fine-tuning (ml/scripts/prepare_dataset.py) and evaluation, and the user turn
# must be the RAW transcript (no wrapper) — so the deployed model runs exactly
# in-distribution. The fine-tuned model learned the JSON shape from training, so
# no schema spec is needed here. Changing this risks degrading live accuracy
# relative to the measured eval numbers.
EXTRACTION_SYSTEM = (
    "You extract structured task information from informal voice-note transcripts "
    "in Roman Urdu, Urdu, or mixed Urdu-English. Return ONLY a JSON object with keys "
    "tasks, deadline, people, meetings."
)


FOLLOWUP_QUESTION_TEMPLATE = """The following information is missing from the extracted tasks: {missing}.

Write ONE short, friendly clarifying question (in English) asking the user to provide the missing information. Output the question only, no preamble."""


FOLLOWUP_MERGE_SYSTEM = """You update a partial task extraction with new information from the user's reply.

You are given the current JSON extraction, the fields that were missing, and the user's reply.
Return the updated JSON with EXACTLY this shape:
{
  "tasks": [string, ...],
  "deadline": string | null,
  "people": [string, ...],
  "meetings": [string, ...]
}

Rules:
- Only modify fields that the reply addresses; keep the others unchanged.
- Use [] for empty lists and null for a missing deadline.
- Output JSON only. No prose, no markdown fences."""


FOLLOWUP_MERGE_USER_TEMPLATE = """Current extraction:
{extraction}

Missing fields: {missing}

User reply: {reply}

Updated JSON:"""
