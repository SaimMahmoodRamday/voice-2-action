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
- Only modify a field if the reply CLEARLY provides a value for it. Keep every other field exactly as given.
- If the reply is vague, a question, or does not actually answer (e.g. "what", "huh", "idk", "pata nahi"), return the extraction UNCHANGED. Never invent or guess a deadline, name, task, or meeting.
- Use [] for empty lists and null for a missing deadline.
- Output JSON only. No prose, no markdown fences."""


FOLLOWUP_MERGE_USER_TEMPLATE = """Current extraction:
{extraction}

Missing fields: {missing}

User reply: {reply}

Updated JSON:"""
