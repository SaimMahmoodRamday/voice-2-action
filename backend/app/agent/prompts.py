EXTRACTION_SYSTEM = """You extract structured task information from informal voice-note transcripts in Roman Urdu, Urdu, or mixed Urdu-English.

Return ONLY a JSON object with this exact shape:
{
  "tasks": [string, ...],        // short imperative English task descriptions
  "deadline": string | null,     // natural deadline if present, e.g. "Friday", "kal", "5pm"
  "people": [string, ...],       // names of people mentioned
  "meetings": [string, ...]      // meetings/events mentioned
}

Rules:
- Translate Roman Urdu tasks into concise English (e.g. "Ali ko call karna hai" -> "Call Ali").
- Use null for a missing deadline. Use [] for missing lists.
- Output JSON only. No prose, no markdown fences."""


EXTRACTION_USER_TEMPLATE = """Transcript:
{transcript}

JSON:"""


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
