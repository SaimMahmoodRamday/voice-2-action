EXTRACTION_SYSTEM = """You extract structured task information from informal voice-note transcripts in Roman Urdu, Urdu, or mixed Urdu-English.

Return ONLY a JSON object with this exact shape:
{
  "tasks": [string, ...],        // short imperative English task descriptions
  "deadline": string | null,     // natural deadline if present, e.g. "Friday", "kal", "5pm"
  "people": [string, ...],       // names of people mentioned
  "meetings": [string, ...],     // meetings/events mentioned
  "priority": "Low" | "Medium" | "High" | null
}

Rules:
- Translate Roman Urdu tasks into concise English (e.g. "Ali ko call karna hai" -> "Call Ali").
- Use null for missing deadline/priority. Use [] for missing lists.
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
  "meetings": [string, ...],
  "priority": "Low" | "Medium" | "High" | null
}

Rules:
- "priority" MUST be exactly one of "Low", "Medium", "High" (capitalized) or null. Map cues like "urgent"/"asap" to "High", "koi jaldi nahi" to "Low".
- Only modify fields that the reply addresses; keep the others unchanged.
- Use [] for empty lists and null for a missing deadline/priority.
- Output JSON only. No prose, no markdown fences."""


FOLLOWUP_MERGE_USER_TEMPLATE = """Current extraction:
{extraction}

Missing fields: {missing}

User reply: {reply}

Updated JSON:"""
