// Typed client for the Voice2Action FastAPI backend.

export interface TaskExtraction {
  tasks: string[];
  deadline: string | null;
  people: string[];
  meetings: string[];
}

export interface ProcessResponse {
  extraction: TaskExtraction;
  missing_fields: string[];
  followup_question: string | null;
  // Short, rule-based explanation of why the follow-up was asked (null if none).
  reason: string | null;
  // Deterministic, high-level trace of the agent's decisions for this request.
  agent_trace: string[];
  // URL of the Notion page created when execution ran (null otherwise).
  notion_url: string | null;
}

export interface TranscribeResponse {
  transcript: string;
  language: string | null;
}

export interface FollowupRequest {
  extraction: TaskExtraction;
  missing_fields: string[];
  user_reply: string;
  execute?: boolean;
}

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function unwrap<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) {
        detail =
          typeof body.detail === "string"
            ? body.detail
            : JSON.stringify(body.detail);
      }
    } catch {
      // non-JSON error body — keep the status line
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

/** Audio file → transcript (Faster-Whisper). */
export async function transcribe(file: File): Promise<TranscribeResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/transcribe`, { method: "POST", body: form });
  return unwrap<TranscribeResponse>(res);
}

/** Transcript → structured extraction + follow-up question. */
export async function processText(
  transcript: string,
  execute = false,
): Promise<ProcessResponse> {
  const res = await fetch(`${BASE}/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ transcript, execute }),
  });
  return unwrap<ProcessResponse>(res);
}

/** Apply the user's answer to a follow-up question and get an updated extraction. */
export async function followup(req: FollowupRequest): Promise<ProcessResponse> {
  const res = await fetch(`${BASE}/followup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return unwrap<ProcessResponse>(res);
}
