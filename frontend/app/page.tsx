"use client";

import { useRef, useState } from "react";
import {
  followup,
  processText,
  transcribe,
  type ProcessResponse,
} from "@/lib/api";

type Mode = "upload" | "text";

const SAMPLE =
  "client ne kaha proposal friday tak bhejna hai aur Ali se approval bhi lena hai";

export default function Home() {
  const [mode, setMode] = useState<Mode>("upload");
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [result, setResult] = useState<ProcessResponse | null>(null);
  const [answer, setAnswer] = useState("");
  const [answering, setAnswering] = useState(false);
  const [execute, setExecute] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const busy = loading || answering;

  async function handleSubmit() {
    setError(null);
    setResult(null);
    setTranscript(null);
    setAnswer("");

    if (mode === "upload") {
      if (!file) return setError("Choose an audio file first.");
    } else if (!text.trim()) {
      return setError("Type a voice-note transcript first.");
    }

    let source: string;
    setLoading(true);
    try {
      if (mode === "upload") {
        setStage("Transcribing audio…");
        const r = await transcribe(file!);
        if (!r.transcript.trim()) {
          throw new Error("Couldn't hear anything in that recording.");
        }
        source = r.transcript;
      } else {
        source = text.trim();
      }
      setTranscript(source);
      setStage(execute ? "Extracting & creating task…" : "Extracting tasks…");
      setResult(await processText(source, execute));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
      setStage("");
    }
  }

  async function handleAnswer() {
    if (!result || !answer.trim()) return;
    setAnswering(true);
    setError(null);
    try {
      const updated = await followup({
        extraction: result.extraction,
        missing_fields: result.missing_fields,
        user_reply: answer.trim(),
        execute,
      });
      setResult(updated);
      setAnswer("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't apply your answer.");
    } finally {
      setAnswering(false);
    }
  }

  return (
    <main className="flex flex-1 justify-center bg-gradient-to-b from-slate-50 to-slate-100 px-4 py-10 sm:py-16">
      <div className="w-full max-w-2xl">
        {/* Header */}
        <header className="mb-8 flex items-center gap-3">
          <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 text-white shadow-sm">
            <MicIcon />
          </span>
          <div>
            <h1 className="text-xl font-semibold tracking-tight text-slate-900">
              Voice2Action
            </h1>
            <p className="text-sm text-slate-500">
              Turn Roman Urdu voice notes into structured tasks.
            </p>
          </div>
        </header>

        {/* Input card */}
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 inline-flex rounded-lg bg-slate-100 p-1 text-sm font-medium">
            <TabButton active={mode === "upload"} onClick={() => setMode("upload")}>
              Upload audio
            </TabButton>
            <TabButton active={mode === "text"} onClick={() => setMode("text")}>
              Type text
            </TabButton>
          </div>

          {mode === "upload" ? (
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              className="flex w-full flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-slate-300 px-4 py-10 text-center transition hover:border-indigo-400 hover:bg-indigo-50/40"
            >
              <UploadIcon />
              {file ? (
                <span className="text-sm font-medium text-slate-700">
                  {file.name}
                </span>
              ) : (
                <span className="text-sm text-slate-500">
                  Click to choose an audio file
                  <span className="block text-xs text-slate-400">
                    .m4a, .mp3, .ogg, .wav
                  </span>
                </span>
              )}
              <input
                ref={fileRef}
                type="file"
                accept="audio/*,.m4a,.ogg"
                className="hidden"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </button>
          ) : (
            <div>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={4}
                placeholder={`e.g. "${SAMPLE}"`}
                className="w-full resize-none rounded-xl border border-slate-300 px-3.5 py-3 text-sm text-slate-800 outline-none transition placeholder:text-slate-400 focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100"
              />
              <button
                type="button"
                onClick={() => setText(SAMPLE)}
                className="mt-1.5 text-xs font-medium text-indigo-600 hover:text-indigo-700"
              >
                Use an example
              </button>
            </div>
          )}

          <label className="mt-4 flex cursor-pointer items-center gap-2.5 text-sm text-slate-600">
            <input
              type="checkbox"
              checked={execute}
              onChange={(e) => setExecute(e.target.checked)}
              className="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-400"
            />
            Create the task in Notion when nothing is ambiguous
          </label>

          <button
            type="button"
            onClick={handleSubmit}
            disabled={busy}
            className="mt-3 flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? (
              <>
                <Spinner />
                {stage || "Working…"}
              </>
            ) : (
              "Extract tasks"
            )}
          </button>
        </section>

        {error && (
          <div className="mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <section className="mt-6 space-y-5">
            {transcript && (
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <Label>Transcript</Label>
                <p className="mt-1.5 text-sm leading-relaxed text-slate-600">
                  {transcript}
                </p>
              </div>
            )}

            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="mb-3">
                <Label>Tasks</Label>
              </div>

              {result.extraction.tasks.length ? (
                <ul className="space-y-2">
                  {result.extraction.tasks.map((task, i) => (
                    <li
                      key={i}
                      className="flex items-start gap-2.5 text-sm text-slate-800"
                    >
                      <CheckIcon />
                      <span>{task}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-slate-400">No tasks detected.</p>
              )}

              <div className="mt-5 grid grid-cols-1 gap-4 border-t border-slate-100 pt-4 sm:grid-cols-3">
                <Meta label="Deadline">
                  {result.extraction.deadline ? (
                    <span className="text-sm font-medium text-slate-800">
                      {result.extraction.deadline}
                    </span>
                  ) : (
                    <Muted />
                  )}
                </Meta>
                <Meta label="People">
                  <Chips items={result.extraction.people} tone="indigo" />
                </Meta>
                <Meta label="Meetings">
                  <Chips items={result.extraction.meetings} tone="violet" />
                </Meta>
              </div>
            </div>

            {/* Notion execution result */}
            {result.notion_url && (
              <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-5 shadow-sm">
                <div className="flex items-start gap-2.5">
                  <span className="mt-0.5 text-emerald-600">
                    <CheckIcon />
                  </span>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-emerald-900">
                      Task created in Notion.
                    </p>
                    <a
                      href={result.notion_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="mt-1 inline-block text-sm font-medium text-emerald-700 underline underline-offset-2 hover:text-emerald-800"
                    >
                      Open page →
                    </a>
                  </div>
                </div>
              </div>
            )}

            {/* Follow-up */}
            {result.followup_question && (
              <div className="rounded-2xl border border-amber-200 bg-amber-50 p-5 shadow-sm">
                <div className="flex items-start gap-2.5">
                  <span className="mt-0.5 text-amber-500">
                    <QuestionIcon />
                  </span>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-amber-900">
                      {result.followup_question}
                    </p>
                    {result.reason && (
                      <p className="mt-1 text-xs text-amber-700">
                        <span className="font-semibold">Why I asked:</span>{" "}
                        {result.reason}
                      </p>
                    )}
                    <div className="mt-3 flex gap-2">
                      <input
                        value={answer}
                        onChange={(e) => setAnswer(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleAnswer()}
                        placeholder="Your answer…"
                        className="flex-1 rounded-lg border border-amber-300 bg-white px-3 py-2 text-sm text-slate-800 outline-none transition placeholder:text-slate-400 focus:border-amber-400 focus:ring-2 focus:ring-amber-100"
                      />
                      <button
                        type="button"
                        onClick={handleAnswer}
                        disabled={answering || !answer.trim()}
                        className="flex items-center gap-1.5 rounded-lg bg-amber-500 px-3.5 py-2 text-sm font-semibold text-white transition hover:bg-amber-600 disabled:opacity-60"
                      >
                        {answering ? <Spinner /> : "Send"}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Agent reasoning trace */}
            {result.agent_trace.length > 0 && (
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <Label>Agent reasoning</Label>
                <ol className="mt-2.5 space-y-1.5">
                  {result.agent_trace.map((step, i) => (
                    <li
                      key={i}
                      className="flex items-start gap-2.5 text-sm text-slate-600"
                    >
                      <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-slate-100 text-xs font-semibold text-slate-500">
                        {i + 1}
                      </span>
                      <span>{step}</span>
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </section>
        )}

        <footer className="mt-10 text-center text-xs text-slate-400">
          Voice2Action · Faster-Whisper + fine-tuned Qwen2.5 via Ollama
        </footer>
      </div>
    </main>
  );
}

/* ---------- small presentational pieces ---------- */

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-md px-3.5 py-1.5 transition ${
        active
          ? "bg-white text-slate-900 shadow-sm"
          : "text-slate-500 hover:text-slate-700"
      }`}
    >
      {children}
    </button>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
      {children}
    </h2>
  );
}

function Meta({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <Label>{label}</Label>
      <div className="mt-1.5">{children}</div>
    </div>
  );
}

function Muted() {
  return <span className="text-sm text-slate-400">—</span>;
}

function Chips({
  items,
  tone,
}: {
  items: string[];
  tone: "indigo" | "violet";
}) {
  if (!items.length) return <Muted />;
  const cls =
    tone === "indigo"
      ? "bg-indigo-50 text-indigo-700 ring-indigo-600/15"
      : "bg-violet-50 text-violet-700 ring-violet-600/15";
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((item, i) => (
        <span
          key={i}
          className={`rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${cls}`}
        >
          {item}
        </span>
      ))}
    </div>
  );
}

/* ---------- icons ---------- */

function Spinner() {
  return (
    <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
      />
    </svg>
  );
}

function MicIcon() {
  return (
    <svg
      className="h-5 w-5"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 10a7 7 0 0 0 14 0M12 17v5" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg
      className="h-7 w-7 text-slate-400"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg
      className="mt-0.5 h-4 w-4 shrink-0 text-indigo-500"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M20 6 9 17l-5-5" />
    </svg>
  );
}

function QuestionIcon() {
  return (
    <svg
      className="h-5 w-5"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3M12 17h.01" />
    </svg>
  );
}
