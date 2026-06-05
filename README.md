# Voice2Action

Voice2Action is an agentic AI system that converts informal voice notes in **Roman Urdu, Urdu, and mixed Urdu-English** into structured, actionable tasks — complete with deadlines, people, priorities, and follow-up clarification when information is missing.

---

## Problem Statement

People frequently communicate tasks through voice notes like:

> "yar client ne kaha proposal bhejna hai friday tak aur Ali se approval bhi lena hai"

These messages often contain tasks, deadlines, and people — but in an unstructured, informal way. Important details get lost, deadlines go missing, and there's no system to clarify incomplete information.

---

## Solution

Voice2Action processes a raw voice note and produces a structured action plan.

**Input (voice note transcript)**

```
yar client ne proposal bhejna hai aur Ali se approval lena hai
```

**Output (structured JSON)**

```json
{
  "tasks": [
    "Send proposal",
    "Get approval from Ali"
  ],
  "deadline": null,
  "people": ["Ali"]
}
```

**Agent follow-up (when information is missing)**

```
I detected the following tasks:
  - Send proposal
  - Get approval from Ali

No deadline was mentioned. Would you like to add one?
```

---

## Key Features

### 1. Voice Note Processing
- Accepts audio input (WhatsApp-style voice notes)
- Transcribes using Faster-Whisper
- Supports Roman Urdu, Urdu, and mixed Urdu-English speech

### 2. Fine-Tuned Task Extraction
A fine-tuned LLM extracts structured fields from noisy, informal text:
- Tasks
- Deadlines
- People involved
- Meetings

**Example**

Input:
```
Ali ko call karna hai aur friday tak proposal bhejna hai
```

Output:
```json
{
  "tasks": ["Call Ali", "Send proposal"],
  "deadline": "Friday",
  "people": ["Ali"]
}
```

### 3. Agentic Follow-Up System
If required information is missing, the system actively asks the user clarifying questions and updates the structured output after the user responds.

### 4. Multi-Step Agent Workflow (LangGraph)
The system runs as a graph-based pipeline:

```
Voice Input
  → Faster-Whisper (Speech-to-Text)
  → Fine-Tuned LLM (Task Extraction)
  → Validation Node
  → Missing Info Detector
  → Follow-Up Question Generator
  → User Response Handler
  → Final Structured Output
```

---

## System Architecture

```
┌────────────────────┐
│    Voice Input     │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│  Faster-Whisper    │
│  (Speech-to-Text)  │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│  Fine-Tuned LLM    │
│  (Qwen / Gemma)    │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Structured Output  │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│  LangGraph Agent   │
│ (Validation + QA)  │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│  Final Action Plan │
└────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python |
| Speech-to-Text | Faster-Whisper |
| LLM | Qwen2.5-3B / Gemma 3B (fine-tuned via QLoRA + PEFT) |
| Agent System | LangGraph |
| Database | _Planned_ — SQLite (MVP), PostgreSQL (scaling) |

---

## Why Fine-Tuning?

Rather than relying solely on prompting, the model is fine-tuned to:
- Understand Roman Urdu sentence structure
- Extract tasks consistently from informal speech
- Produce well-formed JSON outputs
- Handle noisy, mixed-language input

This gives us domain-specific behavior, consistent output formatting, and a model that can be deployed offline at lower inference cost — things that generic APIs don't reliably provide.

---

## Dataset Format

Training examples follow this structure:

```
Input:  yar Ali ko call karna hai
Output: { "tasks": ["Call Ali"] }
```

```
Input:  client se baat hui thi friday tak proposal bhejna hai aur Ali se approval lena hai
Output: {
  "tasks": ["Send proposal", "Get approval from Ali"],
  "deadline": "Friday",
  "people": ["Ali"]
}
```

---

## Agent Logic

The agent detects missing fields and triggers a clarification loop:

```
Step 1 — Extraction:   "Send proposal"
Step 2 — Validation:   Missing: deadline
Step 3 — Agent asks:   "Would you like to add a deadline?"
Step 4 — User replies: "Friday"
Step 5 — Final output: { "tasks": ["Send proposal"], "deadline": "Friday" }
```

---

## Getting Started

> Prerequisites: Python 3.10+, [Ollama](https://ollama.com), Node.js 18+ (frontend, later)

### Option A — Docker (recommended)

Everything (Ollama + model registration + backend) comes up with one command:

```bash
docker compose up
```

This starts Ollama, auto-registers the fine-tuned `voice2action` model via the
`model-init` service, then starts the backend on http://localhost:8000.

> **First run requires the model artifact.** The quantized weights
> (`ml/serving/voice2action-q8_0.gguf`, ~3.3 GB) are gitignored. Generate them
> once with the steps in [`ml/serving/README.md`](ml/serving/README.md) before
> `docker compose up`, or `model-init` will fail.

To use the GPU for inference (needs the NVIDIA Container Toolkit):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

### Option B — Local (host)

```bash
# 1. Register the fine-tuned model with your host Ollama (one time)
#    See ml/serving/README.md to generate the GGUF first.
cd ml/serving && ollama create voice2action -f Modelfile && cd ../..

# 2. Backend
cd backend
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux
cp .env.example .env        # then set OLLAMA_MODEL=voice2action
uvicorn app.main:app --reload
```

The backend is then available at http://localhost:8000 (`GET /health` to check).

**Frontend** (placeholder — not yet implemented)

```bash
cd frontend
npm install
npm run dev
```

---

## Example Use Cases

- "Ali ko call karna hai" → Task: Call Ali
- "Proposal friday tak bhejna hai" → Task + deadline
- "Client meeting kal hai" → Meeting extraction
- "Electricity bill pay karna hai" → Reminder
- "Doodh le ana" → Shopping item

---

## Project Highlights

- Fine-tuned LLM (not just prompting)
- Agentic workflow with clarification loop (LangGraph)
- Speech-to-text integration (Faster-Whisper)
- Roman Urdu NLP — a genuinely underserved domain
- End-to-end production-style architecture

---

## Future Roadmap

- Google Calendar integration
- WhatsApp bot interface
- Mobile app (Flutter)
- Personal task history and memory
- Multilingual expansion (Arabic, Hindi)

---

## Status

In active development — currently at MVP stage.