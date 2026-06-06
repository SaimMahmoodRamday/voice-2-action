# Voice2Action

> Convert informal voice notes in Roman Urdu, Urdu, and mixed Urdu-English into structured, actionable tasks — with deadlines, people, meetings, and agentic follow-up clarification.

---

## Overview

People communicate tasks informally — through voice notes, quick messages, and casual speech. Something like:

> *"yar client ne kaha proposal bhejna hai friday tak aur Ali se approval bhi lena hai"*

contains a deadline, two tasks, and a person — but buried in informal, mixed-language speech. Voice2Action extracts all of it automatically.

The system transcribes audio, runs a fine-tuned language model to extract structured fields, and then uses an agentic graph to detect missing information and ask targeted follow-up questions — producing a clean, actionable output every time.

---

## Demo

**Input (voice note)**
```
yar client ne proposal bhejna hai friday tak aur Ali se approval bhi lena hai
```

**Extracted Output**
```json
{
  "tasks": ["Send proposal", "Get approval from Ali"],
  "deadline": "Friday",
  "people": ["Ali"],
  "meetings": []
}
```

**Agent follow-up (when fields are missing)**
```
I found the following tasks:
  - Send proposal
  - Get approval from Ali

No deadline was mentioned — when does this need to be done?
```

---

## Key Features

- **Roman Urdu NLP** — handles the real way Pakistanis communicate: informal, code-switched, spoken-language text
- **Fine-tuned LLM** — Qwen2.5-3B fine-tuned with QLoRA on a curated 1,500+ example dataset; not just prompting
- **Agentic clarification loop** — LangGraph detects missing fields and asks targeted follow-up questions
- **Full speech-to-text pipeline** — Faster-Whisper (medium) transcribes voice notes in Urdu and Roman Urdu
- **One-command deployment** — Docker Compose brings up the entire stack (Ollama + model + backend + frontend)
- **GPU-accelerated inference** — optional NVIDIA GPU support via a compose override file

---

## Architecture

```
┌──────────────────────┐
│   Voice Note (audio) │
└──────────┬───────────┘
           │  POST /voice2action  or  POST /transcribe → POST /process
           ▼
┌──────────────────────┐
│   Faster-Whisper     │  Speech-to-text (medium model, int8, Urdu/Roman Urdu)
│   Speech-to-Text     │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   Fine-Tuned LLM     │  Qwen2.5-3B-Instruct + QLoRA adapter → GGUF, served via Ollama
│   Task Extraction    │  Extracts: tasks · deadline · people · meetings
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   LangGraph Agent    │  Validation → Missing field detection → Follow-up generation
│   Agentic Pipeline   │  → User reply merge → Final structured output
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Structured JSON     │  Returned via FastAPI REST API
│  Action Plan         │
└──────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| **Frontend** | Next.js 14, TypeScript, Tailwind CSS | App router, real-time voice recording |
| **Backend** | FastAPI, Python 3.11, Uvicorn | Async REST API, Pydantic v2 validation |
| **Speech-to-Text** | Faster-Whisper (medium) | int8 quantized, Urdu + Roman Urdu |
| **LLM** | Qwen2.5-3B-Instruct | Fine-tuned via QLoRA + PEFT |
| **LLM Serving** | Ollama | GGUF (q8_0), local inference |
| **Agent Framework** | LangGraph + LangChain Core | Graph-based multi-step agent |
| **Training** | Transformers, PEFT, BitsAndBytes | 4-bit NF4 QLoRA, bfloat16 compute |
| **Containerisation** | Docker, Docker Compose | GPU override via compose file |

---

## Fine-Tuning

Rather than relying on prompt engineering alone, the model is fine-tuned end-to-end on a curated dataset of **1,500+ Roman Urdu / mixed-language examples** covering:

- Casual task mentions (`yar Ali ko call karna hai`)
- Multi-task entries with deadlines (`proposal friday tak bhejna hai aur contract sign karna hai`)
- Meeting extraction (`monday ko client meeting hai 3 baje`)
- Low-priority errands (`doodh le ana`)
- Urgent high-priority items (`urgent hai deployment karna hai aaj tak`)

**Training configuration** (`ml/configs/qlora.yaml`):

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Method | QLoRA (4-bit NF4, bfloat16 compute) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q/k/v/o proj + gate/up/down proj |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Max sequence length | 1024 |

Fine-tuning produces a LoRA adapter that is merged and exported to GGUF format for local serving via Ollama.

---

## Fine-Tuning Results

Does fine-tuning actually beat prompting? Measured on a **148-example held-out test set** that the model never saw during training (split stratified by example type *and* spread across the dataset to avoid positional bias; verified zero train/test leakage).

**Conditions** — all three receive the *same* system prompt and the raw transcript, decoded greedily (`temperature 0`). The only differences are in-context examples and fine-tuning:

1. **Base, zero-shot** — `qwen2.5:3b` with the instruction only
2. **Base, few-shot** — same model + 5 worked examples in context
3. **Fine-tuned** — the QLoRA-adapted `voice2action` model

**Results**

| Condition | JSON valid | Task F1 | People F1 | Meeting F1 | Deadline acc | **Exact match** |
|---|---|---|---|---|---|---|
| Base — zero-shot | 96.6% | 0.36 | 0.90 | 0.82 | 73.6% | **21.6%** |
| Base — few-shot (5 ex) | 100% | 0.72 | 0.93 | 0.83 | 81.8% | **45.9%** |
| **Fine-tuned** | **100%** | **0.99** | **0.99** | **1.00** | **98.6%** | **95.9%** |

> Fine-tuning lifted exact-match from **21.6%** (zero-shot) and **45.9%** (5-shot) to **95.9%**, and task-extraction F1 from **0.36 → 0.99** — gains prompting alone could not achieve. Few-shot prompting closed only about half the gap.

**Why the gap is where it is.** Base models are already decent at copying literal **names** (People F1 ≈ 0.90) — but weak at **task extraction** (F1 0.36), which requires Roman-Urdu→English translation, splitting compound actions, and disciplined JSON. That domain skill is exactly what fine-tuning installs.

**Representative cases** (base zero-shot vs. fine-tuned):

```text
Input:  yar utility bills pay karne hain aur contract sign karna hai June ke end tak
Base:   tasks as objects with an invented "priority" key — wrong schema
Tuned:  {"tasks": ["Pay utility bills", "Sign the contract"], "deadline": "end of June", ...}   ✓

Input:  sun presentation ready karni hai wednesday tak
Base:   {"tasks": ["sun presentation ready"], ...}        # untranslated, keeps filler "sun"
Tuned:  {"tasks": ["Prepare presentation"], "deadline": "Wednesday", ...}                       ✓

Input:  urgent hai Rukhsana ke sath sync karni hai
Base:   {"tasks": ["sync with Rukhsana"], "meetings": null}   # misclassifies a meeting as a task
Tuned:  {"tasks": [], "meetings": ["Sync with Rukhsana"], ...}                                  ✓
```

**Reproduce**

```bash
python ml/scripts/split_dataset.py            # stratified, position-spread 90/10 split
python ml/scripts/prepare_dataset.py --input ml/data/examples_train.jsonl --output ml/data/train.jsonl
python ml/scripts/train_qlora.py --config ml/configs/qlora.yaml
python ml/scripts/evaluate.py                 # base (0-shot + few-shot) vs fine-tuned → table
```

> **Scope note:** the test set is drawn from the same curated distribution, so these are strong *in-distribution* numbers; on noisy real-world speech all conditions drop somewhat, but the relative gap holds. Matching is lenient (set/substring/token-overlap) since task text is free-form; exact-match is the strict complement. Full per-example outputs: `ml/eval_results.json`.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/transcribe` | Upload audio → returns transcript |
| `POST` | `/process` | Transcript → structured extraction + follow-up |
| `POST` | `/followup` | User reply → merges into existing extraction |
| `POST` | `/voice2action` | Upload audio → full pipeline in one call |

**`POST /process`** — request:
```json
{ "transcript": "Ali ko call karna hai friday tak" }
```

**`POST /process`** — response:
```json
{
  "extraction": {
    "tasks": ["Call Ali"],
    "deadline": "Friday",
    "people": ["Ali"],
    "meetings": []
  },
  "missing_fields": [],
  "followup_question": null
}
```

---

## Project Structure

```
voice-2-action/
├── backend/                  # FastAPI service
│   ├── app/
│   │   ├── agent/            # LangGraph nodes, graph definitions, prompts, LLM client
│   │   ├── api/              # REST routes
│   │   ├── stt/              # Faster-Whisper integration
│   │   ├── schemas.py        # Pydantic request/response models
│   │   ├── config.py         # Environment settings
│   │   └── main.py           # FastAPI app entry point
│   ├── tests/                # Pytest smoke tests
│   └── Dockerfile
├── frontend/                 # Next.js 14 app
│   ├── app/                  # App router pages
│   └── lib/                  # API client utilities
├── ml/                       # Model training & serving
│   ├── configs/
│   │   └── qlora.yaml        # QLoRA training configuration
│   ├── data/
│   │   ├── examples.jsonl    # Curated training examples (1,500+)
│   │   ├── train.jsonl       # Formatted training split
│   │   └── test.jsonl        # Held-out evaluation split
│   ├── scripts/
│   │   ├── prepare_dataset.py  # Formats examples into chat template
│   │   ├── train_qlora.py      # QLoRA fine-tuning entrypoint
│   │   ├── merge_lora.py       # Merges adapter into base model
│   │   ├── infer.py            # Local inference / evaluation
│   │   └── split_dataset.py    # Train/test split utility
│   └── serving/
│       └── Modelfile           # Ollama model definition
├── docker-compose.yml          # Full stack (Ollama + backend + frontend)
├── docker-compose.gpu.yml      # GPU override
└── README.md
```

---

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/) and Docker Compose
- The quantized model weights: `ml/serving/voice2action-q8_0.gguf` (~3.3 GB)

> **Generate the GGUF once** before running Docker. See [`ml/README.md`](ml/README.md) for the full training → merge → export pipeline. If you have a pre-built artifact, drop it directly into `ml/serving/`.

### Option A — Docker (recommended)

```bash
docker compose up
```

This starts:
1. **Ollama** (model server) on port `11434`
2. **model-init** — one-shot service that registers the fine-tuned `voice2action` model (idempotent)
3. **Backend** (FastAPI) on port `8000`

For GPU acceleration (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

### Option B — Local Development

```bash
# 1. Register the model with your host Ollama instance
cd ml/serving
ollama create voice2action -f Modelfile
cd ../..

# 2. Backend
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env        # set OLLAMA_MODEL=voice2action
uvicorn app.main:app --reload
# → http://localhost:8000
```

```bash
# 3. Frontend
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

### Training From Scratch

```bash
cd ml

# Prepare the dataset
python scripts/prepare_dataset.py

# Fine-tune (requires a CUDA GPU)
python scripts/train_qlora.py --config configs/qlora.yaml

# Merge LoRA adapter into the base model
python scripts/merge_lora.py

# Export to GGUF (requires llama.cpp)
# See ml/README.md for the full export steps
```

---

## Example Use Cases

| Voice Note | Extracted |
|---|---|
| `"Ali ko call karna hai"` | Task: Call Ali |
| `"Proposal friday tak bhejna hai"` | Task + deadline: Friday |
| `"Monday ko client meeting hai 3 baje"` | Meeting: Monday 3pm |
| `"Electricity bill pay karna hai aaj tak"` | Task + deadline: today |
| `"Urgent hai deployment karna hai"` | Task + priority: High |
| `"No rush, doodh le ana"` | Task + priority: Low |

---

## Why This Matters

Roman Urdu is how hundreds of millions of people communicate daily — in WhatsApp messages, voice notes, and casual speech — but it is almost entirely absent from standard NLP tooling. This project demonstrates that a small, fine-tuned model (3B parameters) can outperform generic prompting on this domain by learning the specific vocabulary, sentence structures, and mixed-language patterns that characterise real Pakistani communication.

The agentic follow-up system means the output is always complete and actionable, even when the speaker omits details — which is the norm in conversational task delegation.

---

## Future Roadmap

- [ ] Google Calendar / Outlook integration — auto-schedule extracted tasks
- [ ] WhatsApp bot interface — receive and respond to voice notes in-app
- [ ] Personal task memory — cross-session context and history
- [ ] Mobile app (Flutter / React Native)
- [ ] Multilingual expansion — Arabic, Hindi, Bengali

---

*Built with FastAPI · LangGraph · Faster-Whisper · Qwen2.5 · QLoRA · Next.js · Docker*