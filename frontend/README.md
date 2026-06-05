# Voice2Action — Frontend

Next.js (App Router) + TypeScript + Tailwind UI for the Voice2Action backend.

Upload a voice note (`.m4a/.mp3/.ogg/.wav`) or type a transcript, and get back
structured tasks, deadline, people, and meetings — with an inline
follow-up loop when information is missing.

## Run

```bash
npm install
npm run dev          # http://localhost:3000
```

The backend must be running (default `http://localhost:8000`). Configure the URL
in `.env.local`:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## How it talks to the backend

`lib/api.ts` wraps these endpoints:

- **Upload audio** → `POST /transcribe` (audio → transcript) → `POST /process`
  (transcript → extraction). Two calls so the transcript can be shown.
- **Type text** → `POST /process` directly.
- **Follow-up answer** → `POST /followup` (updates the extraction in place).

> CORS: the backend allows all origins in dev (`app/main.py`). Tighten before
> deploying.

## Structure

```
app/
  layout.tsx     # metadata + fonts
  page.tsx       # the whole UI (client component)
  globals.css    # Tailwind + theme tokens
lib/
  api.ts         # typed backend client
```
