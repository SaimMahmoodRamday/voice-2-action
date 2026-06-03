# Frontend (placeholder)

Not the current focus. Initialize when backend is stable:

```bash
cd ..
rm -rf frontend
npx create-next-app@latest frontend --typescript --tailwind --eslint --app --no-src-dir --import-alias "@/*"
```

The backend exposes:

- `POST /transcribe` — multipart audio → `{ transcript, language }`
- `POST /process` — `{ transcript }` → `{ extraction, missing_fields, followup_question }`
- `POST /followup` — `{ extraction, missing_fields, user_reply }` → updated extraction
- `POST /voice2action` — multipart audio → full pipeline in one call
