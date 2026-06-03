import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.config import settings
from app.schemas import (
    TranscribeResponse,
    ProcessRequest,
    ProcessResponse,
    FollowupRequest,
    FollowupResponse,
    TaskExtraction,
)
from app.stt import whisper
from app.agent.graph import build_process_graph, build_followup_graph

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...)):
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "audio").suffix or ".wav"
    path = os.path.join(settings.upload_dir, f"{uuid.uuid4().hex}{suffix}")
    try:
        with open(path, "wb") as f:
            f.write(await file.read())
        text, lang = whisper.transcribe(path)
        return TranscribeResponse(transcript=text, language=lang)
    finally:
        if os.path.exists(path):
            os.remove(path)


@router.post("/process", response_model=ProcessResponse)
def process(req: ProcessRequest):
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="transcript is empty")
    graph = build_process_graph()
    result = graph.invoke({"transcript": req.transcript})
    return ProcessResponse(
        extraction=result["extraction"],
        missing_fields=result.get("missing_fields", []),
        followup_question=result.get("followup_question"),
    )


@router.post("/followup", response_model=FollowupResponse)
def followup(req: FollowupRequest):
    graph = build_followup_graph()
    result = graph.invoke({
        "extraction": req.extraction,
        "missing_fields": req.missing_fields,
        "user_reply": req.user_reply,
    })
    return FollowupResponse(
        extraction=result["extraction"],
        missing_fields=result.get("missing_fields", []),
        followup_question=result.get("followup_question"),
    )


@router.post("/voice2action", response_model=ProcessResponse)
async def voice2action(file: UploadFile = File(...)):
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "audio").suffix or ".wav"
    path = os.path.join(settings.upload_dir, f"{uuid.uuid4().hex}{suffix}")
    try:
        with open(path, "wb") as f:
            f.write(await file.read())
        transcript, _ = whisper.transcribe(path)
    finally:
        if os.path.exists(path):
            os.remove(path)

    if not transcript.strip():
        return ProcessResponse(
            extraction=TaskExtraction(),
            missing_fields=[],
            followup_question="I couldn't hear anything in that recording. Could you try again?",
        )

    graph = build_process_graph()
    result = graph.invoke({"transcript": transcript})
    return ProcessResponse(
        extraction=result["extraction"],
        missing_fields=result.get("missing_fields", []),
        followup_question=result.get("followup_question"),
    )
