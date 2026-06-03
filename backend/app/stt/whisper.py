from functools import lru_cache
from faster_whisper import WhisperModel

from app.config import settings


@lru_cache(maxsize=1)
def get_model() -> WhisperModel:
    return WhisperModel(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )


def transcribe(audio_path: str) -> tuple[str, str | None]:
    model = get_model()
    segments, info = model.transcribe(
        audio_path,
        language=None,
        vad_filter=True,
        beam_size=5,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, info.language
