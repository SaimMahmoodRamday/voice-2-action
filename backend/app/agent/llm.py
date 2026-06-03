import json
import re
import httpx

from app.config import settings


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def generate(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system

    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{settings.ollama_base_url}/api/generate", json=payload)
        r.raise_for_status()
        return r.json().get("response", "").strip()


def generate_json(prompt: str, system: str | None = None) -> dict:
    raw = generate(prompt, system=system)
    cleaned = _strip_fences(raw)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {raw!r}")
    return json.loads(match.group(0))
