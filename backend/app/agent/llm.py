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

    # Fail fast and clearly if the model service is unreachable/slow (e.g. Ollama
    # still booting in Docker) instead of hanging or surfacing a raw httpx error.
    try:
        with httpx.Client(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
            r = client.post(f"{settings.ollama_base_url}/api/generate", json=payload)
            r.raise_for_status()
            return r.json().get("response", "").strip()
    except httpx.HTTPError as e:
        raise RuntimeError(f"LLM service unavailable: {e}") from e


def generate_json(prompt: str, system: str | None = None, retries: int = 1,
                  temperature: float = 0.2) -> dict:
    """Generate and parse a JSON object, self-repairing on malformed output.

    On a parse failure the model is re-prompted at temperature 0 (greedy), which
    reliably recovers the occasional non-JSON or truncated response instead of
    failing the whole request.
    """
    last = ""
    for attempt in range(retries + 1):
        last = generate(prompt, system=system, temperature=0.0 if attempt else temperature)
        cleaned = _strip_fences(last)
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
    raise ValueError(f"No valid JSON object after {retries + 1} attempts: {last!r}")
