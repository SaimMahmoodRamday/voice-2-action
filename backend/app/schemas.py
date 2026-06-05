from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


Priority = Literal["Low", "Medium", "High"]


class TaskExtraction(BaseModel):
    tasks: List[str] = Field(default_factory=list)
    deadline: Optional[str] = None
    people: List[str] = Field(default_factory=list)
    meetings: List[str] = Field(default_factory=list)
    priority: Optional[Priority] = None

    @field_validator("tasks", "people", "meetings", mode="before")
    @classmethod
    def _none_to_empty_list(cls, v):
        # The LLM sometimes emits null for an empty list instead of [].
        return [] if v is None else v

    @field_validator("priority", mode="before")
    @classmethod
    def _normalize_priority(cls, v):
        # The LLM is inconsistent about casing/wording ("high", "urgent",
        # "koi jaldi nahi"). Normalize to the canonical enum; unrecognized
        # values become None rather than raising a 500.
        if v is None or not isinstance(v, str):
            return None
        key = v.strip().lower()
        if not key:
            return None
        exact = {"low": "Low", "medium": "Medium", "high": "High",
                 "normal": "Medium", "urgent": "High", "asap": "High"}
        if key in exact:
            return exact[key]
        # Loose fallback — check "low" cues first so negations win.
        if "low" in key or "no rush" in key or "jaldi nahi" in key:
            return "Low"
        if "high" in key or "urgent" in key or "asap" in key:
            return "High"
        if "medium" in key or "normal" in key:
            return "Medium"
        return None


class TranscribeResponse(BaseModel):
    transcript: str
    language: Optional[str] = None


class ProcessRequest(BaseModel):
    transcript: str


class ProcessResponse(BaseModel):
    extraction: TaskExtraction
    missing_fields: List[str]
    followup_question: Optional[str] = None


class FollowupRequest(BaseModel):
    extraction: TaskExtraction
    missing_fields: List[str]
    user_reply: str


class FollowupResponse(BaseModel):
    extraction: TaskExtraction
    missing_fields: List[str]
    followup_question: Optional[str] = None
