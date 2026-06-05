from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class TaskExtraction(BaseModel):
    tasks: List[str] = Field(default_factory=list)
    deadline: Optional[str] = None
    people: List[str] = Field(default_factory=list)
    meetings: List[str] = Field(default_factory=list)

    @field_validator("tasks", "people", "meetings", mode="before")
    @classmethod
    def _none_to_empty_list(cls, v):
        # The LLM sometimes emits null for an empty list instead of [].
        return [] if v is None else v


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
