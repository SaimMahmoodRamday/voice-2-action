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
    # Opt-in Notion task execution. OFF by default → unchanged behavior.
    execute: bool = False


class ProcessResponse(BaseModel):
    extraction: TaskExtraction
    missing_fields: List[str]
    followup_question: Optional[str] = None
    # Why the follow-up question was asked (rule-based; None when none asked).
    reason: Optional[str] = None
    # Deterministic, high-level trace of the agent's decisions for this request.
    agent_trace: List[str] = Field(default_factory=list)
    # URL of the Notion page created when execution ran (None otherwise).
    notion_url: Optional[str] = None


class FollowupRequest(BaseModel):
    extraction: TaskExtraction
    missing_fields: List[str]
    user_reply: str
    execute: bool = False


class FollowupResponse(BaseModel):
    extraction: TaskExtraction
    missing_fields: List[str]
    followup_question: Optional[str] = None
    reason: Optional[str] = None
    agent_trace: List[str] = Field(default_factory=list)
    notion_url: Optional[str] = None
