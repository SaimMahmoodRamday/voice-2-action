from typing import List, Literal, Optional
from pydantic import BaseModel, Field


Priority = Literal["Low", "Medium", "High"]


class TaskExtraction(BaseModel):
    tasks: List[str] = Field(default_factory=list)
    deadline: Optional[str] = None
    people: List[str] = Field(default_factory=list)
    meetings: List[str] = Field(default_factory=list)
    priority: Optional[Priority] = None


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
