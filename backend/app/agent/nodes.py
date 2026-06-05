from typing import List, Optional, TypedDict

from app.agent import llm, prompts
from app.schemas import TaskExtraction


class AgentState(TypedDict, total=False):
    transcript: str
    extraction: TaskExtraction
    missing_fields: List[str]
    followup_question: Optional[str]
    user_reply: Optional[str]           


def extract_node(state: AgentState) -> AgentState:
    transcript = state["transcript"]
    data = llm.generate_json(
        prompts.EXTRACTION_USER_TEMPLATE.format(transcript=transcript),
        system=prompts.EXTRACTION_SYSTEM,
    )
    return {"extraction": TaskExtraction(**data)}


def validate_node(state: AgentState) -> AgentState:
    ex = state["extraction"]
    missing: List[str] = []
    if ex.tasks and ex.deadline is None:
        missing.append("deadline")
    if ex.tasks and not ex.people:
        missing.append("people")
    return {"missing_fields": missing}


def followup_question_node(state: AgentState) -> AgentState:
    missing = state.get("missing_fields") or []
    if not missing:
        return {"followup_question": None}
    question = llm.generate(
        prompts.FOLLOWUP_QUESTION_TEMPLATE.format(missing=", ".join(missing)),
        temperature=0.4,
    )
    return {"followup_question": question.strip()}


def merge_reply_node(state: AgentState) -> AgentState:
    reply = state.get("user_reply")
    if not reply:
        return {}
    ex = state["extraction"]
    missing = state.get("missing_fields") or []
    data = llm.generate_json(
        prompts.FOLLOWUP_MERGE_USER_TEMPLATE.format(
            extraction=ex.model_dump_json(),
            missing=", ".join(missing),
            reply=reply,
        ),
        system=prompts.FOLLOWUP_MERGE_SYSTEM,
    )
    return {"extraction": TaskExtraction(**data)}
