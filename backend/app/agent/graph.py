from functools import lru_cache
from langgraph.graph import StateGraph, START, END

from app.agent.nodes import (
    AgentState,
    extract_node,
    validate_node,
    followup_question_node,
    merge_reply_node,
)


def _needs_followup(state: AgentState) -> str:
    """Route to the follow-up node only when there is an ambiguous gap to ask about."""
    return "followup" if state.get("missing_fields") else "end"


@lru_cache(maxsize=1)
def build_process_graph():
    g = StateGraph(AgentState)
    g.add_node("extract", extract_node)
    g.add_node("validate", validate_node)
    g.add_node("followup", followup_question_node)
    g.add_edge(START, "extract")
    g.add_edge("extract", "validate")
    g.add_conditional_edges("validate", _needs_followup, {"followup": "followup", "end": END})
    g.add_edge("followup", END)
    return g.compile()


@lru_cache(maxsize=1)
def build_followup_graph():
    g = StateGraph(AgentState)
    g.add_node("merge", merge_reply_node)
    g.add_node("validate", validate_node)
    g.add_node("followup", followup_question_node)
    g.add_edge(START, "merge")
    g.add_edge("merge", "validate")
    g.add_conditional_edges("validate", _needs_followup, {"followup": "followup", "end": END})
    g.add_edge("followup", END)
    return g.compile()
