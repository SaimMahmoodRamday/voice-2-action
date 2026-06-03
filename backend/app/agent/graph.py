from functools import lru_cache
from langgraph.graph import StateGraph, START, END

from app.agent.nodes import (
    AgentState,
    extract_node,
    validate_node,
    followup_question_node,
    merge_reply_node,
)


@lru_cache(maxsize=1)
def build_process_graph():
    g = StateGraph(AgentState)
    g.add_node("extract", extract_node)
    g.add_node("validate", validate_node)
    g.add_node("followup", followup_question_node)
    g.add_edge(START, "extract")
    g.add_edge("extract", "validate")
    g.add_edge("validate", "followup")
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
    g.add_edge("validate", "followup")
    g.add_edge("followup", END)
    return g.compile()
