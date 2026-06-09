from functools import lru_cache
from langgraph.graph import StateGraph, START, END

from app.agent.nodes import (
    AgentState,
    extract_node,
    validate_node,
    followup_question_node,
    merge_reply_node,
    execute_node,
)


def _route_after_validate(state: AgentState) -> str:
    """Decide where validation leads:

    - ``followup`` — there is an ambiguous gap to ask about.
    - ``execute``  — no gaps, execution was requested, and there is something
                     concrete to create (guards against empty/chit-chat input).
    - ``end``      — otherwise (the default, fully backward-compatible path).
    """
    if state.get("missing_fields"):
        return "followup"
    ex = state.get("extraction")
    if state.get("execute") and ex and (ex.tasks or ex.meetings):
        return "execute"
    return "end"


# Router return value → node name. The node is named "notion" (not "execute") to
# avoid colliding with the "execute" state key, which LangGraph disallows.
_ROUTES = {"followup": "followup", "execute": "notion", "end": END}


@lru_cache(maxsize=1)
def build_process_graph():
    g = StateGraph(AgentState)
    g.add_node("extract", extract_node)
    g.add_node("validate", validate_node)
    g.add_node("followup", followup_question_node)
    g.add_node("notion", execute_node)
    g.add_edge(START, "extract")
    g.add_edge("extract", "validate")
    g.add_conditional_edges("validate", _route_after_validate, _ROUTES)
    g.add_edge("followup", END)
    g.add_edge("notion", END)
    return g.compile()


@lru_cache(maxsize=1)
def build_followup_graph():
    g = StateGraph(AgentState)
    g.add_node("merge", merge_reply_node)
    g.add_node("validate", validate_node)
    g.add_node("followup", followup_question_node)
    g.add_node("notion", execute_node)
    g.add_edge(START, "merge")
    g.add_edge("merge", "validate")
    g.add_conditional_edges("validate", _route_after_validate, _ROUTES)
    g.add_edge("followup", END)
    g.add_edge("notion", END)
    return g.compile()
