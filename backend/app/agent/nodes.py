import operator
import re
from typing import Annotated, List, Optional, TypedDict

from app.agent import llm, prompts
from app.schemas import TaskExtraction
from app.services import notion


class AgentState(TypedDict, total=False):
    transcript: str
    extraction: TaskExtraction
    missing_fields: List[str]
    followup_question: Optional[str]
    user_reply: Optional[str]
    asked_fields: List[str]
    # Optional Notion task execution — opt-in per request, OFF by default.
    execute: bool
    # URL of the Notion page created when execution runs (None otherwise).
    notion_url: Optional[str]
    # Why the follow-up was asked — a short, rule-based explanation set by the
    # follow-up node (None when no question is asked).
    reason: Optional[str]
    # Deterministic, high-level record of the decisions each node made. The
    # `operator.add` reducer makes every node's returned entries *append* to the
    # running trace instead of overwriting it.
    agent_trace: Annotated[List[str], operator.add]


# Tokens the extraction model sometimes emits for an *unresolved* person
# reference (e.g. "usko call karna hai" -> people: ["user"]). These are not real
# names, so we normalize them out — and treat their absence of a real name as the
# signal to ask "who?". This is a light cleanup of a known artifact; the
# fine-tuned model still does all the real entity extraction.
_PLACEHOLDER_PEOPLE = {
    "user", "someone", "somebody", "anyone", "him", "her", "them", "he", "she",
    "they", "usko", "unko", "isko", "inko", "us", "use", "uss", "uske", "unke",
}

# Verbs whose task implies a target person. If such a task has no resolved
# recipient, the genuinely useful clarification is "who?" — as opposed to blindly
# asking "who is involved?" for every task (e.g. "buy milk").
_INTERACTION_VERBS = (
    "call", "email", "send", "share", "forward", "message", "msg", "ask",
    "tell", "inform", "remind", "meet", "contact", "reply", "notify",
)

# Time references — used to verify a merged-in deadline is actually grounded in
# the user's reply (so a confused reply like "what" can't conjure a deadline).
_TIME = re.compile(
    r"(\btak\b|\bkal\b|parso|\bprson\b|\baaj\b|\baj\b|subah|shaam|\bsham\b|raat|dopahar|morning|evening|"
    r"afternoon|noon|midnight|tonight|tomorrow|today|yesterday|week|weekend|month|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|somwar|mangal|jumma|"
    r"hafta|hafte|mahin|din|eod|end of day|deadline|next|\d\s*[ap]\.?m|\d{1,2}:\d{2}|"
    r"\d{1,2}\s*baje|آج|کل|صبح|شام|رات|تک|بجے)",
    re.I,
)

# Vague deadlines that aren't actionable — treated as "no concrete deadline" so
# the agent asks for a specific time.
_VAGUE_DEADLINE = {"soon", "jaldi", "asap", "later", "baad mein", "kabhi", "jald", "abhi"}

# High-confidence, unambiguous relative-time expressions the fine-tuned model
# occasionally misses on noisy spellings — e.g. it resolves "parso" to "day after
# tomorrow" but returns nothing for the contracted "parson"/"prson". Applied only
# as a deterministic *fallback*: when the model produces no deadline but the
# source text clearly contains one of these, we fill the canonical English value.
# This keeps deadlines grounded in the actual text and independent of exact
# spelling. Keep entries unambiguous (avoid e.g. "kal", which is tomorrow OR
# yesterday); add new ones here as real-world spellings surface.
_CANONICAL_DEADLINE = (
    (re.compile(r"\b(?:parso|parson|parsoon|parsun|prso|prson|prsoon)\b", re.I),
     "day after tomorrow"),
)


def _recover_deadline(text: str) -> Optional[str]:
    for pattern, canonical in _CANONICAL_DEADLINE:
        if pattern.search(text or ""):
            return canonical
    return None

# Action cues — used to tell a *missed* task ("usey milna hai" -> empty) apart
# from genuine chit-chat ("aaj mausam acha hai"), so recovery is only offered on
# input that actually looks like a request. Deliberately excludes the bare copula
# "hai" (present in both tasks and small talk).
_ACTION_CUE = re.compile(
    r"(karna|karni|karne|karo|kar do|lena|leni|le ?ana|laana|laani|bhej|dena|milna|"
    r"jana|likh|bana|mangwa|\b(call|send|email|meet|buy|pay|remind|tell|ask|share|"
    r"forward|message|book|schedule|cancel|finish|complete|prepare|submit|sign)\b)",
    re.I,
)


def _real_people(people: List[str]) -> List[str]:
    return [p for p in (people or []) if p.strip().lower() not in _PLACEHOLDER_PEOPLE]


def _interaction_verb(task: str) -> Optional[str]:
    t = (task or "").lower()
    for v in _INTERACTION_VERBS:
        if re.search(rf"\b{v}", t):
            return v
    return None


def _is_vague_deadline(d: Optional[str]) -> bool:
    return bool(d) and d.strip().lower() in _VAGUE_DEADLINE


def extract_node(state: AgentState) -> AgentState:
    # Normalize whitespace (R7), then extract with the fine-tuned model using
    # greedy decoding (R1) — deterministic and matching the evaluated condition.
    transcript = " ".join((state.get("transcript") or "").split())
    data = llm.generate_json(transcript, system=prompts.EXTRACTION_SYSTEM, temperature=0.0)
    ex = TaskExtraction(**data)
    ex.people = _real_people(ex.people)  # drop placeholder pseudo-names

    # Deterministic fallback: recover a deadline the model missed on a noisy
    # spelling (e.g. "prson"), grounded in the transcript.
    recovered = _recover_deadline(transcript) if ex.deadline is None else None
    if recovered:
        ex.deadline = recovered

    trace = [
        f"Extracted {len(ex.tasks)} task(s), {len(ex.meetings)} meeting(s), "
        f"{len(ex.people)} person(s); deadline "
        f"{'found' if ex.deadline else 'not found'}"
    ]
    if recovered:
        trace.append(f"Recovered deadline '{recovered}' from transcript "
                     "(model missed the spelling)")
    return {"extraction": ex, "agent_trace": trace}


def validate_node(state: AgentState) -> AgentState:
    """Detect genuinely-ambiguous gaps — not blind field presence.

    - unclear:   nothing extracted, but the transcript looks actionable (R2)
    - recipient: an interaction task ("call/send/...") with no resolved person
    - deadline:  tasks/meetings with no concrete due date (R3 meetings, R4 vague)

    Gaps already asked (carried in `asked_fields`) are dropped, so the agent asks
    each gap at most once instead of nagging.
    """
    ex = state["extraction"]

    # R2: nothing extracted. Distinguish a missed task from real chit-chat —
    # only offer recovery when the input actually looks like a request.
    if not ex.tasks and not ex.meetings and not ex.people:
        if _ACTION_CUE.search(state.get("transcript") or ""):
            return {"missing_fields": ["unclear"],
                    "agent_trace": ["Validation: nothing extracted, but transcript "
                                    "looks actionable — flagging for recovery"]}
        return {"missing_fields": [],
                "agent_trace": ["Validation: no actionable content (chit-chat) — "
                                "nothing to clarify"]}

    asked = set(state.get("asked_fields") or [])
    detected: List[str] = []
    if ex.tasks and not _real_people(ex.people) and any(_interaction_verb(t) for t in ex.tasks):
        detected.append("recipient")
    if (ex.tasks or ex.meetings) and (ex.deadline is None or _is_vague_deadline(ex.deadline)):
        detected.append("deadline")

    gaps = [g for g in detected if g not in asked]
    skipped = [g for g in detected if g in asked]

    trace: List[str] = []
    if "recipient" in detected:
        trace.append("Validation: interaction task with no resolved recipient — "
                     "missing 'recipient'")
    if "deadline" in detected:
        trace.append("Validation: no concrete deadline — missing 'deadline'")
    if skipped:
        trace.append(f"Validation: already asked, not re-asking: {', '.join(skipped)}")
    if not gaps:
        trace.append("Validation: no outstanding gaps")
    return {"missing_fields": gaps, "agent_trace": trace}


def followup_question_node(state: AgentState) -> AgentState:
    """Compose a targeted question grounded in the actual task.

    Built deterministically so the question always matches `missing_fields`
    exactly (no LLM drift between what's flagged and what's asked).
    """
    gaps = state.get("missing_fields") or []
    if not gaps:
        return {"followup_question": None, "reason": None}

    # R2: soft recovery for a missed extraction.
    if "unclear" in gaps:
        return {
            "followup_question":
                "I may have missed the task. Could you rephrase it or provide a little more detail?",
            "reason":
                "The transcript looked like a request, but no task could be extracted from it.",
            "agent_trace": ["Generated recovery prompt (no task could be extracted)"],
        }

    ex = state["extraction"]
    parts: List[str] = []
    reasons: List[str] = []
    if "recipient" in gaps:
        task = next((t for t in ex.tasks if _interaction_verb(t)), None)
        verb = _interaction_verb(task) if task else None
        if verb in {"send", "share", "forward", "email"}:
            parts.append("who should I send this to")
        elif verb:
            parts.append(f"who should I {verb}")
        else:
            parts.append("who is this for")
        reasons.append("the task involves contacting someone but no recipient was named")
    if "deadline" in gaps:
        if ex.meetings and not ex.tasks:
            parts.append("when is the meeting")
            reasons.append("the meeting has no specified time")
        else:
            parts.append("when does this need to be done")
            reasons.append("no concrete deadline was provided")
    q = ", and ".join(parts)
    return {
        "followup_question": q[0].upper() + q[1:] + "?",
        "reason": "Asked because " + " and ".join(reasons) + ".",
        "agent_trace": [f"Generated follow-up question for: {', '.join(gaps)}"],
    }


def merge_reply_node(state: AgentState) -> AgentState:
    reply = state.get("user_reply")
    if not reply:
        return {"agent_trace": ["No user reply provided — nothing to merge"]}
    ex = state["extraction"]
    asked = state.get("asked_fields") or []
    data = llm.generate_json(
        prompts.FOLLOWUP_MERGE_USER_TEMPLATE.format(
            extraction=ex.model_dump_json(),
            missing=", ".join(asked),
            reply=reply,
        ),
        system=prompts.FOLLOWUP_MERGE_SYSTEM,
        temperature=0.0,  # deterministic — don't sample a guessed answer
    )
    merged = TaskExtraction(**data)
    merged.people = _real_people(merged.people)

    trace: List[str] = ["Merged user reply into extraction"]

    # Grounding guard: reject values the reply doesn't actually contain, so a
    # confused reply ("what") can't fabricate a deadline or a name.
    rl = reply.lower()
    if merged.deadline and merged.deadline != ex.deadline and not _TIME.search(reply):
        merged.deadline = ex.deadline
        trace.append("Rejected ungrounded deadline from reply (no time reference found)")

    # Deterministic fallback: if the model still produced no deadline but the
    # reply clearly names one (on a spelling it doesn't recognize, e.g. "prson"),
    # recover the canonical value — grounded in the reply.
    if merged.deadline is None:
        recovered = _recover_deadline(reply)
        if recovered:
            merged.deadline = recovered
            trace.append(f"Recovered deadline '{recovered}' from reply "
                         "(model missed the spelling)")

    merged.people = [p for p in merged.people if p in ex.people or p.lower() in rl]

    # R5: a follow-up may add or refine, but must never *lose* already-extracted
    # tasks/meetings (protects the fine-tune's good output through the loop).
    if len(merged.tasks) < len(ex.tasks):
        merged.tasks = ex.tasks
        trace.append("Restored previously-extracted tasks (merge must not lose them)")
    if len(merged.meetings) < len(ex.meetings):
        merged.meetings = ex.meetings
        trace.append("Restored previously-extracted meetings (merge must not lose them)")
    return {"extraction": merged, "agent_trace": trace}


def execute_node(state: AgentState) -> AgentState:
    """Optional tool step: create the task in Notion.

    Only reached when execution was requested *and* validation found no gaps (the
    routing in graph.py enforces this), so the extraction is complete and clean.
    A Notion failure is reported in the trace rather than raised — the extraction
    result is never lost over a side-effect that didn't land.
    """
    ex = state["extraction"]
    try:
        url = notion.create_task_page(ex)
        return {"notion_url": url, "agent_trace": [f"Executed: created Notion page ({url})"]}
    except RuntimeError as e:
        return {"agent_trace": [f"Notion execution failed: {e}"]}
