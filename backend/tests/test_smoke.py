from fastapi.testclient import TestClient

from app.main import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_validate_node_flags_missing():
    from app.agent.nodes import validate_node
    from app.schemas import TaskExtraction

    # "Send proposal" is an interaction task with no resolved recipient → asks who + when
    state = {"extraction": TaskExtraction(tasks=["Send proposal"])}
    out = validate_node(state)
    assert "deadline" in out["missing_fields"]
    assert "recipient" in out["missing_fields"]


def test_validate_no_recipient_ask_for_plain_errand():
    from app.agent.nodes import validate_node
    from app.schemas import TaskExtraction

    # "Buy milk" involves no person → must NOT ask "who?"
    out = validate_node({"extraction": TaskExtraction(tasks=["Buy milk"])})
    assert "recipient" not in out["missing_fields"]
    assert "deadline" in out["missing_fields"]


def test_validate_placeholder_person_triggers_who():
    from app.agent.nodes import validate_node
    from app.schemas import TaskExtraction

    # model emitted a placeholder "user" for an unnamed reference → still ask who
    out = validate_node({"extraction": TaskExtraction(tasks=["Call user"], people=["user"])})
    assert "recipient" in out["missing_fields"]


def test_validate_unclear_on_actionable_empty():
    from app.agent.nodes import validate_node
    from app.schemas import TaskExtraction

    # R2: empty extraction but the transcript clearly asks for something
    out = validate_node({"extraction": TaskExtraction(), "transcript": "usey milna hai"})
    assert out["missing_fields"] == ["unclear"]


def test_validate_empty_chitchat_no_recovery():
    from app.agent.nodes import validate_node
    from app.schemas import TaskExtraction

    # R2: empty extraction on genuine chit-chat → no question (no false recovery)
    out = validate_node({"extraction": TaskExtraction(), "transcript": "yaar aaj mausam acha hai"})
    assert out["missing_fields"] == []


def test_validate_vague_deadline_is_missing():
    from app.agent.nodes import validate_node
    from app.schemas import TaskExtraction

    # R4: "soon" is not a concrete deadline → still ask when
    out = validate_node({"extraction": TaskExtraction(tasks=["Pay the bill"], deadline="soon")})
    assert "deadline" in out["missing_fields"]


def test_validate_meeting_without_time():
    from app.agent.nodes import validate_node
    from app.schemas import TaskExtraction

    # R3: a meeting with no time (and no tasks) → ask when
    out = validate_node({"extraction": TaskExtraction(meetings=["Meeting with Ali"], people=["Ali"])})
    assert "deadline" in out["missing_fields"]


def test_validate_emits_agent_trace():
    from app.agent.nodes import validate_node
    from app.schemas import TaskExtraction

    # Every validate decision is recorded as a deterministic, high-level step.
    out = validate_node({"extraction": TaskExtraction(tasks=["Send proposal"])})
    assert out["agent_trace"]  # non-empty
    assert any("recipient" in s for s in out["agent_trace"])
    assert any("deadline" in s for s in out["agent_trace"])


def test_followup_question_carries_reason():
    from app.agent.nodes import followup_question_node
    from app.schemas import TaskExtraction

    # A generated follow-up explains *why* it was asked, by ambiguity type.
    out = followup_question_node({
        "extraction": TaskExtraction(tasks=["Send proposal"]),
        "missing_fields": ["recipient", "deadline"],
    })
    assert out["followup_question"]
    assert out["reason"] and out["reason"].startswith("Asked because")
    assert "recipient" in out["reason"]
    assert "deadline" in out["reason"]
    assert out["agent_trace"]


def test_followup_reason_for_unclear_recovery():
    from app.agent.nodes import followup_question_node
    from app.schemas import TaskExtraction

    # R2 recovery prompt also carries a reason and no question depends on extraction.
    out = followup_question_node({
        "extraction": TaskExtraction(),
        "missing_fields": ["unclear"],
    })
    assert out["reason"]
    assert "rephrase" in out["followup_question"].lower()


def test_recover_deadline_handles_spelling_variants():
    from app.agent.nodes import _recover_deadline

    # The model resolves "parso" but misses the contracted "parson"/"prson";
    # the deterministic fallback recovers all of them to a canonical value.
    for token in ["parso", "parson", "prson", "parsoon"]:
        assert _recover_deadline(f"Ali se milna hai {token}") == "day after tomorrow"
    assert _recover_deadline("Ali se milna hai") is None  # no time word → no guess


def test_no_followup_no_reason():
    from app.agent.nodes import followup_question_node
    from app.schemas import TaskExtraction

    out = followup_question_node({"extraction": TaskExtraction(), "missing_fields": []})
    assert out["followup_question"] is None
    assert out["reason"] is None


# --- Notion task execution (optional tool) -------------------------------

def test_notion_off_by_default(monkeypatch):
    from app.services import notion

    monkeypatch.setattr(notion.settings, "notion_token", "")
    monkeypatch.setattr(notion.settings, "notion_database_id", "")
    assert notion.is_configured() is False


def test_route_executes_only_when_requested_and_clean():
    from app.agent.graph import _route_after_validate
    from app.schemas import TaskExtraction

    ex = TaskExtraction(tasks=["Buy milk"], deadline="today")
    # default (no execute) → end: fully backward-compatible
    assert _route_after_validate({"extraction": ex, "missing_fields": []}) == "end"
    # execute + no gaps + real content → run the tool
    assert _route_after_validate(
        {"extraction": ex, "missing_fields": [], "execute": True}) == "execute"
    # execute + a gap → ask first, never execute on ambiguous input
    assert _route_after_validate(
        {"extraction": ex, "missing_fields": ["deadline"], "execute": True}) == "followup"
    # execute but nothing concrete (chit-chat) → end, never create an empty page
    assert _route_after_validate(
        {"extraction": TaskExtraction(), "missing_fields": [], "execute": True}) == "end"


def test_execute_node_returns_url(monkeypatch):
    from app.agent import nodes
    from app.schemas import TaskExtraction

    monkeypatch.setattr(nodes.notion, "create_task_page", lambda ex: "https://notion.so/abc")
    out = nodes.execute_node({"extraction": TaskExtraction(tasks=["Buy milk"])})
    assert out["notion_url"] == "https://notion.so/abc"
    assert any("Notion" in s for s in out["agent_trace"])


def test_execute_node_failure_is_nonfatal(monkeypatch):
    from app.agent import nodes
    from app.schemas import TaskExtraction

    def boom(ex):
        raise RuntimeError("bad token")

    monkeypatch.setattr(nodes.notion, "create_task_page", boom)
    out = nodes.execute_node({"extraction": TaskExtraction(tasks=["Buy milk"])})
    # extraction is preserved; the failure is reported, not raised
    assert "notion_url" not in out
    assert any("failed" in s.lower() for s in out["agent_trace"])


def test_notion_page_payload_shape():
    from app.services import notion
    from app.schemas import TaskExtraction

    ex = TaskExtraction(tasks=["Send proposal", "Get approval"], deadline="Friday", people=["Ali"])
    assert notion._page_title(ex).startswith("Voice note")  # multi-item summary
    blocks = notion._blocks(ex)
    assert [b["type"] for b in blocks].count("to_do") == 2   # one checkbox per task
    assert any(b["type"] == "paragraph" for b in blocks)     # deadline / people lines
