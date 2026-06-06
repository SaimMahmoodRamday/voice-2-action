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
