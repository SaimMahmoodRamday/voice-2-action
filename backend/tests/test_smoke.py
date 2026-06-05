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

    state = {"extraction": TaskExtraction(tasks=["Send proposal"])}
    out = validate_node(state)
    assert "deadline" in out["missing_fields"]
    assert "people" in out["missing_fields"]
