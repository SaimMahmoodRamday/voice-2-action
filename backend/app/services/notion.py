"""Notion task-execution tool.

A thin, production-grade client over the official Notion REST API (no SDK, no
mock). Given a completed extraction, it creates ONE page per voice note in the
configured database: the page title summarizes the note, and the tasks,
meetings, deadline, and people are written into the page body.

The feature is OFF by default — `is_configured()` is False until both
NOTION_TOKEN and NOTION_DATABASE_ID are set via the environment.
"""

import httpx

from app.config import settings
from app.schemas import TaskExtraction

_NOTION_API = "https://api.notion.com/v1/pages"
# Pinned API version — Notion requires this header and treats it as a contract.
_NOTION_VERSION = "2022-06-28"


def is_configured() -> bool:
    """True only when the integration token and target database are both set."""
    return bool(settings.notion_token and settings.notion_database_id)


def _rich_text(content: str) -> list:
    return [{"type": "text", "text": {"content": content}}]


def _page_title(ex: TaskExtraction) -> str:
    """A concise, human-readable page title for the note."""
    items = ex.tasks or ex.meetings
    if len(items) == 1:
        return items[0]
    if items:
        return f"Voice note – {len(ex.tasks)} task(s), {len(ex.meetings)} meeting(s)"
    return "Voice note"


def _para(content: str) -> dict:
    return {"object": "block", "type": "paragraph",
            "paragraph": {"rich_text": _rich_text(content)}}


def _blocks(ex: TaskExtraction) -> list:
    """Body blocks: tasks as checkboxes, meetings as bullets, then metadata."""
    blocks: list = []
    for task in ex.tasks:
        blocks.append({"object": "block", "type": "to_do",
                       "to_do": {"rich_text": _rich_text(task), "checked": False}})
    for meeting in ex.meetings:
        blocks.append({"object": "block", "type": "bulleted_list_item",
                       "bulleted_list_item": {"rich_text": _rich_text(f"Meeting: {meeting}")}})
    if ex.deadline:
        blocks.append(_para(f"Deadline: {ex.deadline}"))
    if ex.people:
        blocks.append(_para(f"People: {', '.join(ex.people)}"))
    return blocks


def create_task_page(ex: TaskExtraction) -> str:
    """Create the Notion page for this extraction and return its URL.

    Raises RuntimeError on any misconfiguration or API/transport failure so the
    caller can report it without crashing the request (the extraction itself is
    never lost).
    """
    if not is_configured():
        raise RuntimeError(
            "Notion is not configured — set NOTION_TOKEN and NOTION_DATABASE_ID")

    payload = {
        "parent": {"database_id": settings.notion_database_id},
        "properties": {
            settings.notion_title_prop: {"title": _rich_text(_page_title(ex))},
        },
        "children": _blocks(ex),
    }
    headers = {
        "Authorization": f"Bearer {settings.notion_token}",
        "Notion-Version": _NOTION_VERSION,
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
            r = client.post(_NOTION_API, json=payload, headers=headers)
            r.raise_for_status()
            return r.json().get("url", "")
    except httpx.HTTPStatusError as e:
        # Surface Notion's own error message — usually a clear cause (bad token,
        # database not shared with the integration, wrong title property name).
        raise RuntimeError(
            f"Notion API error {e.response.status_code}: {e.response.text}") from e
    except httpx.HTTPError as e:
        raise RuntimeError(f"Notion service unavailable: {e}") from e
