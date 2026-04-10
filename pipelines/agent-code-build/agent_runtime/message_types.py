"""Message and session-state helpers for the runtime layer.

This module centralizes history normalization and duplicate suppression rules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from .common import build_log_message, flatten_content, get_logger, message_digest


logger = get_logger("message_types")


class SessionState(TypedDict, total=False):
    """Typed shape of persisted per-chat runtime state."""

    chat_id: str
    history: List[Dict[str, Any]]
    summary_message: Optional[Dict[str, Any]]
    attachment_manifest: Dict[str, Any]
    file_read_progress: Dict[str, Any]
    transcript_paths: List[str]
    last_answer: str
    consecutive_compaction_failures: int
    permission_mode: str
    last_active_at: int
    compaction_snapshot: Dict[str, Any]
    last_prepare_turn_report: Dict[str, Any]
    prompt_history_state: Dict[str, Any]
    dynamic_read_budget: Dict[str, Any]


def default_session_state(chat_id: str, now_ts: int) -> SessionState:
    """Return a fresh default state for a chat session."""

    return {
        "chat_id": chat_id,
        "history": [],
        "summary_message": None,
        "attachment_manifest": {"files": []},
        "file_read_progress": {},
        "transcript_paths": [],
        "last_answer": "",
        "consecutive_compaction_failures": 0,
        "permission_mode": "auto",
        "last_active_at": now_ts,
        "compaction_snapshot": {},
        "last_prepare_turn_report": {},
        "prompt_history_state": {},
        "dynamic_read_budget": {},
    }


def latest_user_message(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the latest user-role message from a message list."""

    for message in reversed(messages or []):
        if isinstance(message, dict) and message.get("role") == "user":
            return message
    return None


def contains_tool_calls(message: Dict[str, Any]) -> bool:
    """Check whether a model response includes OpenAI-style tool calls."""

    tool_calls = message.get("tool_calls")
    return isinstance(tool_calls, list) and len(tool_calls) > 0


def contains_tool_use_block(content: Any) -> bool:
    """Check whether Anthropic-style `tool_use` blocks are present."""

    if not isinstance(content, list):
        return False
    return any(isinstance(item, dict) and item.get("type") == "tool_use" for item in content)


def contains_tool_result_block(content: Any) -> bool:
    """Check whether Anthropic-style `tool_result` blocks are present."""

    if not isinstance(content, list):
        return False
    return any(isinstance(item, dict) and item.get("type") == "tool_result" for item in content)


def normalize_history_item(message: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw message into runtime history format with `_kind` labels."""

    role = str(message.get("role", "user") or "user")
    content = message.get("content")
    if content in (None, "") and isinstance(message.get("output"), list):
        content = flatten_content(message.get("output"))

    if contains_tool_calls(message) or contains_tool_use_block(content):
        kind = "assistant_tool_use"
    elif role == "tool" or contains_tool_result_block(content):
        kind = "user_tool_result"
    elif message.get("_kind") == "memory_summary":
        kind = "memory_summary"
    elif str(message.get("_kind") or "") in {"internal_compaction_event", "internal_artifact_ref"}:
        kind = str(message.get("_kind") or "")
    elif role == "system":
        kind = "system"
    elif role == "assistant":
        kind = "assistant_text"
    else:
        kind = "user_text"

    normalized: Dict[str, Any] = {
        "role": role,
        "content": content,
        "_kind": kind,
    }
    for key in ["name", "tool_call_id", "tool_calls", "file", "files", "images"]:
        if message.get(key) not in (None, "", []):
            normalized[key] = message.get(key)
    if message.get("_source_name"):
        normalized["_source_name"] = message.get("_source_name")
    logger.debug(
        build_log_message(
            "message",
            "normalize_history_item",
            role=role,
            kind=kind,
        )
    )
    return normalized


def append_if_new(history: List[Dict[str, Any]], message: Dict[str, Any]) -> None:
    """Append only when the message digest differs from the current tail."""

    if not history:
        history.append(message)
        logger.debug(build_log_message("message", "append_if_new", action="append_initial"))
        return
    if message_digest(history[-1]) == message_digest(message):
        logger.debug(build_log_message("message", "append_if_new", action="skip_duplicate"))
        return
    history.append(message)
    logger.debug(build_log_message("message", "append_if_new", action="append"))
