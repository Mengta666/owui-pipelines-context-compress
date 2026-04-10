"""Shared utility helpers used across the agent runtime.

Includes logging helpers, content flattening, message rendering, and hashing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List


logger = logging.getLogger("mini_agent_code_build")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def set_debug(enabled: bool) -> None:
    """Switch runtime logger verbosity between INFO and DEBUG."""

    logger.setLevel(logging.DEBUG if enabled else logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the shared runtime logger namespace."""

    return logger.getChild(str(name or "runtime"))


def now_ts() -> int:
    """Return current Unix timestamp in seconds."""

    return int(time.time())


def ensure_dir(path: Path) -> None:
    """Create the directory tree when missing."""

    path.mkdir(parents=True, exist_ok=True)


def sha1_text(text: str) -> str:
    """Compute SHA1 digest for deterministic lightweight IDs."""

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def flatten_content(content: Any) -> str:
    """Convert provider-specific content structures into plain text."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = str(item.get("type", "content"))
                if item_type in {"text", "input_text", "output_text"}:
                    parts.append(str(item.get("text", "")))
                elif item_type == "image_url":
                    parts.append("[image_url]")
                else:
                    parts.append(f"[{item_type}]")
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


def serialize_tool_calls(tool_calls: Any) -> str:
    """Serialize tool call arrays to a stable JSON preview string."""

    if not tool_calls:
        return ""
    try:
        return json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(tool_calls)


def trim_text(text: str, head_chars: int, tail_chars: int) -> str:
    """Trim long text while preserving head/tail signal."""

    if len(text) <= head_chars + tail_chars + 64:
        return text
    return text[:head_chars] + "\n\n[...middle content omitted...]\n\n" + text[-tail_chars:]


def display_path_name(path_str: str) -> str:
    """Return basename only for display-safe path logging."""

    return Path(path_str).name if path_str else ""


def render_message(message: Dict[str, Any]) -> str:
    """Render one history item into transcript-friendly text."""

    parts = [f"kind={message.get('_kind', 'chat_message')}", f"role={message.get('role', 'user')}"]
    if message.get("name"):
        parts.append(f"name={message['name']}")
    if message.get("tool_call_id"):
        parts.append(f"tool_call_id={message['tool_call_id']}")
    if message.get("_source_name"):
        parts.append(f"source={message['_source_name']}")
    tool_calls = serialize_tool_calls(message.get("tool_calls"))
    if tool_calls:
        parts.append("tool_calls=" + tool_calls)
    content = flatten_content(message.get("content"))
    if content:
        parts.append(content)
    return "\n".join(parts)


def message_digest(message: Dict[str, Any]) -> str:
    """Build a hash digest used for de-duplication of history items."""

    payload = {
        "role": message.get("role"),
        "content": message.get("content"),
        "name": message.get("name"),
        "tool_call_id": message.get("tool_call_id"),
        "tool_calls": message.get("tool_calls"),
        "file": message.get("file"),
        "files": message.get("files"),
        "images": message.get("images"),
        "_kind": message.get("_kind"),
    }
    try:
        return sha1_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    except Exception:
        return sha1_text(str(payload))


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace for compact logging output."""

    return re.sub(r"\s+", " ", str(text or "")).strip()


def format_log_value(value: Any) -> str:
    """Format arbitrary values into concise single-token log fields."""

    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        text = normalize_whitespace(value)
        if not text:
            return '""'
        if len(text) > 160:
            text = text[:157] + "..."
        if any(ch.isspace() for ch in text) or any(ch in text for ch in {"=", "[", "]", "{", "}", ","}):
            return json.dumps(text, ensure_ascii=False)
        return text
    if isinstance(value, dict):
        preview = ",".join(str(key) for key in list(value.keys())[:4])
        return f"<dict:{len(value)} keys={preview}>"
    if isinstance(value, (list, tuple, set)):
        return f"<{type(value).__name__}:{len(value)}>"
    return format_log_value(str(value))


def build_log_message(domain: str, event: str, **fields: Any) -> str:
    """Build structured key=value log lines with a stable prefix."""

    parts = [f"[{domain}.{event}]"]
    for key, value in fields.items():
        if value is ...:
            continue
        parts.append(f"{key}={format_log_value(value)}")
    return " ".join(parts)
