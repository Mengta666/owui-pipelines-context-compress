"""Read compacted transcripts or stored artifacts for a session."""

from __future__ import annotations

from typing import Any, Dict

from ..common import build_log_message, get_logger


logger = get_logger("tool.read_transcript")


def schema() -> Dict[str, Any]:
    """Return schema for transcript/artifact retrieval."""

    return {
        "type": "function",
        "function": {
            "name": "read_transcript",
            "description": "Read a compacted transcript or a stored session text artifact for the current chat session. If path is omitted, read the latest compaction transcript.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Transcript or artifact filename, or a full session transcript/artifact path. Optional.",
                    }
                },
                "additionalProperties": False,
            },
        },
    }


def execute(store: Any, state: Dict[str, Any], path: str = "") -> str:
    """Read transcript text with truncation note when output exceeds budget."""

    logger.info(build_log_message("tool", "read_transcript", chat_id=state.get("chat_id"), path=path or "(latest)"))
    target = store.resolve_transcript_path(state["chat_id"], path)
    if target is None:
        logger.warning(build_log_message("tool", "read_transcript_missing", chat_id=state.get("chat_id"), path=path or "(latest)"))
        return "No transcript is available for this chat session."

    text, truncated = store.read_text_file(target, max_chars=store.max_transcript_chars)
    if not text:
        logger.warning(build_log_message("tool", "read_transcript_unreadable", path=target))
        return f"Transcript is empty or unreadable: {target.name}"
    note = ""
    if truncated:
        note = f"\n\n[note] Transcript output was truncated to the first {store.max_transcript_chars} characters."
    logger.info(build_log_message("tool", "read_transcript_complete", path=target, truncated=truncated))
    return f"Transcript: {target}\n\n{text}{note}"
