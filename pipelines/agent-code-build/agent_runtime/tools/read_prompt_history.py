"""Recover omitted prompt-history segments from persisted storage."""

from __future__ import annotations

from typing import Any, Dict

from ..common import build_log_message, get_logger


logger = get_logger("tool.read_prompt_history")


def schema() -> Dict[str, Any]:
    """Return schema for controlled prompt-history recovery requests."""

    return {
        "type": "function",
        "function": {
            "name": "read_prompt_history",
            "description": "Read the most recent omitted portion of prompt history for the current chat session. This is for prompt history that was saved outside the inline user_context budget.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum number of omitted prompt-history characters to return. The tool caps this automatically to keep total context growth bounded.",
                        "default": 0,
                    }
                },
                "additionalProperties": False,
            },
        },
    }


def execute(store: Any, state: Dict[str, Any], max_chars: int = 0) -> str:
    """Return the latest omitted segment just before the inline prompt-history window."""

    snapshot = dict(state.get("prompt_history_state") or {})
    payload = store.load_prompt_history(state["chat_id"])
    rendered = store.render_prompt_history_text(state["chat_id"], payload)
    if not rendered.strip():
        return "No prompt history is stored for this chat session."

    omitted_chars = max(0, int(snapshot.get("omitted_chars") or 0))
    inline_start_char = max(0, int(snapshot.get("inline_start_char") or 0))
    history_budget = max(200, int(snapshot.get("history_read_max_chars") or 2000))
    requested = max(0, int(max_chars or 0))
    effective_max = min(history_budget, requested) if requested > 0 else history_budget

    logger.info(
        build_log_message(
            "tool",
            "read_prompt_history",
            chat_id=state.get("chat_id"),
            omitted_chars=omitted_chars,
            inline_start_char=inline_start_char,
            effective_max=effective_max,
        )
    )

    if omitted_chars <= 0 or inline_start_char <= 0:
        return "Inline user_context already includes the available prompt history. No omitted prompt-history segment needs to be read."

    segment_start = max(0, inline_start_char - effective_max)
    segment_end = inline_start_char
    segment = rendered[segment_start:segment_end].strip()
    if not segment:
        return "No omitted prompt-history segment is available to read."

    prefix = []
    if segment_start > 0:
        prefix.append("[note] Older prompt-history content exists before this returned segment.")
    prefix.append(
        f"[note] Returned the latest omitted prompt-history segment before the inline user_context window ({len(segment)} chars, capped at {effective_max})."
    )
    prefix.append(
        f"[note] Inline prompt-history budget: {int(snapshot.get('prompt_history_max_chars') or 0)} chars; additional prompt-history read budget: {history_budget} chars."
    )
    return "\n".join([*prefix, "", segment]).strip()
