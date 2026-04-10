"""List registered workspace attachments for the current chat."""

from __future__ import annotations

from typing import Any, Dict

from ..common import build_log_message, get_logger


logger = get_logger("tool.list_attachments")


def schema() -> Dict[str, Any]:
    """Return the function-call schema exposed to the model."""

    return {
        "type": "function",
        "function": {
            "name": "list_attachments",
            "description": "List the files that are registered into the workspace for the current chat session.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }


def execute(store: Any, state: Dict[str, Any], **_: Any) -> str:
    """Render a human-readable attachment manifest summary."""

    files = list((state.get("attachment_manifest", {}) or {}).get("files", []) or [])
    logger.info(build_log_message("tool", "list_attachments", chat_id=state.get("chat_id"), file_count=len(files)))
    if not files:
        return "No attachments are registered in the workspace."

    lines = ["Workspace attachments:"]
    for item in files:
        lines.append(
            "- {name} | file_id={file_id} | type={content_type} | reference={reference}".format(
                name=item.get("display_name", ""),
                file_id=item.get("file_id", ""),
                content_type=item.get("content_type", "") or "unknown",
                reference=item.get("display_name", "") or item.get("file_id", ""),
            )
        )
    lines.append("Use the display name or file_id as the tool path/reference. Do not prefer raw absolute paths.")
    return "\n".join(lines)
