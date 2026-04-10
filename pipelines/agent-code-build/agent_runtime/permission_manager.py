"""Permission policy for runtime tool calls.

Current policy intentionally allows only read-only tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .common import build_log_message, get_logger


logger = get_logger("permission")


@dataclass
class AllowDecision:
    """Result returned by permission checks for one tool call."""

    allowed: bool
    reason: str


class PermissionManager:
    """Enforces a read-only tool whitelist."""

    READ_ONLY_TOOLS = {
        "list_attachments",
        "search_in_file",
        "read_file_chunk",
        "read_file_range",
        "read_prompt_history",
        "read_transcript",
    }

    def check(self, tool_call: Dict[str, Any], state: Dict[str, Any]) -> AllowDecision:
        """Validate whether the requested tool may run for this session."""

        mode = str(state.get("permission_mode", "auto") or "auto")
        name = str(tool_call.get("name", "") or "")
        if name in self.READ_ONLY_TOOLS:
            logger.info(build_log_message("permission", "allow", tool_name=name, mode=mode, reason="read-only tool"))
            return AllowDecision(True, "read-only tool")
        if mode == "strict":
            logger.warning(build_log_message("permission", "deny", tool_name=name, mode=mode, reason="strict mode"))
            return AllowDecision(False, "tool denied by strict mode")
        logger.warning(build_log_message("permission", "deny", tool_name=name, mode=mode, reason="tool disabled in v1"))
        return AllowDecision(False, "tool disabled in v1")
