"""Registry for runtime-exposed read-only tools."""

from __future__ import annotations

from typing import Any, Dict, List

from .common import build_log_message, get_logger
from .tools import (
    list_attachments_execute,
    list_attachments_schema,
    read_file_chunk_execute,
    read_file_chunk_schema,
    read_file_range_execute,
    read_file_range_schema,
    read_prompt_history_execute,
    read_prompt_history_schema,
    read_transcript_execute,
    read_transcript_schema,
    search_in_file_execute,
    search_in_file_schema,
)


class ToolRegistry:
    """Provides tool schemas and dispatches tool execution by name."""

    def __init__(self, store: Any) -> None:
        """Bind workspace store and initialize tool executor map."""

        self.store = store
        self.logger = get_logger("tool_registry")
        self._tool_map = {
            "list_attachments": list_attachments_execute,
            "search_in_file": search_in_file_execute,
            "read_file_chunk": read_file_chunk_execute,
            "read_file_range": read_file_range_execute,
            "read_prompt_history": read_prompt_history_execute,
            "read_transcript": read_transcript_execute,
        }

    def tool_schemas(self) -> List[Dict[str, Any]]:
        """Return all tool schemas advertised to the model."""

        schemas = [
            list_attachments_schema(),
            search_in_file_schema(),
            read_file_chunk_schema(),
            read_file_range_schema(),
            read_prompt_history_schema(),
            read_transcript_schema(),
        ]
        self.logger.debug(build_log_message("tool_registry", "tool_schemas", schema_count=len(schemas)))
        return schemas

    def execute(
        self, tool_call: Dict[str, Any], state: Dict[str, Any], decision: Any
    ) -> Dict[str, Any]:
        """Execute one tool call and wrap output as a history-compatible tool message."""

        tool_use_id = str(tool_call.get("id") or "")
        name = str(tool_call.get("name") or "")
        self.logger.info(
            build_log_message(
                "tool_registry",
                "execute_start",
                tool_name=name,
                tool_use_id=tool_use_id,
                allowed=decision.allowed,
            )
        )
        if not decision.allowed:
            self.logger.warning(build_log_message("tool_registry", "execute_denied", tool_name=name, reason=decision.reason))
            return {
                "role": "tool",
                "tool_call_id": tool_use_id,
                "name": name,
                "content": f"Permission denied: {decision.reason}",
                "_kind": "user_tool_result",
            }

        executor = self._tool_map.get(name)
        if executor is None:
            self.logger.warning(build_log_message("tool_registry", "execute_unknown", tool_name=name))
            return {
                "role": "tool",
                "tool_call_id": tool_use_id,
                "name": name,
                "content": f"Unknown tool: {name}",
                "_kind": "user_tool_result",
            }

        try:
            result = executor(self.store, state, **dict(tool_call.get("input", {}) or {}))
        except Exception as exc:
            self.logger.exception(
                build_log_message(
                    "tool_registry",
                    "execute_failed",
                    tool_name=name,
                    error_type=type(exc).__name__,
                )
            )
            result = f"Tool execution failed: {type(exc).__name__}: {exc}"
        else:
            self.logger.info(
                build_log_message(
                    "tool_registry",
                    "execute_complete",
                    tool_name=name,
                    result_preview=str(result)[:160],
                )
            )

        return {
            "role": "tool",
            "tool_call_id": tool_use_id,
            "name": name,
            "content": result,
            "_kind": "user_tool_result",
        }
