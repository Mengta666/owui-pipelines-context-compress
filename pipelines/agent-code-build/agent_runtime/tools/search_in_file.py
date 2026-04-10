"""Substring search for text-compatible attachments."""

from __future__ import annotations

from typing import Any, Dict, List

from ..common import build_log_message, get_logger


logger = get_logger("tool.search_in_file")


def schema() -> Dict[str, Any]:
    """Return JSON schema for tool invocation parameters."""

    return {
        "type": "function",
        "function": {
            "name": "search_in_file",
            "description": "Search a workspace attachment by exact substring and return matching line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Attachment reference. Accepts display name, file_id, workspace path, or attachment filename.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Substring to search for.",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether the match is case-sensitive.",
                        "default": False,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of hits to return.",
                        "default": 20,
                    },
                },
                "required": ["path", "query"],
                "additionalProperties": False,
            },
        },
    }


def execute(
    store: Any,
    state: Dict[str, Any],
    path: str,
    query: str,
    case_sensitive: bool = False,
    max_results: int = 20,
) -> str:
    """Resolve attachment path and return matching lines with line numbers."""

    logger.info(
        build_log_message(
            "tool",
            "search_in_file",
            chat_id=state.get("chat_id"),
            path=path,
            query=query,
            case_sensitive=case_sensitive,
            max_results=max_results,
        )
    )
    target = store.resolve_attachment_path(
        chat_id=state["chat_id"],
        reference=path,
        manifest=state.get("attachment_manifest", {}),
    )
    if target is None:
        return f"Attachment not found: {path}"

    search_result = store.search_text_file(
        target,
        query=query,
        case_sensitive=case_sensitive,
        max_results=max_results,
    )
    if not search_result.get("supported"):
        return f"File is empty or not supported for text search: {target}"

    results = list(search_result.get("results") or [])

    if not results:
        logger.info(build_log_message("tool", "search_in_file_no_match", path=path, query=query))
        return f"No matches found in {target.name} for query: {query}"

    header = [f"Search results for {query!r} in {target.name}:", *results]
    if search_result.get("hit_limit_reached"):
        header.append(
            f"[note] Returned the first {len(results)} matches. More matches may exist in the file."
        )
    logger.info(
        build_log_message(
            "tool",
            "search_in_file_match",
            path=target,
            match_count=len(results),
            hit_limit_reached=bool(search_result.get("hit_limit_reached")),
        )
    )
    return "\n".join(header)
