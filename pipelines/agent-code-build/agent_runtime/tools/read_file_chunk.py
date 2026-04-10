"""Adaptive chunk reader for long text attachments.

Computes a safe per-turn chunk size from runtime budgets and tracks progress.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..common import build_log_message, get_logger


logger = get_logger("tool.read_file_chunk")


def _safe_requested_chars(available_chars: int, inline_char_cap: int) -> int:
    """Compute a conservative source-char target for this chunk call."""

    formatted_budget = max(800, int(inline_char_cap or 0))
    # Reserve space for metadata/footer and for line-number expansion so the tool result
    # can usually stay inline in history like read_file_range/search_in_file.
    source_cap = max(800, int(max(800, formatted_budget - 1600) * 0.72))
    if available_chars > 0:
        return max(800, min(int(available_chars), source_cap))
    return source_cap


def _merge_ranges(ranges: List[List[int]], start: int, end: int) -> List[List[int]]:
    """Merge one read range into the historical range list."""

    merged = sorted([*[list(item) for item in ranges if isinstance(item, list) and len(item) == 2], [start, end]])
    if not merged:
        return []
    result: List[List[int]] = [merged[0]]
    for current_start, current_end in merged[1:]:
        last = result[-1]
        if current_start <= last[1] + 1:
            last[1] = max(last[1], current_end)
        else:
            result.append([current_start, current_end])
    return result


def _contiguous_until_start(ranges: List[List[int]]) -> int:
    """Return contiguous line coverage from file start."""

    if not ranges:
        return 0
    merged = sorted(ranges, key=lambda item: (item[0], item[1]))
    if merged[0][0] > 1:
        return 0
    covered = merged[0][1]
    for start, end in merged[1:]:
        if start > covered + 1:
            break
        covered = max(covered, end)
    return covered


def schema() -> Dict[str, Any]:
    """Return schema for budget-aware incremental reading."""

    return {
        "type": "function",
        "function": {
            "name": "read_file_chunk",
            "description": (
                "Read the next safe chunk from a workspace attachment. "
                "The tool automatically sizes the chunk from the remaining token budget of the current turn, "
                "continues from the next unread contiguous line by default, and reports progress plus EOF status. "
                "For whole-file analysis, call it repeatedly until EOF is reached or explicitly scope the answer to the inspected range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Attachment reference. Accepts display name, file_id, workspace path, or attachment filename.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional 1-based starting line. Omit it to continue from the next unread contiguous line.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }


def execute(
    store: Any,
    state: Dict[str, Any],
    path: str,
    start_line: int | None = None,
) -> str:
    """Read one adaptive chunk and persist progress metadata into state."""

    logger.info(
        build_log_message(
            "tool",
            "read_file_chunk",
            chat_id=state.get("chat_id"),
            path=path,
            start_line=start_line,
        )
    )
    target = store.resolve_attachment_path(
        chat_id=state["chat_id"],
        reference=path,
        manifest=state.get("attachment_manifest", {}),
    )
    if target is None:
        return f"Attachment not found: {path}"

    progress_map = state.setdefault("file_read_progress", {})
    progress_key = str(target)
    progress_entry = dict(progress_map.get(progress_key) or {})
    previous_ranges = list(progress_entry.get("ranges", []) or [])
    contiguous_until = _contiguous_until_start(previous_ranges)
    start = max(1, int(start_line)) if start_line is not None else max(1, contiguous_until + 1)

    budget = dict(state.get("dynamic_read_budget") or {})
    available_tokens = max(0, int(budget.get("available_read_tokens") or 0))
    available_chars = max(0, int(budget.get("available_read_chars") or 0))
    inline_char_cap = max(800, int(budget.get("inline_char_cap") or 0))
    requested_chars = _safe_requested_chars(available_chars, inline_char_cap)

    if available_tokens <= 0 and available_chars <= 0:
        return (
            "No safe read budget remains in the current turn after accounting for current context, reserved model output, "
            "and safety margin. Let the next compaction step run, then call read_file_chunk again."
        )

    read_result = store.read_file_chunk(target, start, requested_chars)
    if not read_result.get("supported"):
        return f"File is empty or not supported for chunk reading: {target}"

    total_lines = int(read_result.get("total_lines") or 0)
    if total_lines <= 0:
        return f"File is empty or not supported for chunk reading: {target}"
    if start > total_lines:
        return f"Requested chunk starts after the end of file. Total lines: {total_lines}"

    actual_end = int(read_result.get("actual_end") or 0)
    excerpt = [
        f"{line_number}: {line}"
        for line_number, line in zip(
            range(start, actual_end + 1),
            list(read_result.get("lines") or []),
        )
    ]
    merged_ranges = _merge_ranges(previous_ranges, start, actual_end)
    contiguous_until = _contiguous_until_start(merged_ranges)
    progress_entry.update(
        {
            "display_name": progress_entry.get("display_name") or target.name,
            "workspace_path": str(target),
            "total_lines": total_lines,
            "ranges": merged_ranges,
            "last_read_start": start,
            "last_read_end": actual_end,
            "contiguous_until": contiguous_until,
            "last_chunk_chars": int(read_result.get("used_chars") or 0),
            "last_chunk_token_budget": available_tokens,
        }
    )
    progress_map[progress_key] = progress_entry

    footer = [
        f"[note] Dynamic chunk budget for this turn: about {available_tokens} tokens, capped to about {requested_chars} chars before line-number formatting.",
        f"[note] This chunk covered lines {start}-{actual_end} and used about {int(read_result.get('used_chars') or 0)} chars of source text.",
    ]
    meta_lines = [
        f"[chunk_meta] file={target.name}",
        f"[chunk_meta] chunk_range={start}-{actual_end}",
        f"[chunk_meta] total_lines={total_lines}",
        f"[chunk_meta] contiguous_until={contiguous_until}",
    ]
    if contiguous_until < total_lines:
        next_start = contiguous_until + 1
        footer.append(f"[note] Contiguous coverage is now 1-{contiguous_until} of {total_lines}.")
        footer.append(f"[note] Remaining unread lines after contiguous coverage: {total_lines - contiguous_until}.")
        footer.append(f"[note] Next automatic chunk will start from line {next_start}.")
        footer.append("[note] Continue with read_file_chunk if more evidence is needed, or stop if you already have the key information.")
        meta_lines.extend(
            [
                "[chunk_meta] eof=false",
                f"[chunk_meta] remaining_lines={total_lines - contiguous_until}",
                f"[chunk_meta] next_start={next_start}",
            ]
        )
    else:
        footer.append(f"[note] Reached EOF for {target.name}. Total lines: {total_lines}.")
        meta_lines.extend(
            [
                "[chunk_meta] eof=true",
                "[chunk_meta] remaining_lines=0",
                f"[chunk_meta] next_start={total_lines + 1}",
            ]
        )

    logger.info(
        build_log_message(
            "tool",
            "read_file_chunk_complete",
            path=target,
            returned_lines=len(excerpt),
            total_lines=total_lines,
            requested_chars=requested_chars,
            used_chars=read_result.get("used_chars"),
            eof=contiguous_until >= total_lines,
        )
    )
    return "\n".join(
        [
            *meta_lines,
            f"Read chunk lines {start}-{actual_end} from {target.name}:",
            *excerpt,
            *footer,
        ]
    )
