"""Line-range reader for workspace attachments."""

from __future__ import annotations

from typing import Any, Dict, List

from ..common import build_log_message, get_logger


logger = get_logger("tool.read_file_range")


def _merge_ranges(ranges: List[List[int]], start: int, end: int) -> List[List[int]]:
    """Merge a new range into existing read ranges."""

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
    """Return contiguous coverage from line 1 based on merged ranges."""

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
    """Return schema for explicit line-range reads."""

    return {
        "type": "function",
        "function": {
            "name": "read_file_range",
            "description": (
                "Read a numbered line range from a workspace attachment. "
                "If end_line exceeds the file length, the tool automatically returns the remaining lines up to EOF "
                "and reports the unread remainder or EOF status."
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
                        "description": "1-based starting line number.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "1-based ending line number.",
                    },
                },
                "required": ["path", "start_line", "end_line"],
                "additionalProperties": False,
            },
        },
    }


def execute(
    store: Any,
    state: Dict[str, Any],
    path: str,
    start_line: int,
    end_line: int,
) -> str:
    """Read one explicit line range and update per-file read progress."""

    logger.info(
        build_log_message(
            "tool",
            "read_file_range",
            chat_id=state.get("chat_id"),
            path=path,
            start_line=start_line,
            end_line=end_line,
        )
    )
    target = store.resolve_attachment_path(
        chat_id=state["chat_id"],
        reference=path,
        manifest=state.get("attachment_manifest", {}),
    )
    if target is None:
        return f"Attachment not found: {path}"

    read_result = store.read_file_line_range(target, start_line, end_line)
    if not read_result.get("supported"):
        return f"File is empty or not supported for line reading: {target}"

    start = max(1, int(start_line))
    end = max(start, int(end_line))
    total_lines = int(read_result.get("total_lines") or 0)
    if total_lines <= 0:
        return f"File is empty or not supported for line reading: {target}"
    if start > total_lines:
        logger.warning(build_log_message("tool", "read_file_range_oob", path=target, total_lines=total_lines, start_line=start))
        return f"Requested range starts after the end of file. Total lines: {total_lines}"

    actual_end = int(read_result.get("actual_end") or 0)
    excerpt = [
        f"{line_number}: {line}"
        for line_number, line in zip(
            range(start, actual_end + 1),
            list(read_result.get("lines") or []),
        )
    ]

    progress_map = state.setdefault("file_read_progress", {})
    progress_key = str(target)
    progress_entry = dict(progress_map.get(progress_key) or {})
    merged_ranges = _merge_ranges(list(progress_entry.get("ranges", []) or []), start, actual_end)
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
        }
    )
    progress_map[progress_key] = progress_entry

    footer = []
    meta_lines = [
        f"[range_meta] file={target.name}",
        f"[range_meta] requested_range={start}-{end}",
        f"[range_meta] returned_range={start}-{actual_end}",
        f"[range_meta] total_lines={total_lines}",
        f"[range_meta] contiguous_until={contiguous_until}",
    ]
    if end > total_lines:
        footer.append(
            f"[note] Requested end_line {end} exceeded the file length {total_lines}. Returned the remaining lines up to EOF."
        )
    if contiguous_until < total_lines:
        suggested_end = min(contiguous_until + max(actual_end - start + 1, 100), total_lines)
        remaining_lines = total_lines - contiguous_until
        meta_lines.extend(
            [
                "[range_meta] eof=false",
                f"[range_meta] remaining_lines={remaining_lines}",
                f"[range_meta] next_start={contiguous_until + 1}",
                f"[range_meta] next_end={suggested_end}",
            ]
        )
        if contiguous_until > 0:
            footer.append(
                f"[note] Read progress for {target.name}: contiguous lines 1-{contiguous_until} of {total_lines}."
            )
        else:
            footer.append(
                f"[note] Read progress for {target.name}: no contiguous coverage from the start of the file yet."
            )
        footer.append(
            f"[note] Remaining unread lines after the current contiguous coverage: {remaining_lines}."
        )
        footer.append(
            f"[note] Suggested next range if you need the remaining content: {contiguous_until + 1}-{suggested_end}."
        )
    else:
        meta_lines.extend(
            [
                "[range_meta] eof=true",
                "[range_meta] remaining_lines=0",
                f"[range_meta] next_start={total_lines + 1}",
                f"[range_meta] next_end={total_lines + 1}",
            ]
        )
        footer.append(f"[note] Reached the end of {target.name}. Total lines: {total_lines}.")
    logger.info(build_log_message("tool", "read_file_range_complete", path=target, returned_lines=len(excerpt), total_lines=total_lines))
    return "\n".join(
        [
            *meta_lines,
            f"Read lines {start}-{actual_end} from {target.name}:",
            *excerpt,
            *footer,
        ]
    )
