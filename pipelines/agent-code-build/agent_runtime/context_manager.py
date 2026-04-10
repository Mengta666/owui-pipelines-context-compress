"""Context assembly for each runtime turn.

This module builds system/user-context prompts and coordinates compaction checks.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .common import build_log_message, display_path_name, flatten_content, get_logger


class ContextManager:
    """Builds model-visible messages from runtime session state."""

    def __init__(self, valves: Any, compactor: Any) -> None:
        """Bind configuration and compactor dependencies."""
        self.valves = valves
        self.compactor = compactor
        self.logger = get_logger("context")

    def build_messages(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compose full request messages: system prompt, user context, visible history."""
        messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(state),
                "_kind": "system_prompt",
            },
            {
                "role": "user",
                "content": self.build_user_context(state),
                "_kind": "user_context",
            },
        ]
        messages.extend(
            message for message in (state.get("history", []) or []) if self._is_model_visible(message)
        )
        sanitized = [self._sanitize_for_model(message) for message in messages]
        self.logger.debug(
            build_log_message(
                "context",
                "build_messages",
                chat_id=state.get("chat_id"),
                history_count=len(state.get("history", [])),
                request_message_count=len(sanitized),
            )
        )
        return sanitized

    def build_system_prompt(self, state: Dict[str, Any]) -> str:
        """Render global operating rules for the model in the current turn."""
        attachment_count = len((state.get("attachment_manifest", {}) or {}).get("files", []) or [])
        return "\n".join(
            [
                "You are a tool-using coding and document agent running inside an Open WebUI pipeline.",
                "Rules:",
                "1. Do not guess file contents.",
                "2. If the latest user message already includes native multimodal attachments that your model can inspect directly, inspect them directly before falling back to workspace file tools.",
                "3. If older details were compacted, use read_transcript. If inline prompt history was truncated, use read_prompt_history.",
                "4. Tool results outrank memory summaries when they conflict.",
                "5. Keep answers concise and action-oriented.",
                "6. Answer in the same language as the user unless they request otherwise.",
                "7. Attached files may be available both as native multimodal inputs and through workspace tools. Decide yourself which path fits the current request.",
                "8. Use list_attachments, search_in_file, read_file_chunk, read_file_range, read_prompt_history, and read_transcript when you need workspace lookup, chunked file reading, text search, prompt-history recovery, transcript access, or when native multimodal inspection is unavailable.",
                "9. If a file tool reports that a PDF, image, or other binary file is unsupported for text reading, inspect the original attached file directly instead of guessing from the filename.",
                "10. If attachments are listed for this session, do not claim the user failed to upload or provide files unless tool evidence confirms that.",
                "11. Treat the recent user messages shown in the user context as active constraints for the current turn. Do not let memory summaries override them.",
                "12. When a user asks for full extraction or verbatim coverage, keep reading contiguous ranges until the tool reports EOF or explicitly state that unread content remains.",
                "13. For long text attachments, prefer read_file_chunk. It continues from the next unread contiguous line, auto-sizes the chunk from the current turn budget, and reports whether EOF was reached.",
                "14. If a file-reading tool reports remaining unread lines or shows EOF was not reached, do not present a final comprehensive analysis as if the file were fully covered.",
                "15. Before giving a definitive file-based answer, check the read progress in the user context. If the relevant file is only partially covered, continue reading or clearly scope the answer to the inspected portion.",
                "16. Before finalizing a file-based answer, run this self-check: identify the attachment(s) your answer depends on, inspect their read progress, and if any dependency still shows partial contiguous coverage then continue reading or explicitly say the answer is only based on the inspected ranges.",
                "17. When a tool result includes [chunk_meta] or [range_meta], treat those progress fields as authoritative over any impression from the excerpt body.",
                "18. read_file_chunk may be externalized after the turn: later history may keep only a summary plus progress metadata instead of the original excerpt body.",
                "19. If a chunk or tool result says the full original was saved as a transcript artifact, treat the inline text as lossy. Do not rely on that summary for exact facts, enumerations, quotations, or comprehensive extraction.",
                "20. When exact evidence from an externalized chunk is needed, call read_transcript with the artifact filename shown in the tool result before answering.",
                "21. Never infer omitted source text from a chunk summary, light-compacted replacement, or artifact placeholder. Re-read the transcript artifact or explicitly say the evidence is incomplete.",
                f"Current attachment count: {attachment_count}",
            ]
        )

    def build_user_context(self, state: Dict[str, Any]) -> str:
        """Render session-specific context: files, progress, budgets, prompt history."""
        files = list((state.get("attachment_manifest", {}) or {}).get("files", []) or [])
        read_progress = dict(state.get("file_read_progress", {}) or {})
        transcripts = list(state.get("transcript_paths", []) or [])[-3:]
        summary_present = "yes" if state.get("summary_message") else "no"
        dynamic_read_budget = dict(state.get("dynamic_read_budget") or {})
        prompt_history_payload = self.compactor.store.load_prompt_history(state["chat_id"])
        prompt_history_text = self.compactor.store.render_prompt_history_text(
            state["chat_id"],
            prompt_history_payload,
        ).strip()
        prompt_history_path = self.compactor.store.prompt_history_path(state["chat_id"])
        file_lines = [
            f"- {item.get('display_name', '')} -> {item.get('workspace_path', '')} (mode={item.get('storage_mode', 'copy')})"
            for item in files
        ]
        progress_lines = []
        for item in list(read_progress.values())[-6:]:
            if not isinstance(item, dict):
                continue
            display_name = str(item.get("display_name") or "").strip()
            total_lines = int(item.get("total_lines") or 0)
            contiguous_until = int(item.get("contiguous_until") or 0)
            last_start = int(item.get("last_read_start") or 0)
            last_end = int(item.get("last_read_end") or 0)
            if not display_name:
                continue
            if total_lines > 0 and contiguous_until < total_lines:
                remaining_lines = total_lines - contiguous_until
                if contiguous_until > 0:
                    progress_lines.append(
                        f"- {display_name}: last read {last_start}-{last_end}; contiguous coverage 1-{contiguous_until}/{total_lines}; remaining {remaining_lines} lines; next suggested range {contiguous_until + 1}-{min(total_lines, contiguous_until + max(last_end - last_start + 1, 100))}; read_file_chunk will continue from line {contiguous_until + 1}"
                    )
                else:
                    progress_lines.append(
                        f"- {display_name}: last read {last_start}-{last_end}; no contiguous coverage from the start yet; remaining {remaining_lines} lines; next suggested range 1-{min(total_lines, max(last_end - last_start + 1, 100))}; read_file_chunk can restart from line 1"
                    )
            elif total_lines > 0:
                progress_lines.append(
                    f"- {display_name}: last read {last_start}-{last_end}; file fully covered 1-{total_lines}/{total_lines}"
                )
        transcript_lines = [f"- {display_path_name(path)}" for path in transcripts]
        read_budget_lines = []
        available_read_tokens = int(dynamic_read_budget.get("available_read_tokens") or 0)
        available_read_chars = int(dynamic_read_budget.get("available_read_chars") or 0)
        if dynamic_read_budget:
            read_budget_lines.append(
                f"- Current automatic chunk budget: about {available_read_tokens} tokens / {available_read_chars} chars before formatting."
            )
            read_budget_lines.append(
                f"- Safety margin already reserves about {int(dynamic_read_budget.get('safety_tokens') or 0)} tokens beyond the configured output budget."
            )

        base_lines = [
            f"Workspace: {self.valves.WORKSPACE_ROOT}/sessions/{state['chat_id']}",
            f"Summary available: {summary_present}",
            f"Permission mode: {state.get('permission_mode', 'auto')}",
            "",
            "Attachments:",
            *(file_lines or ["- (none)"]),
            "",
            "Recent transcripts:",
            *(transcript_lines or ["- (none)"]),
            "",
            "Recent file read progress:",
            *(progress_lines or ["- (none)"]),
            "",
            "Current chunk-read budget:",
            *(read_budget_lines or ["- Not computed yet for this turn."]),
            "",
            "Do not ask the user to paste file contents.",
            "If the latest user message includes native multimodal attachments and your model can inspect them directly, use those attachments directly.",
            "Attached files are also registered in the workspace, but their contents are not preloaded into the prompt.",
            "Use file tools when the current request needs workspace text lookup, transcript access, prompt-history recovery, or when native multimodal inspection is unavailable.",
            "For long text attachments, prefer read_file_chunk over large manual read_file_range calls.",
            "If a workspace file tool cannot parse a PDF or image as text, fall back to the original attachment rather than inferring from the filename.",
            "When extracting a long text file across multiple tool calls, continue from the next unread range instead of rereading overlapping ranges.",
            "read_file_chunk automatically computes the next safe chunk from the current turn budget and reports whether the file is fully read.",
            "You control the requested range size in read_file_range. If you request beyond EOF, the tool will return the remaining lines and tell you that EOF was reached.",
            "If the tool says there are remaining unread lines, do not infer them. Continue reading or explicitly say the file is only partially inspected.",
            "If the read progress says a file is fully covered, answer from the gathered evidence instead of reading earlier ranges again.",
            "If the read progress does not show full coverage for a relevant file, do not give a final all-file conclusion without either continuing to read or stating that the answer is based only on the inspected portion.",
            "If a read_file_chunk or read_file_range result shows [chunk_meta] or [range_meta] with eof=false, treat that file as still partially inspected even if the excerpt already looks representative.",
            "Do not say attachments are missing when they are listed above.",
            "Some oversized tool results may be stored as session artifacts instead of being fully inlined into history.",
            "If a tool result says its full content was saved as a transcript artifact, use read_transcript with that artifact path when exact details are needed.",
            "If an earlier read_file_chunk result was externalized, the later visible history may contain only a summary placeholder rather than the original excerpt body.",
            "Do not treat a chunk summary placeholder as equivalent to the original source text. For exact details, lists, counts, quotations, or full-file claims, call read_transcript on the named artifact first.",
            "If you cannot or did not re-read the transcript artifact, explicitly scope the answer to the retained summary rather than presenting it as source-complete.",
        ]
        base_text = "\n".join(base_lines)
        prompt_history_limit = max(1000, int(getattr(self.valves, "PROMPT_HISTORY_MAX_CHARS", 10000) or 10000))
        total_ratio = max(1.0, float(getattr(self.valves, "PROMPT_HISTORY_TOTAL_RATIO", 1.5) or 1.5))
        history_read_max_chars = max(200, int(prompt_history_limit * max(0.0, total_ratio - 1.0)))
        history_note_lines = [
            "",
            "Prompt history excerpt (treat it as active user constraints for the current turn):",
            "",
        ]
        truncated_note_lines = [
            "",
            f"[note] Complete prompt history is stored outside the inline user_context at: {prompt_history_path.name}",
            f"[note] Inline prompt-history excerpt is limited to {prompt_history_limit} chars.",
            f"[note] If older prompt details are needed, call read_prompt_history. It returns the latest omitted segment before the current inline prompt-history window, capped to {history_read_max_chars} chars so inline prompt history plus recovered history stays within about {total_ratio:.2f}x the inline prompt-history budget.",
            "",
        ]
        prompt_history_inline = prompt_history_text
        inline_start_char = 0
        omitted_chars = 0
        if prompt_history_text and len(prompt_history_text) > prompt_history_limit:
            inline_start_char = len(prompt_history_text) - prompt_history_limit
            prompt_history_inline = prompt_history_text[inline_start_char:]
            omitted_chars = inline_start_char
        elif not prompt_history_text:
            prompt_history_inline = "(none)"

        user_context_lines = [base_text, *history_note_lines, prompt_history_inline]
        if omitted_chars > 0:
            user_context_lines.extend(truncated_note_lines)
        rendered = "\n".join(part for part in user_context_lines if part is not None).strip()
        state["prompt_history_state"] = {
            "path": str(prompt_history_path),
            "full_chars": len(prompt_history_text),
            "inline_chars": len(prompt_history_inline) if prompt_history_inline != "(none)" else 0,
            "inline_start_char": inline_start_char,
            "omitted_chars": omitted_chars,
            "prompt_history_max_chars": prompt_history_limit,
            "history_read_max_chars": history_read_max_chars,
            "total_ratio": total_ratio,
        }
        return rendered

    def _recent_user_request_texts(self, state: Dict[str, Any]) -> List[str]:
        """Collect recent user textual constraints from history tail."""
        recent: List[str] = []
        for item in reversed(state.get("history", []) or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("role", "user") or "user") != "user":
                continue
            if str(item.get("_kind") or "") not in {"", "user_text"}:
                continue
            text = flatten_content(item.get("content")).strip()
            if text:
                recent.append(f"- {text}")
            if len(recent) >= max(1, int(getattr(self.valves, "KEEP_LAST_USER_MESSAGES", 6) or 6)):
                break
        recent.reverse()
        return recent

    def prepare_turn(self, state: Dict[str, Any], request_model: str = "") -> Dict[str, Any]:
        """Apply light/full compaction when needed before model invocation."""
        prompt_tokens = self.compactor.count_messages(
            [
                self._sanitize_for_model(
                    {
                        "role": "system",
                        "content": self.build_system_prompt(state),
                        "_kind": "system_prompt",
                    }
                ),
                self._sanitize_for_model(
                    {
                        "role": "user",
                        "content": self.build_user_context(state),
                        "_kind": "user_context",
                    }
                ),
            ]
        )
        state, lightweight_report = self.compactor.apply_lightcompact(
            state=state,
            prompt_tokens=prompt_tokens,
            request_model=request_model,
        )
        report = {
            "prompt_tokens": prompt_tokens,
            **lightweight_report,
            "full_compaction_applied": False,
        }
        if self.needs_compaction(state):
            self.logger.info(
                build_log_message(
                    "context",
                    "prepare_turn_full_compaction",
                    chat_id=state.get("chat_id"),
                    request_model=request_model,
                )
            )
            state = self.compact_history(state, request_model=request_model)
            report["full_compaction_applied"] = True
            report["full_compaction_reason"] = "threshold_exceeded"
            report["token_after"] = self.compactor.count_messages(self.build_messages(state))
            report["artifact_ref_count"] = sum(
                1 for item in (state.get("history", []) or []) if str(item.get("_kind") or "") == "internal_artifact_ref"
            )
        state["last_prepare_turn_report"] = report
        return state

    def needs_compaction(self, state: Dict[str, Any]) -> bool:
        """Check whether total token estimate exceeds compaction threshold."""
        if not self.valves.ENABLE_HISTORY_COMPACTION:
            self.logger.debug(build_log_message("context", "needs_compaction_disabled"))
            return False
        messages = self.build_messages(state)
        current_tokens = self.compactor.count_messages(messages)
        threshold = self.compactor.compact_threshold()
        needs = current_tokens >= threshold
        self.logger.info(
            build_log_message(
                "context",
                "needs_compaction",
                chat_id=state.get("chat_id"),
                current_tokens=current_tokens,
                threshold=threshold,
                needs=needs,
            )
        )
        return needs

    def compact_history(self, state: Dict[str, Any], request_model: str = "") -> Dict[str, Any]:
        """Run full history compaction and return updated state."""
        prompt_tokens = self.compactor.count_messages(
            [
                self._sanitize_for_model(
                    {
                        "role": "system",
                        "content": self.build_system_prompt(state),
                        "_kind": "system_prompt",
                    }
                ),
                self._sanitize_for_model(
                    {
                        "role": "user",
                        "content": self.build_user_context(state),
                        "_kind": "user_context",
                    }
                ),
            ]
        )
        self.logger.info(
            build_log_message(
                "context",
                "compact_history",
                chat_id=state.get("chat_id"),
                prompt_tokens=prompt_tokens,
                request_model=request_model,
            )
        )
        return self.compactor.compact_if_needed(
            state=state,
            prompt_tokens=prompt_tokens,
            request_model=request_model,
        )

    def _is_model_visible(self, message: Dict[str, Any]) -> bool:
        """Filter out internal runtime bookkeeping messages from model input."""
        kind = str((message or {}).get("_kind") or "")
        return kind not in {"internal_compaction_event", "internal_artifact_ref"}

    def _sanitize_for_model(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Drop runtime-only keys and keep provider-compatible fields."""
        sanitized: Dict[str, Any] = {
            "role": message.get("role", "user"),
            "content": message.get("content", ""),
        }
        for key in ["name", "tool_call_id", "tool_calls", "file", "files", "images"]:
            if message.get(key) not in (None, "", []):
                sanitized[key] = message.get(key)
        if sanitized["content"] is None:
            sanitized["content"] = ""
        return sanitized
