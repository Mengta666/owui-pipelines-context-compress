"""History compaction for long-running sessions.

Includes light compaction, summary compaction, and hard-limit trimming.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import tiktoken
except Exception:
    tiktoken = None

from .common import build_log_message, display_path_name, flatten_content, get_logger, message_digest, now_ts, render_message, trim_text


class TokenCounter:
    """Token estimator with optional tiktoken backend."""

    def __init__(self, encoding_name: str) -> None:
        """Load the preferred tokenizer encoding with fallbacks."""
        self.encoding = self._load_encoding(encoding_name)

    def _load_encoding(self, encoding_name: str):
        if tiktoken is None:
            return None
        for name in [encoding_name, "o200k_base", "cl100k_base"]:
            if not name:
                continue
            try:
                return tiktoken.get_encoding(name)
            except Exception:
                continue
        return None

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self.encoding is None:
            return max(1, len(text) // 2)
        return len(self.encoding.encode(text))

    def count_message(self, message: Dict[str, Any]) -> int:
        return (
            6
            + self.count_text(str(message.get("role", "user")))
            + self.count_text(str(message.get("name", "")))
            + self.count_text(str(message.get("tool_call_id", "")))
            + self.count_text(flatten_content(message.get("content")))
            + self.count_text(json.dumps(message.get("tool_calls", []), ensure_ascii=False))
        )

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Return aggregate token estimate for message list."""
        return sum(self.count_message(message) for message in messages)


class HistoryCompactor:
    """Coordinates context-size governance for runtime conversation history."""

    def __init__(self, valves: Any, store: Any, provider: Any) -> None:
        """Bind configuration, storage, and summarization provider."""
        self.valves = valves
        self.store = store
        self.provider = provider
        self.token_counter = TokenCounter(valves.TOKENIZER_ENCODING)
        self.logger = get_logger("compactor")

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        return self.token_counter.count_messages(messages)

    def _is_internal_message(self, message: Dict[str, Any]) -> bool:
        kind = str((message or {}).get("_kind") or "")
        return kind in {"internal_compaction_event", "internal_artifact_ref"}

    def _visible_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [item for item in (history or []) if not self._is_internal_message(item)]

    def effective_context_window(self) -> int:
        return max(1024, self.valves.MAX_CONTEXT_TOKENS - self.valves.RESERVED_OUTPUT_TOKENS)

    def compact_threshold(self) -> int:
        return max(512, self.effective_context_window() - self.valves.AUTOCOMPACT_BUFFER_TOKENS)

    def lightcompact_threshold(self) -> int:
        return max(512, self.effective_context_window() - self.valves.LIGHTCOMPACT_BUFFER_TOKENS)

    def lightcompact_min_source_chars(self) -> int:
        max_summary_lines = max(3, int(getattr(self.valves, "LIGHTCOMPACT_MAX_SUMMARY_LINES", 6) or 6))
        line_chars = max(60, int(getattr(self.valves, "LIGHTCOMPACT_LINE_CHARS", 180) or 180))
        return max(400, max_summary_lines * line_chars)

    def apply_lightcompact(
        self,
        state: Dict[str, Any],
        prompt_tokens: int = 0,
        request_model: str = "",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Externalize oversized historical messages while preserving progress hints."""
        history = list(state.get("history", []) or [])
        visible_history = self._visible_history(history)
        report = {
            "lightcompact_applied": False,
            "token_before": prompt_tokens + self.token_counter.count_messages(visible_history),
            "token_after": prompt_tokens + self.token_counter.count_messages(visible_history),
            "artifact_ref_count": sum(
                1 for item in history if str(item.get("_kind") or "") == "internal_artifact_ref"
            ),
        }
        if not history:
            return state, report

        snapshot = state.get("compaction_snapshot") or self.store.load_snapshot(state["chat_id"])
        total_tokens = report["token_before"]
        self.logger.info(
            build_log_message(
                "compactor",
                "lightcompact_evaluate",
                chat_id=state.get("chat_id"),
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
                lightcompact_threshold=self.lightcompact_threshold(),
                request_model=request_model,
            )
        )

        changed = False
        if total_tokens >= self.lightcompact_threshold():
            history, compacted = self._lightcompact_history(state["chat_id"], history)
            changed = changed or compacted
            report["lightcompact_applied"] = bool(compacted)
            total_tokens = prompt_tokens + self.token_counter.count_messages(self._visible_history(history))

        if changed:
            snapshot = dict(snapshot)
            snapshot["last_effective_tokens"] = total_tokens
            snapshot["last_compaction_reason"] = "lightcompact"
            snapshot["updated_at"] = now_ts()
            state["history"] = history
            state["compaction_snapshot"] = snapshot
            self.store.save_snapshot(state["chat_id"], snapshot)
            report["artifact_ref_count"] = sum(
                1 for item in history if str(item.get("_kind") or "") == "internal_artifact_ref"
            )
            self.logger.info(
                build_log_message(
                    "compactor",
                    "lightcompact_applied",
                    chat_id=state.get("chat_id"),
                    history_count=len(history),
                    total_tokens=total_tokens,
                )
            )
        report["token_after"] = total_tokens
        return state, report

    def compact_if_needed(
        self,
        state: Dict[str, Any],
        prompt_tokens: int = 0,
        request_model: str = "",
    ) -> Dict[str, Any]:
        """Run full compaction only when thresholds and safety checks allow it."""
        snapshot = state.get("compaction_snapshot") or self.store.load_snapshot(state["chat_id"])
        history_tokens = self.token_counter.count_messages(self._visible_history(state.get("history", [])))
        total_tokens = prompt_tokens + history_tokens
        threshold = self.compact_threshold()
        self.logger.info(
            build_log_message(
                "compactor",
                "evaluate",
                chat_id=state.get("chat_id"),
                history_tokens=history_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
                threshold=threshold,
            )
        )

        if total_tokens < threshold:
            snapshot["last_effective_tokens"] = total_tokens
            snapshot["last_compaction_reason"] = "below_threshold"
            snapshot["updated_at"] = now_ts()
            state["compaction_snapshot"] = snapshot
            self.store.save_snapshot(state["chat_id"], snapshot)
            self.logger.debug(build_log_message("compactor", "skip_below_threshold", chat_id=state.get("chat_id")))
            return state

        if snapshot.get("consecutive_failures", 0) >= self.valves.MAX_CONSECUTIVE_COMPACTION_FAILURES:
            snapshot["last_compaction_reason"] = "circuit_open"
            snapshot["updated_at"] = now_ts()
            state["compaction_snapshot"] = snapshot
            self.store.save_snapshot(state["chat_id"], snapshot)
            self.logger.warning(
                build_log_message(
                    "compactor",
                    "circuit_open",
                    chat_id=state.get("chat_id"),
                    consecutive_failures=snapshot.get("consecutive_failures", 0),
                )
            )
            return state

        try:
            rebuilt_history, new_snapshot = self._compact_history(
                chat_id=state["chat_id"],
                history=state.get("history", []),
                snapshot=snapshot,
                attachment_manifest=state.get("attachment_manifest", {}) or {},
                request_model=request_model or self.valves.TARGET_MODEL,
            )
            rebuilt_history = self._enforce_hard_limit(rebuilt_history, prompt_tokens)
            new_snapshot["consecutive_failures"] = 0
            new_snapshot["last_effective_tokens"] = prompt_tokens + self.token_counter.count_messages(rebuilt_history)
            new_snapshot["last_compaction_reason"] = "threshold_exceeded"
            new_snapshot["updated_at"] = now_ts()
            state["history"] = rebuilt_history
            state["summary_message"] = self._latest_summary_message(rebuilt_history)
            state["compaction_snapshot"] = new_snapshot
            state["transcript_paths"] = list(new_snapshot.get("transcript_paths", []) or [])
            self.store.save_snapshot(state["chat_id"], new_snapshot)
            self.logger.info(
                build_log_message(
                    "compactor",
                    "compact_success",
                    chat_id=state.get("chat_id"),
                    rebuilt_history_count=len(rebuilt_history),
                    transcript_count=len(state.get("transcript_paths", []) or []),
                )
            )
            return state
        except Exception as exc:
            snapshot["consecutive_failures"] = snapshot.get("consecutive_failures", 0) + 1
            snapshot["last_compaction_reason"] = "compaction_failed"
            snapshot["updated_at"] = now_ts()
            state["compaction_snapshot"] = snapshot
            self.store.save_snapshot(state["chat_id"], snapshot)
            self.logger.exception(
                build_log_message(
                    "compactor",
                    "compact_failed",
                    chat_id=state.get("chat_id"),
                    error_type=type(exc).__name__,
                )
            )
            return state

    def _compact_history(
        self,
        chat_id: str,
        history: List[Dict[str, Any]],
        snapshot: Dict[str, Any],
        attachment_manifest: Dict[str, Any],
        request_model: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Rebuild history into summary + preserved anchors + recent working set."""
        previous_summary = self._existing_summary_text(history, snapshot)
        working_history = [
            item
            for item in history
            if item.get("_kind") != "memory_summary" and not self._is_internal_message(item)
        ]
        groups = self._group_history(working_history)
        early_groups, middle_groups, recent_groups, pinned_user_groups = self._split_groups(groups)

        group_digests = [self._group_digest(group) for group in middle_groups]
        delta_start = self._resolve_delta_start(
            group_digests,
            str(snapshot.get("last_compacted_group_anchor", "") or ""),
            int(snapshot.get("last_compacted_group_count", 0) or 0),
        )
        delta_groups = middle_groups[delta_start:]
        delta_messages = self._flatten_groups(delta_groups)
        self.logger.info(
            build_log_message(
                "compactor",
                "compact_history_split",
                chat_id=chat_id,
                group_count=len(groups),
                early_groups=len(early_groups),
                middle_groups=len(middle_groups),
                recent_groups=len(recent_groups),
                pinned_user_groups=pinned_user_groups,
                delta_groups=len(delta_groups),
            )
        )

        transcript_path = ""
        summary = previous_summary
        if delta_messages:
            transcript_path = self.store.write_transcript(chat_id, delta_messages)
            self.logger.info(
                build_log_message(
                    "compactor",
                    "delta_selected",
                    chat_id=chat_id,
                    delta_message_count=len(delta_messages),
                    transcript_name=display_path_name(transcript_path),
                    previous_summary_present=bool(previous_summary.strip()),
                )
            )
            summary = self._merge_summary(
                previous_summary=previous_summary,
                delta_messages=delta_messages,
                transcript_path=transcript_path,
                request_model=request_model,
                attachment_manifest=attachment_manifest,
            )

        new_snapshot = dict(snapshot)
        if summary:
            new_snapshot["summary"] = summary
        if transcript_path:
            transcript_paths = list(snapshot.get("transcript_paths", []) or [])
            transcript_paths.append(transcript_path)
            new_snapshot["transcript_paths"] = transcript_paths[-20:]
        if group_digests:
            new_snapshot["last_compacted_group_anchor"] = group_digests[-1]
        new_snapshot["last_compacted_group_count"] = len(middle_groups)

        rebuilt: List[Dict[str, Any]] = []
        if transcript_path:
            rebuilt.append(
                self._build_internal_compaction_event(
                    event_name="full_compaction",
                    artifact_path=transcript_path,
                    artifact_label=display_path_name(transcript_path),
                    summary_present=bool(summary),
                )
            )
        summary_item = self._build_memory_summary_message(new_snapshot)
        if summary_item is not None:
            rebuilt.append(summary_item)
        rebuilt.extend(self._flatten_groups(early_groups))
        rebuilt.extend(self._flatten_groups(recent_groups))
        rebuilt = self._dedupe_messages(rebuilt)
        self.logger.debug(
            build_log_message(
                "compactor",
                "compact_history_rebuilt",
                chat_id=chat_id,
                rebuilt_count=len(rebuilt),
                has_summary=bool(summary_item),
            )
        )
        return rebuilt, new_snapshot

    def _group_history(self, history: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        groups: List[List[Dict[str, Any]]] = []
        index = 0
        while index < len(history):
            item = history[index]
            if item.get("_kind") == "assistant_tool_use":
                block = [item]
                index += 1
                while index < len(history) and history[index].get("_kind") == "user_tool_result":
                    block.append(history[index])
                    index += 1
                groups.append(block)
                continue
            groups.append([item])
            index += 1
        return groups

    def _lightcompact_history(
        self,
        chat_id: str,
        history: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        changed = False
        candidate_indexes = self._lightcompact_candidate_indexes(history)
        rebuilt: List[Dict[str, Any]] = []
        candidate_index_set = set(candidate_indexes)
        for index, item in enumerate(history):
            if index not in candidate_index_set:
                rebuilt.append(item)
                continue
            if not self._can_lightcompact_message(item):
                rebuilt.append(item)
                continue
            content = flatten_content(item.get("content"))
            if len(content) <= self.lightcompact_min_source_chars():
                rebuilt.append(item)
                continue
            artifact_path = self.store.write_text_artifact(
                chat_id=chat_id,
                prefix=f"lightcompact_{item.get('_kind') or item.get('role') or 'message'}",
                text=content,
            )
            replacement = self.render_lightcompact_replacement(
                label=str(item.get("_kind") or item.get("role") or "message"),
                content=content,
                artifact_label=display_path_name(artifact_path),
            )
            rebuilt.append(
                self._build_internal_artifact_ref(
                    source_kind=str(item.get("_kind") or item.get("role") or "message"),
                    artifact_path=artifact_path,
                    artifact_label=display_path_name(artifact_path),
                )
            )
            rebuilt.append({
                **item,
                "content": replacement,
                "_source_name": display_path_name(artifact_path),
                "_lightcompact_applied": True,
            })
            changed = True
        self.logger.info(
            build_log_message(
                "compactor",
                "lightcompact_history",
                chat_id=chat_id,
                candidate_count=len(candidate_indexes),
                changed=changed,
            )
        )
        return rebuilt if changed else history, changed

    def _split_groups(
        self, groups: List[List[Dict[str, Any]]]
    ) -> Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]], List[List[Dict[str, Any]]], int]:
        total = len(groups)
        early_end = min(total, self.valves.KEEP_FIRST_MESSAGES)
        early_indexes = set(range(early_end))
        recent_start = max(early_end, total - self.valves.KEEP_LAST_MESSAGES)
        recent_indexes = set(range(recent_start, total))
        pinned_user_indexes = self._recent_user_group_indexes(groups)
        preserved_indexes = early_indexes | recent_indexes | pinned_user_indexes
        early_groups = [groups[index] for index in range(total) if index in early_indexes]
        middle_groups = [groups[index] for index in range(total) if index not in preserved_indexes]
        recent_groups = [groups[index] for index in range(total) if index in preserved_indexes and index not in early_indexes]
        return early_groups, middle_groups, recent_groups, len(pinned_user_indexes)

    def _recent_user_group_indexes(self, groups: List[List[Dict[str, Any]]]) -> set[int]:
        keep = max(0, int(getattr(self.valves, "KEEP_LAST_USER_MESSAGES", 0) or 0))
        if keep <= 0:
            return set()
        selected: List[int] = []
        for index in range(len(groups) - 1, -1, -1):
            if self._is_user_text_group(groups[index]):
                selected.append(index)
                if len(selected) >= keep:
                    break
        return set(selected)

    def _is_user_text_group(self, group: List[Dict[str, Any]]) -> bool:
        return (
            len(group) == 1
            and isinstance(group[0], dict)
            and str(group[0].get("_kind") or "") == "user_text"
        )

    def _lightcompact_candidate_indexes(self, history: List[Dict[str, Any]]) -> List[int]:
        total = len(history)
        early_indexes = set(range(min(total, self.valves.KEEP_FIRST_MESSAGES)))
        recent_indexes = set(range(max(0, total - self.valves.KEEP_LAST_MESSAGES), total))
        recent_user_indexes = self._recent_user_message_indexes(history)
        candidate_indexes: List[int] = []
        for index, item in enumerate(history):
            if index in early_indexes or index in recent_indexes or index in recent_user_indexes:
                continue
            if str(item.get("_kind") or "") == "memory_summary" or self._is_internal_message(item):
                continue
            candidate_indexes.append(index)
        return candidate_indexes

    def _recent_user_message_indexes(self, history: List[Dict[str, Any]]) -> set[int]:
        keep = max(0, int(getattr(self.valves, "KEEP_LAST_USER_MESSAGES", 0) or 0))
        selected: List[int] = []
        if keep <= 0:
            return set()
        for index in range(len(history) - 1, -1, -1):
            item = history[index]
            if str(item.get("_kind") or "") != "user_text":
                continue
            selected.append(index)
            if len(selected) >= keep:
                break
        return set(selected)

    def _can_lightcompact_message(self, item: Dict[str, Any]) -> bool:
        kind = str(item.get("_kind") or "")
        if item.get("_lightcompact_applied"):
            return False
        return kind in {"assistant_text", "user_tool_result"}

    def render_lightcompact_replacement(
        self,
        label: str,
        content: str,
        artifact_label: str,
    ) -> str:
        max_summary_lines = max(3, int(getattr(self.valves, "LIGHTCOMPACT_MAX_SUMMARY_LINES", 6) or 6))
        line_chars = max(60, int(getattr(self.valves, "LIGHTCOMPACT_LINE_CHARS", 180) or 180))
        return self._build_lightcompact_replacement(
            label=label,
            content=content,
            artifact_label=artifact_label,
            max_summary_lines=max_summary_lines,
            line_chars=line_chars,
        )

    def _build_lightcompact_replacement(
        self,
        label: str,
        content: str,
        artifact_label: str,
        max_summary_lines: int,
        line_chars: int,
    ) -> str:
        summary_lines = self._summarize_lightcompact_content(
            content=content,
            max_summary_lines=max_summary_lines,
            line_chars=line_chars,
        )
        line_count = len(str(content or "").splitlines()) if content else 0
        summary_block = "\n".join(f"- {line}" for line in summary_lines) or "- No compact summary available."
        return (
            f"[lightly compacted older {label}; full original moved out of inline history]\n"
            "Summary:\n"
            f"{summary_block}\n\n"
            f"Original size: {len(content)} chars across {line_count} lines\n"
            f"Full original saved as transcript artifact: {artifact_label}\n"
            "Use read_transcript with that path if exact details are needed."
        )

    def _summarize_lightcompact_content(
        self,
        content: str,
        max_summary_lines: int,
        line_chars: int,
    ) -> List[str]:
        normalized_lines: List[str] = []
        seen: set[str] = set()
        for raw_line in str(content or "").splitlines():
            collapsed = re.sub(r"\s+", " ", raw_line).strip(" -\t")
            if len(collapsed) < 8:
                continue
            if re.fullmatch(r"[-=*_#.`~]+", collapsed):
                continue
            if collapsed in seen:
                continue
            seen.add(collapsed)
            normalized_lines.append(collapsed)

        prioritized = sorted(
            normalized_lines,
            key=lambda line: (
                0 if re.search(r"(error|exception|traceback|failed|warning|todo|fix|file|path|line|tool|summary)", line, re.I) else 1,
                len(line),
            ),
        )
        selected = prioritized[:max_summary_lines]
        if not selected and content:
            sentence_chunks = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", content).strip())
            selected = [chunk.strip() for chunk in sentence_chunks if len(chunk.strip()) >= 12][:max_summary_lines]

        return [trim_text(line, line_chars, max(20, line_chars // 4)) for line in selected]

    def _resolve_delta_start(self, digests: List[str], saved_anchor: str, saved_count: int) -> int:
        if saved_anchor and saved_anchor in digests:
            return digests.index(saved_anchor) + 1
        if saved_count and saved_count <= len(digests):
            return saved_count
        return 0

    def _group_digest(self, group: List[Dict[str, Any]]) -> str:
        return "||".join(message_digest(item) for item in group)

    def _flatten_groups(self, groups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        flattened: List[Dict[str, Any]] = []
        for group in groups:
            flattened.extend(group)
        return flattened

    def _merge_summary(
        self,
        previous_summary: str,
        delta_messages: List[Dict[str, Any]],
        transcript_path: str,
        request_model: str,
        attachment_manifest: Dict[str, Any],
    ) -> str:
        """Generate or merge continuation summary with fallback handling."""

        model_id = str(self.valves.SUMMARY_MODEL or request_model or "").strip()
        if not model_id:
            self.logger.warning(build_log_message("compactor", "summary_model_missing"))
            return self._fallback_delta_summary(previous_summary, delta_messages, transcript_path, attachment_manifest)

        self.logger.info(
            build_log_message(
                "compactor",
                "summary_merge_start",
                model_id=model_id,
                delta_message_count=len(delta_messages),
                attachment_count=len((attachment_manifest or {}).get("files", []) or []),
                transcript_name=display_path_name(transcript_path),
                previous_summary_chars=len(previous_summary or ""),
            )
        )
        strict_prompt = self._build_summary_prompt(
            delta_messages=delta_messages,
            previous_summary=previous_summary,
            transcript_path=transcript_path,
            attachment_manifest=attachment_manifest,
        )
        try:
            summary = self.provider.summarize(
                model_id=model_id,
                prompt=strict_prompt,
                temperature=self.valves.SUMMARY_TEMPERATURE,
                max_tokens=self.valves.SUMMARY_MAX_TOKENS,
            )
            summary = self._normalize_summary_output(summary)
            self.logger.info(
                build_log_message(
                    "compactor",
                    "strict_summary_result",
                    transcript_name=display_path_name(transcript_path),
                    summary_chars=len(summary),
                    valid_shape=self._validate_summary_shape(summary),
                )
            )
            if self._validate_summary_shape(summary):
                self.logger.info(build_log_message("compactor", "strict_summary_success", transcript=transcript_path))
                return summary.strip()
        except Exception as exc:
            self.logger.warning(
                build_log_message(
                    "compactor",
                    "strict_summary_failed",
                    transcript=transcript_path,
                    error_type=type(exc).__name__,
                )
            )

        loose_prompt = self._build_loose_summary_prompt(
            delta_messages=delta_messages,
            previous_summary=previous_summary,
            transcript_path=transcript_path,
            attachment_manifest=attachment_manifest,
        )
        try:
            summary = self.provider.summarize(
                model_id=model_id,
                prompt=loose_prompt,
                temperature=self.valves.SUMMARY_TEMPERATURE,
                max_tokens=self.valves.SUMMARY_MAX_TOKENS,
            )
            summary = self._normalize_summary_output(summary)
            self.logger.info(
                build_log_message(
                    "compactor",
                    "loose_summary_result",
                    transcript_name=display_path_name(transcript_path),
                    summary_chars=len(summary),
                )
            )
            if summary.strip():
                self.logger.info(build_log_message("compactor", "loose_summary_success", transcript=transcript_path))
                return summary.strip()
        except Exception as exc:
            self.logger.warning(
                build_log_message(
                    "compactor",
                    "loose_summary_failed",
                    transcript=transcript_path,
                    error_type=type(exc).__name__,
                )
            )

        self.logger.warning(build_log_message("compactor", "fallback_summary", transcript=transcript_path))
        return self._fallback_delta_summary(previous_summary, delta_messages, transcript_path, attachment_manifest)

    def _build_summary_prompt(
        self,
        delta_messages: List[Dict[str, Any]],
        previous_summary: str,
        transcript_path: str,
        attachment_manifest: Dict[str, Any],
    ) -> str:
        rendered = self._render_delta_for_summary(delta_messages)
        transcript_name = display_path_name(transcript_path)
        known_files = self._render_known_files(attachment_manifest)
        return "\n".join(
            [
                "Create a detailed continuation summary for a coding agent conversation that is running out of context.",
                "Summarize only conversation history and tool interactions.",
                "Do not restate full attachment contents unless they were explicitly quoted or inspected in the conversation.",
                "First write a private scratchpad in <analysis>...</analysis>.",
                "Then write the final result in <summary>...</summary>.",
                "Inside <summary>, use exactly these markdown headings:",
                "# Session Continuation Summary",
                "## 1. Primary Request and Intent",
                "## 2. Key Technical Concepts",
                "## 3. Files and Code Sections",
                "## 4. Errors and Fixes",
                "## 5. Problem Solving",
                "## 6. All User Messages",
                "## 7. Pending Tasks",
                "## 8. Current Work",
                "## 9. Optional Next Step",
                "",
                "Rules:",
                "- Preserve explicit user requirements, constraints, and unresolved asks.",
                "- In section 3, list only files, modules, APIs, or code sections that were actually discussed, inspected, modified, or requested.",
                "- In section 6, enumerate user messages faithfully and do not collapse them into one bullet.",
                "- Keep the summary compact but concrete enough for the next model turn to continue work without rereading the full transcript.",
                f"- Mention the transcript name when it is relevant for recovery: {transcript_name or '(none)'}",
                "",
                "Known attachments:",
                *(known_files or ["- (none)"]),
                "",
                "Prior Summary:",
                previous_summary.strip() or "(none)",
                "",
                "Delta:",
                rendered or "(empty)",
            ]
        )

    def _build_loose_summary_prompt(
        self,
        delta_messages: List[Dict[str, Any]],
        previous_summary: str,
        transcript_path: str,
        attachment_manifest: Dict[str, Any],
    ) -> str:
        rendered = self._render_delta_for_summary(delta_messages)
        known_files = self._render_known_files(attachment_manifest)
        return "\n".join(
            [
                "Write a compact continuation summary for a coding agent conversation.",
                "Return only a <summary>...</summary> block.",
                "Use the same nine markdown headings as the strict prompt.",
                "Keep facts concrete and preserve all user asks.",
                f"Transcript: {display_path_name(transcript_path) or '(none)'}",
                "Known attachments:",
                *(known_files or ["- (none)"]),
                "",
                "Prior Summary:",
                previous_summary.strip() or "(none)",
                "",
                "Delta:",
                rendered or "(empty)",
            ]
        )

    def _fallback_delta_summary(
        self,
        previous_summary: str,
        delta_messages: List[Dict[str, Any]],
        transcript_path: str,
        attachment_manifest: Dict[str, Any],
    ) -> str:
        head = [trim_text(render_message(message), 300, 150) for message in delta_messages[:3]]
        tail = [trim_text(render_message(message), 300, 150) for message in delta_messages[-3:]]
        previous_primary = self._extract_summary_section(previous_summary, "## 1. Primary Request and Intent")
        user_messages = [
            f"- {trim_text(render_message(message), 220, 120)}"
            for message in delta_messages
            if message.get("role") == "user" and message.get("_kind") != "memory_summary"
        ]
        file_lines = self._render_known_files(attachment_manifest)
        sections = [
            "# Session Continuation Summary",
            "## 1. Primary Request and Intent",
            previous_primary or "- Continue the active user request.",
            "## 2. Key Technical Concepts",
            "- Use workspace-backed file tools rather than guessing file contents.",
            "## 3. Files and Code Sections",
            "\n".join(file_lines) or "- No grounded file references captured.",
            "## 4. Errors and Fixes",
            "- Summary model unavailable; no structured error/fix extraction beyond the delta.",
            "## 5. Problem Solving",
            "- Preserve earlier summary and append the latest observed delta.",
            "## 6. All User Messages",
            "\n".join(user_messages) or "- No user delta captured.",
            "## 7. Pending Tasks",
            "- Read the transcript if older details are needed before continuing.",
            "## 8. Current Work",
            "\n".join(head + tail) or "- No delta captured.",
            "## 9. Optional Next Step",
            f"- If exact pre-compaction details are required, read: {display_path_name(transcript_path) or '(none)'}",
        ]
        return "\n".join(sections).strip()

    def _normalize_summary_output(self, summary: str) -> str:
        text = str(summary or "").strip()
        if not text:
            self.logger.debug(build_log_message("compactor", "summary_normalize_empty"))
            return ""
        had_analysis = bool(re.search(r"<analysis>[\s\S]*?</analysis>", text, flags=re.I))
        text = re.sub(r"<analysis>[\s\S]*?</analysis>", "", text, flags=re.I).strip()
        match = re.search(r"<summary>([\s\S]*?)</summary>", text, flags=re.I)
        self.logger.debug(
            build_log_message(
                "compactor",
                "summary_normalized",
                had_analysis=had_analysis,
                had_summary_block=bool(match),
                output_chars=len(match.group(1).strip()) if match else len(text),
            )
        )
        if match:
            return match.group(1).strip()
        return text

    def _render_known_files(self, attachment_manifest: Dict[str, Any]) -> List[str]:
        items = list((attachment_manifest or {}).get("files", []) or [])
        rendered: List[str] = []
        for item in items[:12]:
            display_name = str(item.get("display_name") or item.get("workspace_name") or item.get("workspace_path") or "").strip()
            workspace_path = str(item.get("workspace_path") or "").strip()
            if display_name and workspace_path:
                rendered.append(f"- {display_name} ({workspace_path})")
            elif display_name:
                rendered.append(f"- {display_name}")
        self.logger.debug(
            build_log_message(
                "compactor",
                "known_files_rendered",
                attachment_count=len(items),
                rendered_count=len(rendered),
            )
        )
        return rendered

    def _validate_summary_shape(self, summary: str) -> bool:
        if not summary.strip():
            return False
        required_headers = [
            "# Session Continuation Summary",
            "## 1. Primary Request and Intent",
            "## 2. Key Technical Concepts",
            "## 3. Files and Code Sections",
            "## 4. Errors and Fixes",
            "## 5. Problem Solving",
            "## 6. All User Messages",
            "## 7. Pending Tasks",
            "## 8. Current Work",
            "## 9. Optional Next Step",
        ]
        return all(header in summary for header in required_headers)

    def _render_delta_for_summary(self, delta_messages: List[Dict[str, Any]]) -> str:
        total_budget = max(4000, int(getattr(self.valves, "SUMMARY_INPUT_MAX_CHARS", 50000) or 50000))
        user_budget = max(400, int(getattr(self.valves, "SUMMARY_USER_MESSAGE_MAX_CHARS", 2000) or 2000))
        event_budget = max(300, int(getattr(self.valves, "SUMMARY_EVENT_MAX_CHARS", 1200) or 1200))

        user_entries: List[str] = []
        timeline_entries: List[str] = []
        for index, message in enumerate(delta_messages, start=1):
            if self._is_internal_message(message):
                continue
            rendered = render_message(message)
            role = str(message.get("role", "user") or "user")
            kind = str(message.get("_kind") or "")
            if role == "user" and kind != "memory_summary":
                user_entries.append(f"[user #{index}]\n{trim_text(rendered, user_budget, min(300, user_budget // 3))}")
            else:
                timeline_entries.append(f"[event #{index}]\n{trim_text(rendered, event_budget, min(240, event_budget // 3))}")

        sections: List[str] = []
        used = 0

        def append_section(title: str, entries: List[str]) -> None:
            nonlocal used
            if not entries or used >= total_budget:
                return
            header = f"{title}\n"
            if used + len(header) > total_budget:
                return
            sections.append(header.rstrip())
            used += len(header)
            for entry in entries:
                block = entry + "\n"
                if used + len(block) > total_budget:
                    remaining = total_budget - used
                    if remaining <= 120:
                        break
                    sections.append(trim_text(entry, max(60, remaining - 80), 20))
                    used = total_budget
                    break
                sections.append(entry)
                used += len(block)

        append_section("All user messages in delta:", user_entries)

        if timeline_entries and used < total_budget:
            head_count = min(6, len(timeline_entries))
            tail_count = min(12, max(0, len(timeline_entries) - head_count))
            selected_timeline = timeline_entries[:head_count]
            if len(timeline_entries) > head_count + tail_count:
                selected_timeline.append(
                    f"... {len(timeline_entries) - head_count - tail_count} earlier assistant/tool events omitted from the middle timeline ..."
                )
            if tail_count:
                selected_timeline.extend(timeline_entries[-tail_count:])
            append_section("Execution timeline digest:", selected_timeline)

        rendered = "\n\n".join(section for section in sections if section)
        self.logger.debug(
            build_log_message(
                "compactor",
                "render_delta_for_summary",
                delta_message_count=len(delta_messages),
                user_entry_count=len(user_entries),
                timeline_entry_count=len(timeline_entries),
                rendered_chars=len(rendered),
                budget=total_budget,
            )
        )
        return rendered or "(empty)"

    def _extract_summary_section(self, summary: str, header: str) -> str:
        pattern = re.escape(header) + r"\s*(.*?)(?=\n## |\Z)"
        match = re.search(pattern, summary, flags=re.S)
        return match.group(1).strip() if match else ""

    def _build_memory_summary_message(self, snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        summary = str(snapshot.get("summary", "") or "").strip()
        if not summary:
            return None
        transcript_paths = list(snapshot.get("transcript_paths", []) or [])
        latest_transcript = transcript_paths[-1] if transcript_paths else ""
        latest_label = display_path_name(latest_transcript)
        content = (
            "This session is being continued from a previous conversation that ran out of context. "
            "The summary below covers the earlier portion of the conversation.\n\n"
            f"{summary}"
        )
        if latest_label:
            content += (
                "\n\nIf you need specific details from before compaction "
                "(like exact code snippets, error messages, or generated content), "
                f"use read_transcript with path: {latest_label}"
            )
        self.logger.info(
            build_log_message(
                "compactor",
                "memory_summary_built",
                transcript_name=display_path_name(latest_transcript),
                content_chars=len(content),
            )
        )
        return {
            "role": "user",
            "content": content,
            "_kind": "memory_summary",
            "_source_name": latest_label,
        }

    def _enforce_hard_limit(self, history: List[Dict[str, Any]], prompt_tokens: int) -> List[Dict[str, Any]]:
        """Apply emergency tail-preserving trim when context still exceeds hard limit."""
        total = prompt_tokens + self.token_counter.count_messages(self._visible_history(history))
        limit = self.effective_context_window()
        if total <= limit:
            return history

        summary = [item for item in history if item.get("_kind") == "memory_summary"][:1]
        visible_non_summary = [
            item
            for item in history
            if item.get("_kind") != "memory_summary" and not self._is_internal_message(item)
        ]
        if not visible_non_summary:
            return self._dedupe_messages(summary)

        groups = self._group_history(visible_non_summary)
        selected_group_indexes: set[int] = set()
        selected_message_digests: set[str] = set()
        tail_groups_kept = 0
        user_groups_kept = 0
        running_total = prompt_tokens + self.token_counter.count_messages(summary)
        tail_group_target = max(1, int(getattr(self.valves, "EMERGENCY_TAIL_KEEP_MESSAGES", 1) or 1))
        user_group_target = max(0, int(getattr(self.valves, "KEEP_LAST_USER_MESSAGES", 0) or 0))

        for index in range(len(groups) - 1, -1, -1):
            group = groups[index]
            should_keep_for_tail = tail_groups_kept < tail_group_target
            should_keep_for_user = self._is_user_text_group(group) and user_groups_kept < user_group_target
            if not should_keep_for_tail and not should_keep_for_user:
                continue

            group_tokens = self.token_counter.count_messages(group)
            would_fit = running_total + group_tokens <= limit
            if selected_group_indexes and not would_fit:
                continue

            selected_group_indexes.add(index)
            running_total += group_tokens
            tail_groups_kept += 1
            if self._is_user_text_group(group):
                user_groups_kept += 1
            for item in group:
                digest = message_digest(item)
                if digest:
                    selected_message_digests.add(digest)

        preserved: List[Dict[str, Any]] = []
        summary_added = False
        for item in history:
            kind = str(item.get("_kind") or "")
            if kind == "memory_summary":
                if not summary_added:
                    preserved.append(item)
                    summary_added = True
                continue
            if self._is_internal_message(item):
                preserved.append(item)
                continue
            digest = message_digest(item)
            if digest and digest in selected_message_digests:
                preserved.append(item)

        return self._dedupe_messages(preserved)

    def _dedupe_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        last_digest = ""
        for message in messages:
            digest = message_digest(message)
            if digest and digest == last_digest:
                continue
            deduped.append(message)
            last_digest = digest
        return deduped

    def _existing_summary_text(self, history: List[Dict[str, Any]], snapshot: Dict[str, Any]) -> str:
        snapshot_summary = str(snapshot.get("summary", "") or "").strip()
        if snapshot_summary:
            self.logger.debug(build_log_message("compactor", "existing_summary_source", source="snapshot"))
            return snapshot_summary
        latest = self._latest_summary_message(history)
        if latest:
            self.logger.debug(build_log_message("compactor", "existing_summary_source", source="history"))
            return self._extract_summary_body(str(latest.get("content", "") or ""))
        self.logger.debug(build_log_message("compactor", "existing_summary_source", source="none"))
        return ""

    def _extract_summary_body(self, text: str) -> str:
        value = str(text or "").strip()
        header_index = value.find("# Session Continuation Summary")
        if header_index >= 0:
            return value[header_index:].strip()
        return value

    def _latest_summary_message(self, history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for item in reversed(history or []):
            if item.get("_kind") == "memory_summary":
                return item
        return None

    def _build_internal_artifact_ref(
        self,
        source_kind: str,
        artifact_path: str,
        artifact_label: str,
    ) -> Dict[str, Any]:
        return {
            "role": "system",
            "content": f"artifact_label={artifact_label}\nartifact_path={artifact_path}\nsource_kind={source_kind}",
            "_kind": "internal_artifact_ref",
            "_source_name": artifact_label,
        }

    def _build_internal_compaction_event(
        self,
        event_name: str,
        artifact_path: str,
        artifact_label: str,
        summary_present: bool,
    ) -> Dict[str, Any]:
        return {
            "role": "system",
            "content": (
                f"event={event_name}\nartifact_label={artifact_label}\nartifact_path={artifact_path}\n"
                f"summary_present={str(bool(summary_present)).lower()}"
            ),
            "_kind": "internal_compaction_event",
            "_source_name": artifact_label,
        }
