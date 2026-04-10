"""Main runtime loop for Mini Agent sessions.

Handles request routing, state persistence, tool execution, and prompt budgeting.
"""

from __future__ import annotations

from hashlib import sha1
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from .common import build_log_message, flatten_content, get_logger, now_ts, trim_text
from .context_manager import ContextManager
from .history_compactor import HistoryCompactor
from .message_types import append_if_new, latest_user_message, normalize_history_item
from .permission_manager import PermissionManager
from .provider_adapter import ProviderAdapter
from .tool_registry import ToolRegistry
from .workspace_store import WorkspaceStore


class MiniAgentRuntime:
    """Runtime controller for one chat request turn."""

    def __init__(self, valves: Any) -> None:
        """Initialize storage, provider, compactor, context, and tool subsystems."""
        self.valves = valves
        self.logger = get_logger("runtime")
        self.store = WorkspaceStore(
            root_dir=valves.WORKSPACE_ROOT,
            max_file_chars=valves.TOOL_MAX_FILE_CHARS,
            max_transcript_chars=valves.TOOL_MAX_TRANSCRIPT_CHARS,
            openwebui_uploads_root=valves.OPENWEBUI_UPLOADS_ROOT,
            mounted_uploads_root=valves.PIPELINE_MOUNTED_UPLOADS_ROOT,
            enforce_upload_root=valves.ENFORCE_UPLOAD_ROOT,
            file_registration_mode=valves.FILE_REGISTRATION_MODE,
        )
        self.provider = ProviderAdapter(
            base_url=valves.MODEL_API_BASE_URL,
            api_key=valves.MODEL_API_KEY,
            timeout_sec=valves.REQUEST_TIMEOUT_SEC,
        )
        self.compactor = HistoryCompactor(valves, self.store, self.provider)
        self.context_manager = ContextManager(valves, self.compactor)
        self.tool_registry = ToolRegistry(self.store)
        self.permission_manager = PermissionManager()
        self.logger.info(
            build_log_message(
                "runtime",
                "init",
                workspace_root=valves.WORKSPACE_ROOT,
                target_model=valves.TARGET_MODEL,
                max_agent_steps=valves.MAX_AGENT_STEPS,
                file_registration_mode=valves.FILE_REGISTRATION_MODE,
            )
        )

    def refresh(self, valves: Any) -> None:
        """Recreate runtime internals after valve/config changes."""
        self.logger.info(
            build_log_message(
                "runtime",
                "refresh",
                target_model=valves.TARGET_MODEL,
                debug=valves.DEBUG,
            )
        )
        self.close()
        self.__init__(valves)

    def close(self) -> None:
        """Release provider resources when runtime is shut down."""
        if hasattr(self, "provider"):
            self.logger.debug(build_log_message("runtime", "close_provider"))
            self.provider.close()

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[Dict[str, Any]],
        body: Dict[str, Any],
        status_callback: Callable[[str, bool], None] | None = None,
        text_callback: Callable[[str], None] | None = None,
    ) -> str:
        """Entry point for one request: classify, run chat/meta path, persist result."""
        self._emit_status(status_callback, "正在解析请求", done=False)
        self.logger.info(
            build_log_message(
                "runtime",
                "pipe_start",
                model_id=model_id,
                user_message_preview=user_message[:120] if user_message else "",
                message_count=len(messages or []),
            )
        )
        request_kind = self._classify_request(user_message, body)
        self.logger.info(build_log_message("runtime", "request_classified", kind=request_kind))
        if request_kind != "chat":
            self._emit_status(status_callback, f"正在处理 {request_kind} 请求", done=False)
            chat_id = self._resolve_chat_id(body, messages)
            self.logger.info(build_log_message("runtime", "meta_chat_resolved", chat_id=chat_id))
            if request_kind == "citations":
                final_text = self._run_citation_request(
                    chat_id=chat_id,
                    messages=messages,
                    body=body,
                    status_callback=status_callback,
                    text_callback=text_callback,
                )
                if final_text:
                    self._emit_status(status_callback, "已完成", done=True)
                    return final_text
            final_text = self._run_contextual_meta_request(
                chat_id=chat_id,
                user_message=user_message,
                messages=messages,
                body=body,
                request_kind=request_kind,
                status_callback=status_callback,
                text_callback=text_callback,
            )
            self._persist_meta_result(
                chat_id=chat_id,
                messages=messages,
                body=body,
                request_kind=request_kind,
                final_text=final_text,
            )
            self._emit_status(status_callback, "已完成", done=True)
            return final_text

        chat_id = self._resolve_chat_id(body, messages)
        self.logger.info(build_log_message("runtime", "chat_resolved", chat_id=chat_id))
        if chat_id is None:
            return self._run_stateless_request(
                messages=messages,
                request_kind="chat",
                reason="missing_chat_id",
                status_callback=status_callback,
                text_callback=text_callback,
            )
        state = self.store.load_state(chat_id)
        state["last_active_at"] = now_ts()
        state["dynamic_read_budget"] = {}
        self.store.sync_prompt_history(chat_id, messages)

        raw_files = self.store.collect_request_files(body)
        if raw_files:
            self._emit_status(status_callback, f"正在登记附件 {len(raw_files)} 个", done=False)
        self.logger.info(
            build_log_message(
                "runtime",
                "request_files_collected",
                chat_id=chat_id,
                raw_file_count=len(raw_files),
            )
        )
        state["attachment_manifest"] = self.store.register_uploads(
            chat_id=chat_id,
            raw_files=raw_files,
            existing_manifest=state.get("attachment_manifest", {}),
        )
        self.logger.info(
            build_log_message(
                "runtime",
                "attachments_registered",
                chat_id=chat_id,
                attachment_count=len((state.get("attachment_manifest", {}) or {}).get("files", []) or []),
            )
        )

        messages = self._attach_request_files_to_latest_user_message(messages, raw_files)

        if not state.get("history"):
            self.logger.info(build_log_message("runtime", "bootstrap_history", chat_id=chat_id))
            state["history"] = self._bootstrap_history(messages)
        else:
            latest = latest_user_message(messages)
            if latest is not None:
                self.logger.debug(
                    build_log_message(
                        "runtime",
                        "append_latest_user",
                        chat_id=chat_id,
                        role=latest.get("role"),
                    )
                )
                append_if_new(
                    state["history"],
                    normalize_history_item(latest),
                )

        final_text = self._run_agent_loop(
            state,
            status_callback=status_callback,
            text_callback=text_callback,
        )
        self.logger.info(
            build_log_message(
                "runtime",
                "pipe_complete",
                chat_id=chat_id,
                history_count=len(state.get("history", [])),
                answer_preview=final_text[:160],
            )
        )
        state["last_answer"] = final_text
        state["last_active_at"] = now_ts()
        state["summary_message"] = self._latest_summary_message(state.get("history", []))
        self.store.save_state(chat_id, state)
        self._emit_status(status_callback, "已完成", done=True)
        return final_text

    def _run_stateless_request(
        self,
        messages: List[Dict[str, Any]],
        request_kind: str,
        reason: str,
        status_callback: Callable[[str, bool], None] | None = None,
        text_callback: Callable[[str], None] | None = None,
        preserve_prefix_count: int = 0,
    ) -> str:
        """Execute one-off model call without chat session persistence."""
        sanitized = self._fit_request_messages_to_budget(
            messages=messages,
            max_tokens=self.valves.MAX_OUTPUT_TOKENS,
            preserve_prefix_count=preserve_prefix_count,
        )
        self._emit_status(status_callback, f"正在生成 {request_kind} 结果", done=False)
        self.logger.info(
            build_log_message(
                "runtime",
                "stateless_request",
                kind=request_kind,
                reason=reason,
                message_count=len(sanitized),
            )
        )
        if text_callback is not None:
            response = self.provider.call_model_streaming(
                model_id=self.valves.TARGET_MODEL,
                messages=sanitized,
                temperature=self.valves.MODEL_TEMPERATURE,
                max_tokens=self.valves.MAX_OUTPUT_TOKENS,
                on_text_delta=text_callback,
            )
        else:
            response = self.provider.call_model(
                model_id=self.valves.TARGET_MODEL,
                messages=sanitized,
                temperature=self.valves.MODEL_TEMPERATURE,
                max_tokens=self.valves.MAX_OUTPUT_TOKENS,
            )
        text = self.provider.extract_text(response).strip()
        self.logger.info(
            build_log_message(
                "runtime",
                "stateless_complete",
                kind=request_kind,
                answer_preview=text[:160],
            )
        )
        return text

    def _fit_request_messages_to_budget(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        preserve_prefix_count: int = 0,
    ) -> List[Dict[str, Any]]:
        sanitized = self._sanitize_messages_for_model(messages)
        budget = self._request_input_budget(max_tokens)
        original_tokens = self.compactor.count_messages(sanitized)
        if original_tokens <= budget:
            return sanitized

        trimmed = list(sanitized)
        preserve_prefix_count = max(0, min(preserve_prefix_count, len(trimmed)))
        removed_count = 0

        while len(trimmed) > preserve_prefix_count + 1 and self.compactor.count_messages(trimmed) > budget:
            del trimmed[preserve_prefix_count]
            removed_count += 1

        if self.compactor.count_messages(trimmed) > budget and trimmed:
            trim_indexes: List[int] = []
            if len(trimmed) > preserve_prefix_count:
                trim_indexes.append(len(trimmed) - 1)
            if preserve_prefix_count > 0:
                trim_indexes.extend(range(preserve_prefix_count - 1, -1, -1))
            seen_indexes: set[int] = set()
            for index in trim_indexes:
                if index in seen_indexes or index < 0 or index >= len(trimmed):
                    continue
                seen_indexes.add(index)
                trimmed[index] = self._trim_message_to_budget(
                    message=trimmed[index],
                    current_messages=trimmed,
                    index=index,
                    budget=budget,
                )
                if self.compactor.count_messages(trimmed) <= budget:
                    break

        final_tokens = self.compactor.count_messages(trimmed)
        if removed_count > 0 or final_tokens != original_tokens:
            self.logger.warning(
                build_log_message(
                    "runtime",
                    "request_budget_applied",
                    original_tokens=original_tokens,
                    final_tokens=final_tokens,
                    budget=budget,
                    original_message_count=len(sanitized),
                    final_message_count=len(trimmed),
                    removed_count=removed_count,
                    preserve_prefix_count=preserve_prefix_count,
                )
            )
        return trimmed

    def _trim_message_to_budget(
        self,
        message: Dict[str, Any],
        current_messages: List[Dict[str, Any]],
        index: int,
        budget: int,
    ) -> Dict[str, Any]:
        text = flatten_content(message.get("content"))
        if not text:
            return message

        original = text
        head = max(200, min(4000, len(text) // 2))
        tail = max(80, min(1200, len(text) // 4))
        updated = dict(message)

        while len(text) > 512:
            shortened = trim_text(text, head, tail)
            if shortened == text:
                break
            updated["content"] = shortened
            candidate = list(current_messages)
            candidate[index] = updated
            if self.compactor.count_messages(candidate) <= budget:
                return updated
            text = shortened
            head = max(120, head // 2)
            tail = max(40, tail // 2)

        updated["content"] = trim_text(original, 160, 60)
        return updated

    def _request_input_budget(self, max_tokens: int) -> int:
        requested_output = max(0, int(max_tokens or 0))
        return max(1024, int(self.valves.MAX_CONTEXT_TOKENS) - requested_output - 32)

    def _build_dynamic_read_budget(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        current_tokens = self.compactor.count_messages(messages)
        safety_tokens = max(128, int(getattr(self.valves, "LIGHTCOMPACT_BUFFER_TOKENS", 12000) or 12000) // 2)
        available_tokens = max(
            0,
            int(self.valves.MAX_CONTEXT_TOKENS) - current_tokens - int(self.valves.MAX_OUTPUT_TOKENS) - safety_tokens,
        )
        inline_char_cap = max(
            800,
            int(getattr(self.valves, "TOOL_RESULT_INLINE_BUDGET_CHARS", 12000) or 12000) - 600,
        )
        available_chars = min(inline_char_cap, max(0, available_tokens * 3))
        return {
            "current_context_tokens": current_tokens,
            "max_context_tokens": int(self.valves.MAX_CONTEXT_TOKENS),
            "max_output_tokens": int(self.valves.MAX_OUTPUT_TOKENS),
            "safety_tokens": safety_tokens,
            "available_read_tokens": available_tokens,
            "available_read_chars": available_chars,
            "inline_char_cap": inline_char_cap,
        }

    def _run_contextual_meta_request(
        self,
        chat_id: str | None,
        user_message: str,
        messages: List[Dict[str, Any]],
        body: Dict[str, Any],
        request_kind: str,
        status_callback: Callable[[str, bool], None] | None = None,
        text_callback: Callable[[str], None] | None = None,
    ) -> str:
        """Handle meta tasks using captured visible chat history as context."""
        source_messages = body.get("_captured_messages")
        if not chat_id or not isinstance(source_messages, list) or not source_messages:
            return self._run_stateless_request(
                messages=messages,
                request_kind=request_kind,
                reason="meta_task",
                status_callback=status_callback,
                text_callback=text_callback,
            )

        raw_files = self.store.collect_request_files(body)
        visible_history = self._extract_visible_history(source_messages, raw_files=raw_files)
        if not visible_history:
            return self._run_stateless_request(
                messages=messages,
                request_kind=request_kind,
                reason="meta_task_no_visible_history",
                status_callback=status_callback,
                text_callback=text_callback,
            )

        state = self.store.load_state(chat_id)
        state["last_active_at"] = now_ts()
        state["dynamic_read_budget"] = {}
        if raw_files:
            state["attachment_manifest"] = self.store.register_uploads(
                chat_id=chat_id,
                raw_files=raw_files,
                existing_manifest=state.get("attachment_manifest", {}),
            )
        state["history"] = visible_history
        state = self.context_manager.prepare_turn(
            state,
            request_model=self.valves.TARGET_MODEL,
        )

        meta_messages = self._build_compact_meta_instruction_messages(
            request_kind=request_kind,
            user_message=user_message,
        )
        prompt_tokens = self.compactor.count_messages(self.context_manager.build_messages(state)[:2] + meta_messages)
        state["history"] = self.compactor._enforce_hard_limit(state.get("history", []), prompt_tokens)
        request_messages = self.context_manager.build_messages(state) + meta_messages
        request_messages = self._fit_request_messages_to_budget(
            messages=request_messages,
            max_tokens=self.valves.MAX_OUTPUT_TOKENS,
            preserve_prefix_count=2,
        )

        self.logger.info(
            build_log_message(
                "runtime",
                "contextual_meta_request",
                kind=request_kind,
                chat_id=chat_id,
                visible_history_count=len(visible_history),
                meta_message_count=len(meta_messages),
                request_message_count=len(request_messages),
                estimated_tokens=self.compactor.count_messages(request_messages),
            )
        )
        return self._run_stateless_request(
            messages=request_messages,
            request_kind=request_kind,
            reason="contextual_meta_task",
            status_callback=status_callback,
            text_callback=text_callback,
            preserve_prefix_count=2,
        )

    def _build_compact_meta_instruction_messages(
        self,
        request_kind: str,
        user_message: str,
    ) -> List[Dict[str, Any]]:
        lowered = str(user_message or "").strip().lower()
        if request_kind == "follow_ups":
            return [
                {
                    "role": "user",
                    "content": (
                        "Suggest 3-5 relevant follow-up questions or prompts that the user might naturally ask next "
                        "based on the conversation so far. Return strict JSON with the shape "
                        '{"follow_ups":["..."]}. Do not include markdown fences.'
                    ),
                }
            ]
        if request_kind == "title":
            return [
                {
                    "role": "user",
                    "content": (
                        'Generate a concise 3-5 word title for the conversation so far. Return strict JSON with the shape {"title":"..."}. '
                        "Do not include markdown fences."
                    ),
                }
            ]
        if request_kind == "tags":
            return [
                {
                    "role": "user",
                    "content": (
                        'Generate 1-3 broad tags for the conversation so far. Return strict JSON with the shape {"tags":["..."]}. '
                        "Do not include markdown fences."
                    ),
                }
            ]
        if "search quer" in lowered:
            return [
                {
                    "role": "user",
                    "content": (
                        'Analyze the conversation so far and determine whether web search queries are needed. '
                        'Return strict JSON with the shape {"queries":["..."]}. If no search is needed, return {"queries":[]}. '
                        "Do not include markdown fences."
                    ),
                }
            ]
        return [
            {
                "role": "user",
                "content": (
                    "Complete the requested meta task using the conversation so far as context. "
                    "Answer concisely and follow the expected format implied by the task type."
                ),
            }
        ]

    def _run_citation_request(
        self,
        chat_id: str | None,
        messages: List[Dict[str, Any]],
        body: Dict[str, Any],
        status_callback: Callable[[str, bool], None] | None = None,
        text_callback: Callable[[str], None] | None = None,
    ) -> str:
        """Run citation-style request through the tool-capable agent loop."""
        if not chat_id:
            self.logger.warning(build_log_message("runtime", "citation_chat_missing"))
            return ""

        source_messages = body.get("_captured_messages")
        if not isinstance(source_messages, list) or not source_messages:
            self.logger.warning(build_log_message("runtime", "citation_missing_captured_messages", chat_id=chat_id))
            return ""

        raw_files = self.store.collect_request_files(body)
        if raw_files:
            self._emit_status(status_callback, f"正在准备引用回答，检测到附件 {len(raw_files)} 个", done=False)
        visible_history = self._extract_visible_history(source_messages, raw_files=raw_files)
        if not visible_history:
            self.logger.warning(build_log_message("runtime", "citation_no_visible_history", chat_id=chat_id))
            return ""

        state = self.store.load_state(chat_id)
        state["last_active_at"] = now_ts()
        state["attachment_manifest"] = self.store.register_uploads(
            chat_id=chat_id,
            raw_files=raw_files,
            existing_manifest=state.get("attachment_manifest", {}),
        )
        state["history"] = visible_history
        self.logger.info(
            build_log_message(
                "runtime",
                "citation_agent_prepared",
                chat_id=chat_id,
                history_count=len(visible_history),
                attachment_count=len((state.get("attachment_manifest", {}) or {}).get("files", []) or []),
            )
        )
        if self.context_manager.needs_compaction(state):
            state = self.context_manager.compact_history(
                state,
                request_model=self.valves.TARGET_MODEL,
            )
        final_text = self._run_agent_loop(
            state,
            retry_on_attachment_conflict=True,
            status_callback=status_callback,
            text_callback=text_callback,
        )
        state["last_answer"] = final_text
        state["last_active_at"] = now_ts()
        state["summary_message"] = self._latest_summary_message(state.get("history", []))
        self.store.save_state(chat_id, state)
        self.logger.info(
            build_log_message(
                "runtime",
                "citation_agent_complete",
                chat_id=chat_id,
                history_count=len(state.get("history", [])),
                answer_preview=final_text[:160],
            )
        )
        return final_text

    def _persist_meta_result(
        self,
        chat_id: str | None,
        messages: List[Dict[str, Any]],
        body: Dict[str, Any],
        request_kind: str,
        final_text: str,
    ) -> None:
        if not chat_id or request_kind not in {"citations"}:
            self.logger.debug(
                build_log_message(
                    "runtime",
                    "meta_result_not_persisted",
                    chat_id=chat_id,
                    kind=request_kind,
                )
            )
            return

        source_messages = body.get("_captured_messages")
        if not isinstance(source_messages, list) or not source_messages:
            source_messages = messages
        raw_files = self.store.collect_request_files(body)
        visible_history = self._extract_visible_history(source_messages, raw_files=raw_files)
        if not visible_history:
            self.logger.warning(
                build_log_message(
                    "runtime",
                    "meta_result_no_visible_history",
                    chat_id=chat_id,
                    kind=request_kind,
                )
            )
            return

        state = self.store.load_state(chat_id)
        state["dynamic_read_budget"] = {}
        if raw_files:
            state["attachment_manifest"] = self.store.register_uploads(
                chat_id=chat_id,
                raw_files=raw_files,
                existing_manifest=state.get("attachment_manifest", {}),
            )
        state["history"] = visible_history
        append_if_new(
            state["history"],
            {
                "role": "assistant",
                "content": final_text,
                "_kind": "assistant_text",
            },
        )
        state["last_answer"] = final_text
        state["last_active_at"] = now_ts()
        state["summary_message"] = self._latest_summary_message(state.get("history", []))
        self.store.save_state(chat_id, state)
        self.logger.info(
            build_log_message(
                "runtime",
                "meta_result_persisted",
                chat_id=chat_id,
                kind=request_kind,
                history_count=len(state.get("history", [])),
                attachment_count=len((state.get("attachment_manifest", {}) or {}).get("files", []) or []),
            )
        )

    def _run_agent_loop(
        self,
        state: Dict[str, Any],
        retry_on_attachment_conflict: bool = False,
        status_callback: Callable[[str, bool], None] | None = None,
        text_callback: Callable[[str], None] | None = None,
    ) -> str:
        """Iterative reasoning loop: model call, optional tool execution, final answer."""
        retry_instruction = ""
        saw_tool_use_in_this_run = False
        for step in range(1, self.valves.MAX_AGENT_STEPS + 1):
            self._emit_status(status_callback, f"第 {step} 步：正在思考", done=False)
            self.logger.info(
                build_log_message(
                    "runtime",
                    "agent_step_start",
                    chat_id=state.get("chat_id"),
                    step=step,
                    history_count=len(state.get("history", [])),
                )
            )
            state = self.context_manager.prepare_turn(
                state,
                request_model=self.valves.TARGET_MODEL,
            )
            self._emit_prepare_turn_status(status_callback, state.get("last_prepare_turn_report", {}))
            provisional_messages = self._fit_request_messages_to_budget(
                messages=self.context_manager.build_messages(state),
                max_tokens=self.valves.MAX_OUTPUT_TOKENS,
                preserve_prefix_count=2,
            )
            state["dynamic_read_budget"] = self._build_dynamic_read_budget(provisional_messages)
            request_messages = self.context_manager.build_messages(state)
            if retry_instruction:
                request_messages = list(request_messages)
                request_messages.append({"role": "user", "content": retry_instruction})
            request_messages = self._fit_request_messages_to_budget(
                messages=request_messages,
                max_tokens=self.valves.MAX_OUTPUT_TOKENS,
                preserve_prefix_count=2,
            )
            state["dynamic_read_budget"] = self._build_dynamic_read_budget(request_messages)
            self.logger.debug(
                build_log_message(
                    "runtime",
                    "model_request_ready",
                    step=step,
                    request_message_count=len(request_messages),
                    attachment_count=len((state.get("attachment_manifest", {}) or {}).get("files", []) or []),
                    current_context_tokens=state.get("dynamic_read_budget", {}).get("current_context_tokens"),
                    available_read_tokens=state.get("dynamic_read_budget", {}).get("available_read_tokens"),
                )
            )
            streamed_chunks: List[str] = []

            def buffer_text_delta(chunk: str) -> None:
                if chunk:
                    streamed_chunks.append(chunk)

            if text_callback is not None:
                response = self.provider.call_model_streaming(
                    model_id=self.valves.TARGET_MODEL,
                    messages=request_messages,
                    tools=self.tool_registry.tool_schemas(),
                    temperature=self.valves.MODEL_TEMPERATURE,
                    max_tokens=self.valves.MAX_OUTPUT_TOKENS,
                    on_text_delta=buffer_text_delta,
                )
            else:
                response = self.provider.call_model(
                    model_id=self.valves.TARGET_MODEL,
                    messages=request_messages,
                    tools=self.tool_registry.tool_schemas(),
                    temperature=self.valves.MODEL_TEMPERATURE,
                    max_tokens=self.valves.MAX_OUTPUT_TOKENS,
                )

            assistant_item = normalize_history_item(self.provider.to_assistant_history_item(response))
            state["history"].append(assistant_item)
            has_tool_calls = self.provider.has_tool_calls(response)
            self.logger.info(
                build_log_message(
                    "runtime",
                    "assistant_response",
                    step=step,
                    has_tool_calls=has_tool_calls,
                    content_preview=self.provider.extract_text(response)[:160],
                )
            )

            if has_tool_calls:
                saw_tool_use_in_this_run = True
                for tool_call in self.provider.extract_tool_calls(response):
                    self._emit_status(status_callback, self._describe_tool_call(tool_call), done=False)
                    self.logger.info(
                        build_log_message(
                            "runtime",
                            "tool_dispatch",
                            step=step,
                            tool_name=tool_call.get("name"),
                            tool_call_id=tool_call.get("id"),
                        )
                    )
                    decision = self.permission_manager.check(tool_call, state)
                    tool_result = self.tool_registry.execute(tool_call, state, decision)
                    tool_result, internal_messages = self._budget_tool_result_message(state, tool_result)
                    self._emit_status(status_callback, self._describe_tool_result(tool_call, tool_result), done=False)
                    state["history"].append(tool_result)
                    if internal_messages:
                        state["history"].extend(internal_messages)
                continue

            final_text = self.provider.extract_text(response).strip()
            if final_text:
                if (
                    retry_on_attachment_conflict
                    and not saw_tool_use_in_this_run
                    and not retry_instruction
                    and self._should_require_text_attachment_inspection(state)
                ):
                    state["history"].pop()
                    retry_instruction = self._build_text_attachment_inspection_retry_instruction(state)
                    self.logger.warning(
                        build_log_message(
                            "runtime",
                            "text_attachment_retry",
                            step=step,
                            chat_id=state.get("chat_id"),
                            attachment_count=len((state.get("attachment_manifest", {}) or {}).get("files", []) or []),
                            content_preview=final_text[:160],
                        )
                    )
                    continue
                if retry_on_attachment_conflict and self._response_conflicts_with_registered_attachments(
                    state,
                    final_text,
                ):
                    state["history"].pop()
                    retry_instruction = self._build_attachment_conflict_retry_instruction(state)
                    self.logger.warning(
                        build_log_message(
                            "runtime",
                            "attachment_conflict_retry",
                            step=step,
                            chat_id=state.get("chat_id"),
                            attachment_count=len((state.get("attachment_manifest", {}) or {}).get("files", []) or []),
                            content_preview=final_text[:160],
                        )
                    )
                    continue
                if retry_on_attachment_conflict and self._response_deferred_native_multimodal_attachments(
                    state,
                    final_text,
                ):
                    state["history"].pop()
                    retry_instruction = self._build_native_multimodal_retry_instruction(state)
                    self.logger.warning(
                        build_log_message(
                            "runtime",
                            "native_multimodal_retry",
                            step=step,
                            chat_id=state.get("chat_id"),
                            attachment_count=len((state.get("attachment_manifest", {}) or {}).get("files", []) or []),
                            content_preview=final_text[:160],
                        )
                    )
                    continue
                if text_callback is not None:
                    self._flush_streamed_chunks(streamed_chunks, text_callback)
                    self._emit_status(status_callback, "正在生成最终回答", done=False)
                    self.logger.info(
                        build_log_message(
                            "runtime",
                            "agent_step_complete",
                            step=step,
                            finish_reason="streaming_text",
                        )
                    )
                    return final_text
                self.logger.info(
                    build_log_message(
                        "runtime",
                        "agent_step_complete",
                        step=step,
                        finish_reason="text",
                    )
                )
                return final_text
            self.logger.warning(
                build_log_message(
                    "runtime",
                    "agent_step_empty_text",
                    step=step,
                )
            )
            return "The model ended the turn without returning displayable text."

        self.logger.warning(build_log_message("runtime", "agent_step_limit", max_steps=self.valves.MAX_AGENT_STEPS))
        return "Reached the maximum number of agent steps and stopped."

    def _flush_streamed_chunks(
        self,
        chunks: List[str],
        text_callback: Callable[[str], None] | None,
    ) -> None:
        if text_callback is None:
            return
        for chunk in chunks:
            if chunk:
                text_callback(chunk)

    def _budget_tool_result_message(
        self,
        state: Dict[str, Any],
        tool_result: Dict[str, Any],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        content = flatten_content(tool_result.get("content"))
        inline_budget = max(1000, int(getattr(self.valves, "TOOL_RESULT_INLINE_BUDGET_CHARS", 12000) or 12000))
        if len(content) <= inline_budget:
            return (
                normalize_history_item(tool_result),
                [],
            )

        artifact_path = self.store.write_text_artifact(
            chat_id=state["chat_id"],
            prefix=f"tool_result_{tool_result.get('name') or 'tool'}",
            text=content,
        )
        artifact_label = Path(artifact_path).name
        tool_label = str(tool_result.get("name") or "tool_result").strip() or "tool_result"
        preserved_head = self._tool_result_progress_head(content)
        replacement = (
            "Tool result was too large to keep fully inline in conversation history.\n\n"
            + (f"Preserved progress metadata:\n{preserved_head}\n\n" if preserved_head else "")
            + self.compactor.render_lightcompact_replacement(
                label=f"tool_result {tool_label}",
                content=content,
                artifact_label=artifact_label,
            )
        )
        budgeted = dict(tool_result)
        budgeted["content"] = replacement
        budgeted["_source_name"] = artifact_label
        self.logger.info(
            build_log_message(
                "runtime",
                "tool_result_budget_applied",
                chat_id=state.get("chat_id"),
                tool_name=tool_result.get("name"),
                original_chars=len(content),
                replacement_chars=len(replacement),
                artifact_name=artifact_label,
            )
        )
        internal_message = {
            "role": "system",
            "content": (
                f"artifact_label={artifact_label}\nartifact_path={artifact_path}\n"
                f"source_kind=runtime_tool_result\ntool_name={tool_result.get('name') or ''}"
            ),
            "_kind": "internal_artifact_ref",
            "_source_name": artifact_label,
        }
        return (
            normalize_history_item(budgeted),
            [internal_message],
        )

    def _tool_result_progress_head(self, content: str) -> str:
        lines = [str(line).rstrip() for line in str(content or "").splitlines()]
        preserved: List[str] = []
        for line in lines[:24]:
            if line.startswith("[chunk_meta]") or line.startswith("[range_meta]") or line.startswith("[note]"):
                preserved.append(line)
            elif preserved:
                break
        return "\n".join(preserved[:12]).strip()

    def _emit_prepare_turn_status(
        self,
        callback: Callable[[str, bool], None] | None,
        report: Dict[str, Any],
    ) -> None:
        if callback is None or not isinstance(report, dict):
            return
        messages: List[str] = []
        if report.get("lightcompact_applied"):
            messages.append("Applied light compact externalization")
        if report.get("full_compaction_applied"):
            messages.append("Applied full summary compaction")
        if messages:
            before_tokens = report.get("token_before")
            after_tokens = report.get("token_after")
            artifact_refs = report.get("artifact_ref_count")
            detail = f" (tokens: {before_tokens} -> {after_tokens}"
            if artifact_refs not in (None, ""):
                detail += f"; artifact refs: {artifact_refs}"
            detail += ")"
            self._emit_status(callback, "; ".join(messages) + detail, done=False)

    def _emit_final_text_chunks(
        self,
        text: str,
        text_callback: Callable[[str], None],
    ) -> None:
        if not text:
            return
        chunk_size = 120
        for index in range(0, len(text), chunk_size):
            text_callback(text[index:index + chunk_size])

    def _response_conflicts_with_registered_attachments(self, state: Dict[str, Any], text: str) -> bool:
        attachment_count = len((state.get("attachment_manifest", {}) or {}).get("files", []) or [])
        if attachment_count <= 0:
            return False
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False

        conflict_phrases = (
            "你没有提供",
            "未提供",
            "没有上传",
            "未上传",
            "没有附上",
            "未附上",
            "没有找到对比文档",
            "未找到对比文档",
            "没有提供具体的对比对象",
            "没有上传另一个文档",
            "没有上传另一个文件",
            "did not provide",
            "didn't provide",
            "not provide",
            "not uploaded",
            "no attachment",
            "no attachments",
            "no file was provided",
            "no files were provided",
            "no document was provided",
            "no documents were provided",
            "no comparison document",
        )
        return any(phrase in lowered for phrase in conflict_phrases)

    def _emit_status(
        self,
        callback: Callable[[str, bool], None] | None,
        description: str,
        done: bool,
    ) -> None:
        if callback is None:
            return
        try:
            callback(str(description or ""), bool(done))
        except Exception:
            self.logger.debug(build_log_message("runtime", "status_callback_failed"))

    def _build_attachment_conflict_retry_instruction(self, state: Dict[str, Any]) -> str:
        files = list((state.get("attachment_manifest", {}) or {}).get("files", []) or [])
        file_names = [str(item.get("display_name") or "").strip() for item in files if str(item.get("display_name") or "").strip()]
        listed = ", ".join(file_names[:5]) if file_names else "the registered attachments"
        return (
            "Attachment consistency check: attachments are registered for this session. "
            f"Known attachments: {listed}. "
            "Do not claim the user failed to upload or provide files. "
            "If the answer depends on those files, inspect them with list_attachments, read_file_chunk, read_file_range, or search_in_file before answering."
        )

    def _response_deferred_native_multimodal_attachments(self, state: Dict[str, Any], text: str) -> bool:
        files = list((state.get("attachment_manifest", {}) or {}).get("files", []) or [])
        if not files:
            return False

        has_native_multimodal_candidate = any(
            str(item.get("content_type") or "").lower().startswith("image/")
            or str(item.get("content_type") or "").lower() == "application/pdf"
            or str(item.get("display_name") or "").lower().endswith(".pdf")
            for item in files
            if isinstance(item, dict)
        )
        if not has_native_multimodal_candidate:
            return False

        lowered = str(text or "").strip().lower()
        if not lowered:
            return False

        conflict_phrases = (
            "无法通过文本搜索或行读取工具直接解析",
            "暂时无法读取其具体内容",
            "转换为markdown",
            "转换为txt",
            "转换为文本",
            "直接复制为文本",
            "pdf 文件目前无法",
            "cannot parse pdf",
            "cannot read pdf",
            "pdf is not supported for text",
            "convert it to markdown",
            "convert it to txt",
            "convert it to text",
            "paste the pdf content",
        )
        return any(phrase in lowered for phrase in conflict_phrases)

    def _build_native_multimodal_retry_instruction(self, state: Dict[str, Any]) -> str:
        files = list((state.get("attachment_manifest", {}) or {}).get("files", []) or [])
        listed = ", ".join(
            str(item.get("display_name") or "").strip()
            for item in files[:5]
            if isinstance(item, dict) and str(item.get("display_name") or "").strip()
        ) or "the attached files"
        return (
            "Native multimodal check: the latest user message already includes attached files that may be directly inspectable by the model. "
            f"Known attachments: {listed}. "
            "Do not stop at 'PDF/text tool unsupported'. "
            "Inspect the original attached PDF/image directly if your model can see it, and use workspace file tools only as fallback."
        )

    def _should_require_text_attachment_inspection(self, state: Dict[str, Any]) -> bool:
        files = list((state.get("attachment_manifest", {}) or {}).get("files", []) or [])
        for item in files:
            if not isinstance(item, dict):
                continue
            content_type = str(item.get("content_type") or "").lower()
            display_name = str(item.get("display_name") or "").lower()
            if content_type.startswith("text/"):
                return True
            if display_name.endswith((".md", ".txt", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml", ".toml", ".log", ".csv")):
                return True
        return False

    def _build_text_attachment_inspection_retry_instruction(self, state: Dict[str, Any]) -> str:
        files = list((state.get("attachment_manifest", {}) or {}).get("files", []) or [])
        text_files = [
            str(item.get("display_name") or "").strip()
            for item in files
            if isinstance(item, dict)
            and (
                str(item.get("content_type") or "").lower().startswith("text/")
                or str(item.get("display_name") or "").lower().endswith(
                    (".md", ".txt", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml", ".toml", ".log", ".csv")
                )
            )
            and str(item.get("display_name") or "").strip()
        ]
        listed = ", ".join(text_files[:5]) if text_files else "the registered text attachments"
        return (
            "Grounding check: this request includes text-readable attachments. "
            f"Known text attachments: {listed}. "
            "Before answering, inspect the relevant attachment content with search_in_file, read_file_chunk, or read_file_range instead of answering only from conversation memory."
        )

    def _describe_tool_call(self, tool_call: Dict[str, Any]) -> str:
        name = str(tool_call.get("name") or "unknown_tool")
        tool_input = dict(tool_call.get("input", {}) or {})
        if name in {"read_file_chunk", "read_file_range", "search_in_file"}:
            target = str(tool_input.get("path") or tool_input.get("file_id") or "").strip()
            if target:
                return f"正在调用 {name} 处理 {target}"
        if name == "list_attachments":
            return "正在查看附件列表"
        if name == "read_prompt_history":
            return "正在读取历史 prompt"
        if name == "read_transcript":
            return "正在读取历史转储"
        return f"正在调用工具 {name}"

    def _describe_tool_result(self, tool_call: Dict[str, Any], tool_result: Dict[str, Any]) -> str:
        name = str(tool_call.get("name") or "unknown_tool")
        content = str(tool_result.get("content") or "").strip()
        lowered = content.lower()
        if lowered.startswith("file is empty or not supported"):
            return f"{name} 返回：当前文件不支持文本读取"
        if lowered.startswith("attachment not found:"):
            return f"{name} 返回：目标文件未找到"
        if lowered.startswith("permission denied:"):
            return f"{name} 返回：权限不足"
        if lowered.startswith("tool execution failed:"):
            return f"{name} 返回：工具执行失败"
        return f"工具 {name} 已返回结果"

    def _bootstrap_history(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "system":
                continue
            history.append(
                normalize_history_item(message)
            )
        self.logger.debug(build_log_message("runtime", "history_bootstrapped", history_count=len(history)))
        return history

    def _latest_summary_message(self, history: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        for item in reversed(history or []):
            if isinstance(item, dict) and item.get("_kind") == "memory_summary":
                return item
        return None

    def _sanitize_messages_for_model(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            item: Dict[str, Any] = {
                "role": message.get("role", "user"),
                "content": message.get("content", ""),
            }
            for key in ["name", "tool_call_id", "tool_calls", "file", "files", "images"]:
                value = message.get(key)
                if value not in (None, "", []):
                    item[key] = value
            sanitized.append(item)
        return sanitized

    def _message_text(self, message: Dict[str, Any]) -> str:
        if not isinstance(message, dict):
            return ""
        return flatten_content(message.get("content"))

    def _task_head(self, text: str) -> str:
        lowered = (text or "").strip().lower()
        if not lowered:
            return ""
        marker = "### task:"
        index = lowered.find(marker)
        if index < 0:
            return lowered[:800]
        task_block = lowered[index:index + 2000]
        for separator in ["\n\n###", "\n\ncontext:", "\n\nchat history:", "\n\nconversation:", "\n\nsources:"]:
            split_index = task_block.find(separator)
            if split_index > 0:
                return task_block[:split_index]
        return task_block

    def _classify_message_text(self, text: str) -> str:
        lowered = self._task_head(text)
        if not lowered:
            return "chat"
        if "### task:" in lowered:
            if "incorporating inline citations" in lowered:
                return "citations"
            if "follow-up questions" in lowered or "\"follow_ups\"" in lowered:
                return "follow_ups"
            if "generate a concise, 3-5 word title" in lowered or "\"title\"" in lowered:
                return "title"
            if "generate 1-3 broad tags" in lowered or "\"tags\"" in lowered:
                return "tags"
            return "meta_task"
        return "chat"

    def _extract_visible_history(
        self,
        messages: List[Dict[str, Any]],
        raw_files: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user") or "user")
            if role == "system":
                continue
            if self._classify_message_text(self._message_text(message)) != "chat":
                continue
            history.append(
                normalize_history_item(message)
            )
        history = self._attach_request_files_to_latest_user_message(history, raw_files or [])
        self.logger.debug(
            build_log_message(
                "runtime",
                "extract_visible_history",
                message_count=len(messages or []),
                history_count=len(history),
            )
        )
        return history

    def _attach_request_files_to_latest_user_message(
        self,
        messages: List[Dict[str, Any]],
        raw_files: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not raw_files:
            return list(messages or [])

        updated = [dict(message) if isinstance(message, dict) else message for message in (messages or [])]
        for index in range(len(updated) - 1, -1, -1):
            message = updated[index]
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "user") or "user") != "user":
                continue
            existing_files = message.get("files")
            if isinstance(existing_files, list) and existing_files:
                return updated
            message["files"] = [dict(item) if isinstance(item, dict) else item for item in raw_files]
            self.logger.debug(
                build_log_message(
                    "runtime",
                    "latest_user_files_attached",
                    role=message.get("role"),
                    file_count=len(raw_files),
                )
            )
            return updated
        return updated

    def _classify_request(self, user_message: str, body: Dict[str, Any]) -> str:
        if body.get("title"):
            return "title"
        body_messages = body.get("messages")
        if isinstance(body_messages, list):
            for message in reversed(body_messages):
                if not isinstance(message, dict):
                    continue
                if str(message.get("role", "user") or "user") != "user":
                    continue
                text = self._message_text(message)
                if text.strip():
                    return self._classify_message_text(text)
        return self._classify_message_text(user_message)

    def _fallback_chat_id(self, messages: List[Dict[str, Any]]) -> str:
        normalized_parts: List[str] = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            normalized_parts.append(f"{role}:{content}")
        digest = sha1("\n".join(normalized_parts).encode("utf-8")).hexdigest()[:16]
        return f"derived-{digest}"

    def _chat_id_candidates(self) -> List[str]:
        return [
            "chat_id",
            "chatId",
            "conversation_id",
            "conversationId",
            "session_id",
            "sessionId",
            "thread_id",
            "threadId",
            "id",
        ]

    def _iter_search_containers(self, root: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        stack: List[Dict[str, Any]] = [root]
        seen: set[int] = set()
        while stack:
            current = stack.pop(0)
            marker = id(current)
            if marker in seen:
                continue
            seen.add(marker)
            yield current
            for key in ["metadata", "user", "session", "context", "info", "params"]:
                value = current.get(key)
                if isinstance(value, dict):
                    stack.append(value)

    def _resolve_chat_id(self, body: Dict[str, Any], messages: List[Dict[str, Any]]) -> str | None:
        explicit = body.get("_resolved_chat_id")
        if isinstance(explicit, str) and explicit.strip():
            self.logger.debug(build_log_message("runtime", "chat_id_from_explicit", value=explicit.strip()))
            return explicit.strip()
        primary_keys = [key for key in self._chat_id_candidates() if key != "id"]
        fallback_keys = ["id"]
        for keys in [primary_keys, fallback_keys]:
            for container in self._iter_search_containers(body):
                for key in keys:
                    value = container.get(key)
                    if isinstance(value, str) and value.strip():
                        candidate = value.strip()
                        if candidate.lower() == "local":
                            continue
                        self.logger.debug(
                            build_log_message("runtime", "chat_id_from_container", key=key, value=candidate)
                        )
                        return candidate
        self.logger.warning(
            build_log_message(
                "runtime",
                "chat_id_not_found",
                top_level_keys=",".join(sorted(str(key) for key in body.keys())),
            )
        )
        if not self.valves.STATELESS_WHEN_CHAT_ID_MISSING:
            fallback = self._fallback_chat_id(messages)
            self.logger.warning(build_log_message("runtime", "chat_id_fallback_derived", value=fallback))
            return fallback
        self.logger.warning(build_log_message("runtime", "chat_id_missing_stateless"))
        return None
