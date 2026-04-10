"""
required_open_webui_version: 0.8.0
Mini Agent runtime pipeline entry.
"""

from __future__ import annotations

import copy
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Union

from pydantic import BaseModel, Field, field_validator


RUNTIME_ROOT = Path(__file__).resolve().parent / "agent-code-build"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from agent_runtime.common import build_log_message, flatten_content, get_logger, set_debug
from agent_runtime.runtime import MiniAgentRuntime


DEFAULT_WORKSPACE_ROOT = os.getenv("MINI_AGENT_WORKSPACE_ROOT", "")
if not DEFAULT_WORKSPACE_ROOT:
    if os.name == "nt":
        DEFAULT_WORKSPACE_ROOT = str((Path(__file__).resolve().parent / "agent_workspace").resolve())
    else:
        DEFAULT_WORKSPACE_ROOT = "/app/agent_workspace"


def env_str(name: str, default: str = "", fallback_names: List[str] | None = None) -> str:
    names = [name, *(fallback_names or [])]
    for candidate in names:
        value = os.getenv(candidate)
        if value not in (None, ""):
            return str(value)
    return default


def env_bool(name: str, default: bool = False, fallback_names: List[str] | None = None) -> bool:
    raw = env_str(name, "", fallback_names)
    if raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int, fallback_names: List[str] | None = None) -> int:
    raw = env_str(name, "", fallback_names)
    if raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def env_float(name: str, default: float, fallback_names: List[str] | None = None) -> float:
    raw = env_str(name, "", fallback_names)
    if raw == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


class Pipeline:
    """OpenWebUI pipeline adapter wrapping `MiniAgentRuntime`."""

    class Valves(BaseModel):
        """Runtime configuration surface (mapped from env vars / pipeline valves)."""
        MODEL_API_BASE_URL: str = Field(
            default=env_str(
                "MINI_AGENT_MODEL_API_BASE_URL",
                "http://host.docker.internal:11434/v1",
                fallback_names=["OPENAI_API_BASE_URL"],
            )
        )
        MODEL_API_KEY: str = Field(
            default=env_str(
                "MINI_AGENT_MODEL_API_KEY",
                "",
                fallback_names=["OPENAI_API_KEY"],
            )
        )
        TARGET_MODEL: str = Field(
            default=env_str(
                "MINI_AGENT_TARGET_MODEL",
                "",
                fallback_names=["OPENAI_MODEL"],
            )
        )
        SUMMARY_MODEL: str = Field(default=env_str("MINI_AGENT_SUMMARY_MODEL", ""))
        WORKSPACE_ROOT: str = Field(default=env_str("MINI_AGENT_WORKSPACE_ROOT", DEFAULT_WORKSPACE_ROOT))
        PROJECT_ROOT: str = Field(default=env_str("MINI_AGENT_PROJECT_ROOT", str(Path.cwd().resolve())))
        OPENWEBUI_UPLOADS_ROOT: str = Field(
            default=env_str(
                "MINI_AGENT_OPENWEBUI_UPLOADS_ROOT",
                "/app/backend/data/uploads",
                fallback_names=["OPEN_WEBUI_UPLOADS_DIR"],
            )
        )
        PIPELINE_MOUNTED_UPLOADS_ROOT: str = Field(
            default=env_str("MINI_AGENT_PIPELINE_MOUNTED_UPLOADS_ROOT", "")
        )
        ENFORCE_UPLOAD_ROOT: bool = Field(
            default=env_bool("MINI_AGENT_ENFORCE_UPLOAD_ROOT", False)
        )
        FILE_REGISTRATION_MODE: str = Field(
            default=env_str("MINI_AGENT_FILE_REGISTRATION_MODE", "copy")
        )
        STATELESS_WHEN_CHAT_ID_MISSING: bool = Field(
            default=env_bool("MINI_AGENT_STATELESS_WHEN_CHAT_ID_MISSING", True)
        )
        MAX_AGENT_STEPS: int = Field(default=env_int("MINI_AGENT_MAX_AGENT_STEPS", 8))
        REQUEST_TIMEOUT_SEC: int = Field(default=env_int("MINI_AGENT_REQUEST_TIMEOUT_SEC", 120))
        MODEL_TEMPERATURE: float = Field(default=env_float("MINI_AGENT_MODEL_TEMPERATURE", 0.1))
        SUMMARY_TEMPERATURE: float = Field(default=env_float("MINI_AGENT_SUMMARY_TEMPERATURE", 0.1))
        MAX_OUTPUT_TOKENS: int = Field(default=env_int("MINI_AGENT_MAX_OUTPUT_TOKENS", 2200))
        SUMMARY_MAX_TOKENS: int = Field(default=env_int("MINI_AGENT_SUMMARY_MAX_TOKENS", 1800))
        MAX_CONTEXT_TOKENS: int = Field(default=env_int("MINI_AGENT_MAX_CONTEXT_TOKENS", 32768))
        RESERVED_OUTPUT_TOKENS: int = Field(default=env_int("MINI_AGENT_RESERVED_OUTPUT_TOKENS", 8000))
        AUTOCOMPACT_BUFFER_TOKENS: int = Field(default=env_int("MINI_AGENT_AUTOCOMPACT_BUFFER_TOKENS", 4000))
        LIGHTCOMPACT_BUFFER_TOKENS: int = Field(
            default=env_int(
                "MINI_AGENT_LIGHTCOMPACT_BUFFER_TOKENS",
                12000,
                fallback_names=["MINI_AGENT_SNIP_BUFFER_TOKENS", "MINI_AGENT_MICROCOMPACT_BUFFER_TOKENS"],
            )
        )
        MAX_CONSECUTIVE_COMPACTION_FAILURES: int = Field(default=env_int("MINI_AGENT_MAX_CONSECUTIVE_COMPACTION_FAILURES", 3))
        KEEP_FIRST_MESSAGES: int = Field(default=env_int("MINI_AGENT_KEEP_FIRST_MESSAGES", 4))
        KEEP_LAST_MESSAGES: int = Field(default=env_int("MINI_AGENT_KEEP_LAST_MESSAGES", 8))
        KEEP_LAST_USER_MESSAGES: int = Field(default=env_int("MINI_AGENT_KEEP_LAST_USER_MESSAGES", 6))
        EMERGENCY_TAIL_KEEP_MESSAGES: int = Field(default=env_int("MINI_AGENT_EMERGENCY_TAIL_KEEP_MESSAGES", 4))
        TOKENIZER_ENCODING: str = Field(default=env_str("MINI_AGENT_TOKENIZER_ENCODING", "o200k_base"))
        FILE_EXCERPT_CHARS: int = Field(default=env_int("MINI_AGENT_FILE_EXCERPT_CHARS", 1200))
        TOOL_MAX_RESULTS: int = Field(default=env_int("MINI_AGENT_TOOL_MAX_RESULTS", 20))
        TOOL_MAX_FILE_CHARS: int = Field(default=env_int("MINI_AGENT_TOOL_MAX_FILE_CHARS", 200000))
        TOOL_MAX_TRANSCRIPT_CHARS: int = Field(default=env_int("MINI_AGENT_TOOL_MAX_TRANSCRIPT_CHARS", 120000))
        TOOL_RESULT_INLINE_BUDGET_CHARS: int = Field(default=env_int("MINI_AGENT_TOOL_RESULT_INLINE_BUDGET_CHARS", 12000))
        LIGHTCOMPACT_MAX_SUMMARY_LINES: int = Field(default=env_int("MINI_AGENT_LIGHTCOMPACT_MAX_SUMMARY_LINES", 6))
        LIGHTCOMPACT_LINE_CHARS: int = Field(default=env_int("MINI_AGENT_LIGHTCOMPACT_LINE_CHARS", 180))
        SUMMARY_INPUT_MAX_CHARS: int = Field(default=env_int("MINI_AGENT_SUMMARY_INPUT_MAX_CHARS", 50000))
        SUMMARY_USER_MESSAGE_MAX_CHARS: int = Field(default=env_int("MINI_AGENT_SUMMARY_USER_MESSAGE_MAX_CHARS", 2000))
        SUMMARY_EVENT_MAX_CHARS: int = Field(default=env_int("MINI_AGENT_SUMMARY_EVENT_MAX_CHARS", 1200))
        PROMPT_HISTORY_MAX_CHARS: int = Field(
            default=env_int("MINI_AGENT_PROMPT_HISTORY_MAX_CHARS", 10000)
        )
        PROMPT_HISTORY_TOTAL_RATIO: float = Field(
            default=env_float("MINI_AGENT_PROMPT_HISTORY_TOTAL_RATIO", 1.5)
        )
        ENABLE_HISTORY_COMPACTION: bool = Field(default=env_bool("MINI_AGENT_ENABLE_HISTORY_COMPACTION", True))
        DEBUG: bool = Field(default=env_bool("MINI_AGENT_DEBUG", True))

        @field_validator(
            "MODEL_API_BASE_URL",
            "MODEL_API_KEY",
            "TARGET_MODEL",
            "SUMMARY_MODEL",
            "WORKSPACE_ROOT",
            "PROJECT_ROOT",
            "OPENWEBUI_UPLOADS_ROOT",
            "PIPELINE_MOUNTED_UPLOADS_ROOT",
            "FILE_REGISTRATION_MODE",
            "TOKENIZER_ENCODING",
            mode="before",
        )
        @classmethod
        def none_to_empty_string(cls, value):
            if value is None:
                return ""
            return value

    def __init__(self) -> None:
        """Initialize valves, logger, runtime, and inlet context caches."""
        self.name = "Mini Agent Code Build"
        self.valves = self.Valves()
        set_debug(self.valves.DEBUG)
        self.logger = get_logger("pipeline")
        self.runtime = MiniAgentRuntime(self.valves)
        self._inlet_context_by_chat_id: Dict[str, Dict[str, Any]] = {}
        self._latest_meta_chat_id_by_kind: Dict[str, Dict[str, Any]] = {}
        self.logger.info(
            build_log_message(
                "pipeline",
                "init",
                name=self.name,
                workspace_root=self.valves.WORKSPACE_ROOT,
                model_base_url=self.valves.MODEL_API_BASE_URL,
                target_model=self.valves.TARGET_MODEL,
                debug=self.valves.DEBUG,
            )
        )

    async def on_startup(self) -> None:
        """OpenWebUI lifecycle hook: refresh runtime on startup."""
        self.logger.info(build_log_message("pipeline", "startup"))
        self.runtime.refresh(self.valves)

    async def on_shutdown(self) -> None:
        """OpenWebUI lifecycle hook: close runtime resources."""
        self.logger.info(build_log_message("pipeline", "shutdown"))
        self.runtime.close()

    async def on_valves_updated(self) -> None:
        """OpenWebUI lifecycle hook: rebuild runtime when valves change."""
        set_debug(self.valves.DEBUG)
        self.logger.info(
            build_log_message(
                "pipeline",
                "valves_updated",
                debug=self.valves.DEBUG,
                target_model=self.valves.TARGET_MODEL,
                file_registration_mode=self.valves.FILE_REGISTRATION_MODE,
                stateless_when_chat_id_missing=self.valves.STATELESS_WHEN_CHAT_ID_MISSING,
            )
        )
        self.runtime.refresh(self.valves)

    async def inlet(self, body: dict, user: dict | None = None) -> dict:
        """Capture request context before `pipe` for chat/meta continuity."""
        chat_id = self._extract_chat_id(body)
        request_kind = self._classify_request(body)
        files = body.get("files", []) if isinstance(body.get("files"), list) else []
        self.logger.info(
            build_log_message(
                "pipeline",
                "inlet_enter",
                chat_id=chat_id,
                request_kind=request_kind,
                message_count=len(body.get("messages", []) if isinstance(body.get("messages"), list) else []),
                file_count=len(files),
            )
        )
        if chat_id and request_kind == "chat":
            cached_messages = copy.deepcopy(body.get("messages", []) if isinstance(body.get("messages"), list) else [])
            self._inlet_context_by_chat_id[chat_id] = {
                "chat_id": chat_id,
                "cached_at": time.time(),
                "files": copy.deepcopy(files),
                "messages": cached_messages,
                "last_user_text": self._latest_user_text(cached_messages),
            }
            self.logger.info(
                build_log_message(
                    "pipeline",
                    "inlet_cached",
                    chat_id=chat_id,
                    message_count=len(self._inlet_context_by_chat_id[chat_id]["messages"]),
                    file_count=len(self._inlet_context_by_chat_id[chat_id]["files"]),
                )
            )
        elif chat_id:
            self._latest_meta_chat_id_by_kind[request_kind] = {
                "chat_id": chat_id,
                "cached_at": time.time(),
            }
            self.logger.info(
                build_log_message(
                    "pipeline",
                    "inlet_meta_chat_cached",
                    chat_id=chat_id,
                    request_kind=request_kind,
                )
            )
        return body

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """Pipeline execution entrypoint called by OpenWebUI per request."""
        merged_body = self._merge_cached_context(body, user_message)
        self.logger.info(
            build_log_message(
                "pipeline",
                "pipe_enter",
                model_id=model_id,
                message_count=len(messages or []),
                has_files=bool(merged_body.get("files")),
                stream=merged_body.get("stream"),
                title=merged_body.get("title", False),
            )
        )
        if merged_body.get("stream"):
            return self._stream_pipe(
                user_message=user_message,
                model_id=model_id,
                messages=messages,
                body=merged_body,
            )
        return self.runtime.pipe(
            user_message=user_message,
            model_id=model_id,
            messages=messages,
            body=merged_body,
        )

    def _stream_pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Generator:
        """Run runtime in a worker thread and stream status/text events."""
        event_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        streamed_text_seen = [False]

        def emit_status(description: str, done: bool) -> None:
            event_queue.put(
                (
                    "status",
                    {
                        "event": {
                            "type": "status",
                            "data": {
                                "description": str(description or ""),
                                "done": bool(done),
                            },
                        }
                    },
                )
            )

        def emit_text(chunk: str) -> None:
            if not chunk:
                return
            streamed_text_seen[0] = True
            event_queue.put(("text", chunk))

        def worker() -> None:
            try:
                result = self.runtime.pipe(
                    user_message=user_message,
                    model_id=model_id,
                    messages=messages,
                    body=body,
                    status_callback=emit_status,
                    text_callback=emit_text,
                )
                event_queue.put(("result", result))
            except Exception as exc:
                self.logger.exception(build_log_message("pipeline", "stream_worker_failed", error_type=type(exc).__name__))
                event_queue.put(("error", f"{type(exc).__name__}: {exc}"))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            kind, payload = event_queue.get()
            if kind == "status":
                yield payload
                continue
            if kind == "text":
                yield payload
                continue
            if kind == "result":
                if not streamed_text_seen[0]:
                    yield payload
                break
            if kind == "error":
                yield {
                    "event": {
                        "type": "status",
                        "data": {
                            "description": f"执行失败: {payload}",
                            "done": True,
                        },
                    }
                }
                yield f"执行失败: {payload}"
                break

    def _merge_cached_context(self, body: dict, user_message: str) -> dict:
        """Merge cached inlet messages/files into current body when needed."""
        merged = dict(body or {})
        request_kind = self._classify_text(user_message)
        chat_id = self._resolve_effective_chat_id(merged, request_kind, user_message)
        if not chat_id:
            return merged
        merged["_resolved_chat_id"] = chat_id
        cached = self._inlet_context_by_chat_id.get(chat_id)
        if not isinstance(cached, dict):
            return merged

        body_files = merged.get("files")
        if not isinstance(body_files, list) or not body_files:
            cached_files = cached.get("files")
            if isinstance(cached_files, list) and cached_files:
                merged["files"] = copy.deepcopy(cached_files)

        if request_kind != "chat":
            cached_messages = cached.get("messages")
            if isinstance(cached_messages, list) and cached_messages:
                merged["_captured_messages"] = copy.deepcopy(cached_messages)

        self.logger.debug(
            build_log_message(
                "pipeline",
                "pipe_merge_cached_context",
                chat_id=chat_id,
                request_kind=request_kind,
                merged_file_count=len(merged.get("files", []) if isinstance(merged.get("files"), list) else []),
                has_captured_messages=isinstance(merged.get("_captured_messages"), list),
            )
        )
        return merged

    def _resolve_effective_chat_id(self, body: dict, request_kind: str, user_message: str) -> str:
        """Resolve stable chat_id for both chat and meta task requests."""
        extracted_chat_id = self._extract_chat_id(body)
        now = time.time()

        if request_kind == "chat":
            matched_chat_id = self._find_recent_chat_id_for_user_text(user_message, now)
            if matched_chat_id:
                if extracted_chat_id != matched_chat_id:
                    self.logger.info(
                        build_log_message(
                            "pipeline",
                            "pipe_chat_id_overridden",
                            request_kind=request_kind,
                            extracted_chat_id=extracted_chat_id,
                            effective_chat_id=matched_chat_id,
                        )
                    )
                return matched_chat_id
            return extracted_chat_id

        meta_cache = self._latest_meta_chat_id_by_kind.get(request_kind, {})
        cached_chat_id = str(meta_cache.get("chat_id") or "")
        cached_at = float(meta_cache.get("cached_at") or 0)
        if cached_chat_id and now - cached_at <= 300:
            if extracted_chat_id != cached_chat_id:
                self.logger.info(
                    build_log_message(
                        "pipeline",
                        "pipe_chat_id_overridden",
                        request_kind=request_kind,
                        extracted_chat_id=extracted_chat_id,
                        effective_chat_id=cached_chat_id,
                    )
                )
            return cached_chat_id
        latest_chat_id = self._latest_recent_chat_id(now)
        if latest_chat_id:
            if extracted_chat_id != latest_chat_id:
                self.logger.info(
                    build_log_message(
                        "pipeline",
                        "pipe_chat_id_overridden",
                        request_kind=request_kind,
                        extracted_chat_id=extracted_chat_id,
                        effective_chat_id=latest_chat_id,
                    )
                )
            return latest_chat_id
        return extracted_chat_id

    def _latest_recent_chat_id(self, now: float) -> str:
        best_chat_id = ""
        best_cached_at = 0.0
        for chat_id, payload in self._inlet_context_by_chat_id.items():
            cached_at = float(payload.get("cached_at") or 0)
            if now - cached_at > 300:
                continue
            if cached_at >= best_cached_at:
                best_chat_id = chat_id
                best_cached_at = cached_at
        return best_chat_id

    def _find_recent_chat_id_for_user_text(self, user_message: str, now: float) -> str:
        normalized = str(user_message or "").strip()
        if not normalized:
            return self._latest_recent_chat_id(now)
        best_chat_id = ""
        best_cached_at = 0.0
        for chat_id, payload in self._inlet_context_by_chat_id.items():
            cached_at = float(payload.get("cached_at") or 0)
            if now - cached_at > 300:
                continue
            cached_text = str(payload.get("last_user_text") or "")
            if cached_text != normalized:
                continue
            if cached_at >= best_cached_at:
                best_chat_id = chat_id
                best_cached_at = cached_at
        return best_chat_id

    def _extract_chat_id(self, body: dict) -> str:
        containers = [body]
        if isinstance(body, dict):
            for key in ["metadata", "user", "session", "context", "info", "params"]:
                value = body.get(key)
                if isinstance(value, dict):
                    containers.append(value)
        primary_keys = ["chat_id", "chatId", "conversation_id", "conversationId", "session_id", "sessionId", "thread_id", "threadId"]
        fallback_keys = ["id"]
        for keys in [primary_keys, fallback_keys]:
            for container in containers:
                if not isinstance(container, dict):
                    continue
                for key in keys:
                    value = container.get(key)
                    if isinstance(value, str) and value.strip() and value.strip().lower() != "local":
                        return value.strip()
        return ""

    def _classify_request(self, body: dict) -> str:
        messages = body.get("messages", []) if isinstance(body.get("messages"), list) else []
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                return self._classify_text(flatten_content(message.get("content")))
        return "chat"

    def _latest_user_text(self, messages: List[dict]) -> str:
        for message in reversed(messages or []):
            if isinstance(message, dict) and message.get("role") == "user":
                return flatten_content(message.get("content")).strip()
        return ""

    def _classify_text(self, text: str) -> str:
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

    def _task_head(self, text: str) -> str:
        lowered = str(text or "").strip().lower()
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
