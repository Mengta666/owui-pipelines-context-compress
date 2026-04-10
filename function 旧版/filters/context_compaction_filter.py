"""
title: Context Compaction Middleware
author: Codex
version: 0.4.0
required_open_webui_version: 0.3.0
requirements: aiohttp,pydantic,tiktoken
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import mimetypes
import os
import re
import shutil
import time
import zipfile
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import aiohttp
from pydantic import BaseModel, Field

try:
    import tiktoken
except Exception:
    tiktoken = None


logger = logging.getLogger("context_compaction_filter")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


SUPPORTED_TEXT_EXTENSIONS = {
    ".md", ".txt", ".py", ".json", ".yaml", ".yml", ".csv", ".js", ".ts", ".tsx", ".jsx",
    ".html", ".css", ".scss", ".sql", ".toml", ".ini", ".cfg", ".env", ".sh", ".ps1", ".java",
    ".go", ".rs", ".c", ".cc", ".cpp", ".h", ".hpp", ".xml",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".svg", ".heic", ".heif"}


def now_ts() -> int:
    return int(time.time())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = str(item.get("type", "content"))
                if item_type in {"text", "input_text", "output_text"}:
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(f"[{item_type}]")
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


def serialize_tool_calls(tool_calls: Any) -> str:
    if not tool_calls:
        return ""
    try:
        return json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(tool_calls)


def trim_text(text: str, head_chars: int, tail_chars: int) -> str:
    if len(text) <= head_chars + tail_chars + 64:
        return text
    return text[:head_chars] + "\n\n[...middle content omitted...]\n\n" + text[-tail_chars:]


def trim_log(text: str, head_lines: int, tail_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= head_lines + tail_lines + 1:
        return text
    return "\n".join(lines[:head_lines]) + "\n\n[...system note: massive middle log omitted...]\n\n" + "\n".join(lines[-tail_lines:])


def display_path_name(path_str: str) -> str:
    return Path(path_str).name if path_str else ""


def render_message(message: Dict[str, Any]) -> str:
    parts = [f"kind={message.get('_kind', 'chat_message')}", f"role={message.get('role', 'user')}"]
    if message.get("name"):
        parts.append(f"name={message['name']}")
    if message.get("tool_call_id"):
        parts.append(f"tool_call_id={message['tool_call_id']}")
    if message.get("_source_name"):
        parts.append(f"source={message['_source_name']}")
    tool_calls = serialize_tool_calls(message.get("tool_calls"))
    if tool_calls:
        parts.append("tool_calls=" + tool_calls)
    content = flatten_content(message.get("content"))
    if content:
        parts.append(content)
    return "\n".join(parts)


def format_log_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        text = re.sub(r"\s+", " ", value).strip()
        if not text:
            return '""'
        if len(text) > 120:
            text = text[:117] + "..."
        if any(ch.isspace() for ch in text) or any(ch in text for ch in {"=", "[", "]", "{", "}", ","}):
            return json.dumps(text, ensure_ascii=False)
        return text
    if isinstance(value, dict):
        preview = ",".join(str(key) for key in list(value.keys())[:4])
        return f"<dict:{len(value)} keys={preview}>"
    if isinstance(value, (list, tuple, set)):
        return f"<{type(value).__name__}:{len(value)}>"
    return format_log_value(str(value))


def build_log_message(domain: str, event: str, **fields: Any) -> str:
    parts = [f"[{domain}.{event}]"]
    for key, value in fields.items():
        if value is ...:
            continue
        parts.append(f"{key}={format_log_value(value)}")
    return " ".join(parts)


class TokenCounter:
    def __init__(self, encoding_name: str) -> None:
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
            + self.count_text(serialize_tool_calls(message.get("tool_calls")))
        )

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        return sum(self.count_message(message) for message in messages)


class DiskMemoryStore:
    def __init__(self, root_dir: str) -> None:
        self.root = Path(root_dir)
        ensure_dir(self.root)

    def chat_dir(self, chat_id: str) -> Path:
        path = self.root / chat_id
        ensure_dir(path)
        return path

    def snapshot_path(self, chat_id: str) -> Path:
        return self.chat_dir(chat_id) / "snapshot.json"

    def transcript_dir(self, chat_id: str) -> Path:
        path = self.chat_dir(chat_id) / "transcripts"
        ensure_dir(path)
        return path

    def default_snapshot(self) -> Dict[str, Any]:
        return {
            "summary": "",
            "file_memory": {},
            "file_order": [],
            "last_seen_file_ids": [],
            "last_effective_tokens": 0,
            "last_compaction_reason": "",
            "last_cleanup_scan_at": 0,
            "last_compacted_message_anchor": "",
            "last_compacted_message_count": 0,
            "last_compacted_file_anchor": "",
            "last_compacted_file_count": 0,
            "consecutive_failures": 0,
            "updated_at": 0,
            "transcript_paths": [],
            "last_response_metrics": {},
        }

    def load_snapshot(self, chat_id: str) -> Dict[str, Any]:
        path = self.snapshot_path(chat_id)
        if not path.exists():
            return self.default_snapshot()
        try:
            data = json.loads(path.read_text("utf-8"))
        except Exception:
            logger.exception(build_log_message("state", "load_failed", chat_id=chat_id))
            return self.default_snapshot()
        snapshot = self.default_snapshot()
        if isinstance(data, dict):
            snapshot.update(data)
        return snapshot

    def save_snapshot(self, chat_id: str, snapshot: Dict[str, Any]) -> None:
        path = self.snapshot_path(chat_id)
        tmp = path.with_suffix(path.suffix + f".{uuid4().hex}.tmp")
        tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), "utf-8")
        os.replace(str(tmp), str(path))

    def write_transcript(self, chat_id: str, messages: List[Dict[str, Any]]) -> str:
        path = self.transcript_dir(chat_id) / f"{now_ts()}_{uuid4().hex[:8]}.md"
        rendered: List[str] = []
        for message in messages:
            rendered.extend([render_message(message), "", "-" * 60, ""])
        path.write_text("\n".join(rendered).strip(), "utf-8")
        return str(path)

    def delete_chat_state(self, chat_id: str) -> None:
        path = self.root / chat_id
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            logger.info(build_log_message("state", "delete", chat_id=chat_id, deleted=True))

    def iter_chat_dirs(self) -> List[Path]:
        if not self.root.exists():
            return []
        return [path for path in self.root.iterdir() if path.is_dir()]


class Filter:
    class Valves(BaseModel):
        enabled: bool = Field(default=True)
        max_context_tokens: int = Field(default=32768)
        reserved_output_tokens: int = Field(default=8000)
        autocompact_buffer_tokens: int = Field(default=4000)
        max_consecutive_failures: int = Field(default=3)
        keep_first_non_system: int = Field(default=4)
        keep_last_messages: int = Field(default=8)
        emergency_tail_keep_messages: int = Field(default=4)
        keep_summary_as_system_message: bool = Field(default=True)
        max_inline_chars: int = Field(default=6000)
        log_head_lines: int = Field(default=80)
        log_tail_lines: int = Field(default=80)
        tokenizer_encoding: str = Field(default="o200k_base")
        inline_small_file_tokens: int = Field(default=1200)
        file_summary_source_max_chars: int = Field(default=24000)
        file_excerpt_chars: int = Field(default=1200)
        summarize_large_files: bool = Field(default=True)
        takeover_file_context: bool = Field(default=True)
        drop_original_files_on_takeover: bool = Field(default=True)
        persist_inline_file_text: bool = Field(default=False)
        file_virtual_message_role: str = Field(default="system")
        max_total_file_context_tokens: int = Field(default=6000)
        always_include_file_summaries: bool = Field(default=True)
        image_passthrough_token_cost: int = Field(default=1000)
        prune_unseen_file_memory: bool = Field(default=True)
        strip_file_related_features_on_takeover: bool = Field(default=True)
        summary_model: str = Field(default="")
        summary_temperature: float = Field(default=0.1)
        summary_max_tokens: int = Field(default=2200)
        internal_base_url: str = Field(default=os.getenv("OPEN_WEBUI_INTERNAL_URL", "http://127.0.0.1:8080"))
        internal_api_key: str = Field(default=os.getenv("OPEN_WEBUI_API_KEY", ""))
        request_timeout_sec: int = Field(default=120)
        chat_reconcile_base_url: str = Field(default=os.getenv("OPEN_WEBUI_CHAT_RECONCILE_URL", ""))
        chat_reconcile_api_key: str = Field(default=os.getenv("OPEN_WEBUI_CHAT_RECONCILE_API_KEY", ""))
        chat_reconcile_timeout_sec: int = Field(default=15)
        memory_dir: str = Field(default=os.getenv("OPEN_WEBUI_CONTEXT_MEMORY_DIR", "/app/backend/data/cache/context_compaction"))
        snapshot_title: str = Field(default="[[Context Snapshot]]")
        uploads_root: str = Field(default=os.getenv("OPEN_WEBUI_UPLOADS_DIR", ""))
        enforce_upload_root: bool = Field(default=False)
        sync_deleted_chats: bool = Field(default=True)
        cleanup_check_interval_sec: int = Field(default=60)
        cleanup_batch_size: int = Field(default=10)
        cleanup_stale_ephemeral_after_sec: int = Field(default=86400)
        force_cleanup_on_every_request: bool = Field(default=False)
        debug: bool = Field(default=True)

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.store = DiskMemoryStore(self.valves.memory_dir)
        self.token_counter = TokenCounter(self.valves.tokenizer_encoding)
        self._last_cleanup_ts = 0

    async def inlet(self, body: Dict[str, Any], __user__: Optional[Dict[str, Any]] = None, __metadata__: Optional[Dict[str, Any]] = None, __event_emitter__=None) -> Dict[str, Any]:
        if not self.valves.enabled:
            return body
        raw_messages = body.get("messages", [])
        if not isinstance(raw_messages, list) or not raw_messages:
            return body

        chat_id = self._resolve_chat_id(body, __metadata__)
        snapshot = self.store.load_snapshot(chat_id)
        await self._reconcile_current_chat_state(chat_id)
        snapshot = await self._maybe_cleanup_states(chat_id, snapshot)

        effective = await self._build_effective_context(body, snapshot)
        current_tokens = int(effective.get("effective_tokens") or self.token_counter.count_messages(effective["virtual_messages"]))
        threshold = self._compact_threshold()

        if self.valves.debug:
            logger.info(build_log_message(
                "context", "evaluate",
                chat_id=chat_id,
                msg_count=len(raw_messages),
                file_count=int(effective.get("request_file_count", 0) or 0),
                message_file_count=int(effective.get("message_file_count", 0) or 0),
                image_content_count=int(effective.get("message_image_count", 0) or 0),
                compressed_file_count=len(effective.get("file_contexts", []) or []),
                passthrough_file_count=int(effective.get("passthrough_file_count", 0) or 0),
                effective_items=len(effective["virtual_messages"]),
                passthrough_token_cost=int(effective.get("passthrough_token_cost", 0) or 0),
                message_media_token_cost=int(effective.get("message_media_token_cost", 0) or 0),
                tokens=current_tokens,
                threshold=threshold,
            ))

        if current_tokens < threshold:
            snapshot = self._merge_snapshot_metadata(snapshot, effective["file_contexts"], current_tokens, "below_threshold")
            snapshot["updated_at"] = now_ts()
            self.store.save_snapshot(chat_id, snapshot)
            return self._apply_final_context(body, effective, effective["virtual_messages"])

        if snapshot.get("consecutive_failures", 0) >= self.valves.max_consecutive_failures:
            snapshot = self._merge_snapshot_metadata(snapshot, effective["file_contexts"], current_tokens, "circuit_open")
            snapshot["updated_at"] = now_ts()
            self.store.save_snapshot(chat_id, snapshot)
            await self._emit_status(__event_emitter__, "Context compaction skipped because circuit breaker is open.", True)
            return self._apply_final_context(body, effective, effective["virtual_messages"])

        await self._emit_status(__event_emitter__, f"Context near limit ({current_tokens} tokens), compacting.", False)
        try:
            rebuilt, new_snapshot = await self._compact(chat_id, effective["virtual_messages"], snapshot, __event_emitter__, body.get("model"))
            rebuilt = self._enforce_hard_limit(rebuilt, int(effective.get("passthrough_token_cost", 0) or 0) + int(effective.get("message_media_token_cost", 0) or 0))
            new_snapshot = self._merge_snapshot_metadata(new_snapshot, effective["file_contexts"], current_tokens, "threshold_exceeded")
            new_snapshot["consecutive_failures"] = 0
            new_snapshot["updated_at"] = now_ts()
            self.store.save_snapshot(chat_id, new_snapshot)
            body = self._apply_final_context(body, effective, rebuilt)
            await self._emit_status(__event_emitter__, f"Context compacted from {current_tokens} to {self.token_counter.count_messages(body['messages'])} tokens.", True)
            return body
        except Exception as exc:
            logger.exception(build_log_message("compact", "failed", chat_id=chat_id))
            snapshot = self._merge_snapshot_metadata(snapshot, effective["file_contexts"], current_tokens, "compaction_failed")
            snapshot["consecutive_failures"] = snapshot.get("consecutive_failures", 0) + 1
            snapshot["updated_at"] = now_ts()
            self.store.save_snapshot(chat_id, snapshot)
            await self._emit_status(__event_emitter__, f"Context compaction failed: {type(exc).__name__}.", True)
            return self._apply_final_context(body, effective, effective["virtual_messages"])

    async def outlet(self, body: Dict[str, Any], __user__: Optional[Dict[str, Any]] = None, __metadata__: Optional[Dict[str, Any]] = None, __event_emitter__=None) -> Dict[str, Any]:
        chat_id = self._resolve_chat_id(body, __metadata__)
        snapshot = self.store.load_snapshot(chat_id)
        last_assistant = self._find_last_message(body.get("messages"), "assistant")
        snapshot["last_response_metrics"] = {"assistant_preview": trim_text(flatten_content((last_assistant or {}).get("content")), 300, 300), "updated_at": now_ts()}
        snapshot["updated_at"] = now_ts()
        self.store.save_snapshot(chat_id, snapshot)
        return body

    async def stream(self, event: Dict[str, Any], **_: Any) -> Dict[str, Any]:
        return event

    async def _compact(self, chat_id: str, messages: List[Dict[str, Any]], snapshot: Dict[str, Any], emitter=None, request_model: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        pinned, early, middle, recent = self._split_context(messages)
        message_digests = [self._message_digest(item) for item in middle]
        delta_start = self._resolve_delta_start(message_digests, str(snapshot.get("last_compacted_message_anchor", "") or ""), int(snapshot.get("last_compacted_message_count", 0) or 0))
        delta_messages = middle[delta_start:]

        transcript_path = ""
        summary = str(snapshot.get("summary", "") or "").strip()
        if delta_messages:
            transcript_path = self.store.write_transcript(chat_id, delta_messages)
            summary = await self._merge_summary(summary, delta_messages, transcript_path, request_model)

        new_snapshot = dict(snapshot)
        if summary:
            new_snapshot["summary"] = summary
        if transcript_path:
            transcript_paths = list(snapshot.get("transcript_paths", []) or [])
            transcript_paths.append(transcript_path)
            new_snapshot["transcript_paths"] = transcript_paths[-20:]
        if message_digests:
            new_snapshot["last_compacted_message_anchor"] = message_digests[-1]
        new_snapshot["last_compacted_message_count"] = len(middle)

        rebuilt = list(pinned)
        snapshot_item = self._snapshot_context_item(new_snapshot)
        if snapshot_item is not None:
            rebuilt.append(snapshot_item)
        rebuilt.extend(early)
        rebuilt.extend(recent)
        rebuilt = self._dedupe_messages(rebuilt)

        if self.valves.debug:
            logger.info(build_log_message("compact", "split", pinned=len(pinned), early=len(early), middle=len(middle), recent=len(recent), delta=len(delta_messages)))
            logger.info(build_log_message("compact", "rebuilt", messages=len(rebuilt), transcript=display_path_name(transcript_path)))
        return rebuilt, new_snapshot

    def _split_context(self, items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        pinned_kinds = {"system", "memory_snapshot", "file_context"}
        pinned = [item for item in items if item.get("_kind") in pinned_kinds]
        rolling = [item for item in items if item.get("_kind") not in pinned_kinds]
        total = len(rolling)
        early_end = min(total, self.valves.keep_first_non_system)
        recent_start = max(early_end, total - self.valves.keep_last_messages)
        return pinned, rolling[:early_end], rolling[early_end:recent_start], rolling[recent_start:]

    def _resolve_delta_start(self, digests: List[str], saved_anchor: str, saved_count: int) -> int:
        if saved_anchor and saved_anchor in digests:
            return digests.index(saved_anchor) + 1
        if 0 <= saved_count <= len(digests):
            return saved_count
        return 0

    def _build_snapshot_item(self, summary: str, transcript_name: str) -> Dict[str, Any]:
        parts = [self.valves.snapshot_title, "", summary.strip()]
        if transcript_name:
            parts.extend(["", f"Transcript: {transcript_name}"])
        return {"role": "system" if self.valves.keep_summary_as_system_message else "assistant", "content": "\n".join(parts), "_kind": "memory_snapshot", "_source_name": transcript_name}

    def _snapshot_context_item(self, snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        summary = str(snapshot.get("summary", "") or "").strip()
        if not summary:
            return None
        return self._build_snapshot_item(summary, display_path_name(self._last_transcript(snapshot)))

    def _merge_snapshot_metadata(self, snapshot: Dict[str, Any], file_contexts: List[Dict[str, Any]], current_tokens: int, reason: str) -> Dict[str, Any]:
        merged = dict(snapshot)
        file_memory = {} if self.valves.prune_unseen_file_memory else dict(snapshot.get("file_memory", {}) or {})
        for record in file_contexts:
            persisted = dict(record)
            if not self.valves.persist_inline_file_text:
                persisted["inline_text"] = ""
            file_memory[record["file_id"]] = persisted
        merged["file_memory"] = file_memory
        merged["file_order"] = [record["file_id"] for record in file_contexts]
        merged["last_seen_file_ids"] = [record["file_id"] for record in file_contexts]
        merged["last_effective_tokens"] = current_tokens
        merged["last_compaction_reason"] = reason
        merged["last_cleanup_scan_at"] = self._last_cleanup_ts
        return merged

    async def _build_effective_context(self, body: Dict[str, Any], snapshot: Dict[str, Any]) -> Dict[str, Any]:
        raw_messages = body.get("messages", [])
        normalized_messages = [self._normalize_message(message) for message in raw_messages if isinstance(message, dict)]
        file_info = self._collect_request_files(body)
        normalized_files = await self._normalize_uploaded_files(file_info["raw_files"], snapshot, body.get("model"))

        system_messages = [message for message in normalized_messages if message.get("_kind") == "system"]
        rolling_messages = [message for message in normalized_messages if message.get("_kind") != "system"]
        file_virtual_messages = self._select_file_virtual_messages(normalized_files["file_contexts"])
        virtual_messages = system_messages + file_virtual_messages + rolling_messages

        message_image_count = self._message_image_count(normalized_messages)
        message_media_token_cost = self._estimate_message_media_tokens(normalized_messages)
        passthrough_token_cost = int(normalized_files["passthrough_token_cost"])
        effective_tokens = self.token_counter.count_messages(virtual_messages) + passthrough_token_cost + message_media_token_cost

        return {
            "raw_messages": raw_messages,
            "virtual_messages": virtual_messages,
            "file_contexts": normalized_files["file_contexts"],
            "file_virtual_messages": file_virtual_messages,
            "passthrough_files": normalized_files["passthrough_files"],
            "passthrough_file_ids": normalized_files["passthrough_file_ids"],
            "passthrough_file_count": len(normalized_files["passthrough_files"]),
            "passthrough_token_cost": passthrough_token_cost,
            "request_file_count": int(file_info["request_file_count"]),
            "message_file_count": int(file_info["message_file_count"]),
            "message_image_count": message_image_count,
            "message_media_token_cost": message_media_token_cost,
            "effective_tokens": effective_tokens,
        }

    async def _normalize_uploaded_files(self, raw_files: List[Dict[str, Any]], snapshot: Dict[str, Any], request_model: Optional[str]) -> Dict[str, Any]:
        file_memory = dict(snapshot.get("file_memory", {}) or {})
        file_contexts: List[Dict[str, Any]] = []
        passthrough_files: List[Dict[str, Any]] = []
        passthrough_file_ids: List[str] = []
        passthrough_token_cost = 0

        for raw in raw_files:
            meta = self._extract_file_meta(raw)
            file_id = str(meta.get("file_id", "") or "").strip()
            if not file_id:
                continue

            cached = file_memory.get(file_id)
            path_str = self._resolve_safe_file_path(meta)
            fingerprint = self._compute_file_fingerprint(meta, path_str)

            if self._should_passthrough_file(meta, path_str):
                passthrough_files.append(raw)
                passthrough_file_ids.append(file_id)
                passthrough_token_cost += self._passthrough_token_estimate(meta)
                if self.valves.debug:
                    logger.info(build_log_message(
                        "file", "passthrough",
                        file_id=file_id,
                        filename=meta.get("filename"),
                        content_type=meta.get("content_type"),
                        reason="non_text_or_media",
                    ))
                continue

            text = self._load_supported_file_text(path_str, meta)
            if not text and isinstance(cached, dict):
                restored = self._restore_cached_file_record(cached, meta, fingerprint)
                if restored is not None:
                    file_contexts.append(restored)
                    if self.valves.debug:
                        logger.info(build_log_message(
                            "file", "restore",
                            file_id=file_id,
                            filename=restored.get("filename"),
                            mode=restored.get("mode"),
                            cached=True,
                        ))
                    continue

            if not text:
                passthrough_files.append(raw)
                passthrough_file_ids.append(file_id)
                passthrough_token_cost += self._passthrough_token_estimate(meta)
                if self.valves.debug:
                    logger.info(build_log_message(
                        "file", "skip_path",
                        file_id=file_id,
                        filename=meta.get("filename"),
                        path=path_str or "missing",
                    ))
                continue

            token_est = self.token_counter.count_text(text)
            inline_text = trim_text(text, self.valves.max_inline_chars, self.valves.file_excerpt_chars)
            cached_fingerprint = str((cached or {}).get("fingerprint", "") or "")
            cached_summary = str((cached or {}).get("summary_text", "") or "")
            cached_inline = str((cached or {}).get("inline_text", "") or "")
            if cached_fingerprint == fingerprint and cached_inline:
                inline_text = cached_inline

            if token_est <= self.valves.inline_small_file_tokens:
                mode = "inline"
                summary_text = ""
            else:
                mode = "summary" if self.valves.summarize_large_files else "metadata"
                if mode == "summary" and cached_fingerprint == fingerprint and cached_summary:
                    summary_text = cached_summary
                elif mode == "summary":
                    summary_text = await self._summarize_file_once(meta, text, request_model)
                else:
                    summary_text = ""

            record = {
                "file_id": file_id,
                "filename": str(meta.get("filename", "") or file_id),
                "content_type": str(meta.get("content_type", "") or ""),
                "path": path_str,
                "fingerprint": fingerprint,
                "token_est": token_est,
                "inline_text": inline_text,
                "summary_text": summary_text,
                "mode": mode,
            }
            file_contexts.append(record)
            if self.valves.debug:
                logger.info(build_log_message(
                    "file", "load",
                    file_id=file_id,
                    filename=record["filename"],
                    mode=mode,
                    token_est=token_est,
                    cached=cached_fingerprint == fingerprint,
                ))

        return {
            "file_contexts": file_contexts,
            "passthrough_files": passthrough_files,
            "passthrough_file_ids": passthrough_file_ids,
            "passthrough_token_cost": passthrough_token_cost,
        }

    def _restore_cached_file_record(self, cached: Dict[str, Any], meta: Dict[str, Any], fingerprint: str) -> Optional[Dict[str, Any]]:
        if not isinstance(cached, dict):
            return None
        restored = dict(cached)
        restored["file_id"] = str(meta.get("file_id", restored.get("file_id", "")) or "")
        restored["filename"] = str(meta.get("filename", restored.get("filename", "")) or restored.get("file_id", ""))
        restored["content_type"] = str(meta.get("content_type", restored.get("content_type", "")) or "")
        restored["path"] = self._resolve_safe_file_path(meta) or str(restored.get("path", "") or "")
        restored["fingerprint"] = fingerprint or str(restored.get("fingerprint", "") or "")
        restored["mode"] = str(restored.get("mode", "metadata") or "metadata")
        restored["token_est"] = int(restored.get("token_est", 0) or 0)
        restored["inline_text"] = str(restored.get("inline_text", "") or "")
        restored["summary_text"] = str(restored.get("summary_text", "") or "")
        if not (restored["inline_text"] or restored["summary_text"]):
            return None
        return restored

    def _should_passthrough_file(self, meta: Dict[str, Any], path_str: str) -> bool:
        content_type = str(meta.get("content_type", "") or "").lower()
        filename = str(meta.get("filename", "") or "")
        suffix = Path(filename).suffix.lower()
        if self._is_image_file(meta):
            return True
        if content_type.startswith("audio/") or content_type.startswith("video/"):
            return True
        if suffix == ".docx":
            return False
        if suffix in SUPPORTED_TEXT_EXTENSIONS:
            return False
        if content_type.startswith("text/"):
            return False
        if any(tag in content_type for tag in ["json", "xml", "yaml", "csv", "javascript"]):
            return False
        if suffix == ".doc":
            return not bool(path_str)
        return True

    def _passthrough_token_estimate(self, meta: Dict[str, Any]) -> int:
        if self._is_image_file(meta):
            return self.valves.image_passthrough_token_cost
        return min(self.valves.image_passthrough_token_cost, 512)

    def _is_image_file(self, meta: Dict[str, Any]) -> bool:
        content_type = str(meta.get("content_type", "") or "").lower()
        filename = str(meta.get("filename", "") or "")
        suffix = Path(filename).suffix.lower()
        return content_type.startswith("image/") or suffix in IMAGE_EXTENSIONS

    def _select_file_virtual_messages(self, file_contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        remaining = self.valves.max_total_file_context_tokens
        for record in file_contexts:
            chosen: Optional[Dict[str, Any]] = None
            chosen_tokens = 0
            for candidate in self._file_virtual_message_candidates(record):
                token_est = self.token_counter.count_message(candidate)
                if token_est <= remaining:
                    chosen = candidate
                    chosen_tokens = token_est
                    break
                if chosen is None:
                    chosen = candidate
                    chosen_tokens = token_est
            if chosen is None:
                continue
            if chosen_tokens <= remaining or not selected:
                remaining = max(0, remaining - chosen_tokens)
            selected.append(chosen)
            if self.valves.debug and chosen_tokens > remaining and remaining == 0:
                logger.info(build_log_message("file", "skip_budget", file_id=record.get("file_id"), filename=record.get("filename"), mode=record.get("mode")))
        return selected

    def _file_virtual_message_candidates(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        if record.get("inline_text"):
            candidates.append(self._build_file_virtual_message(record, "inline", record["inline_text"]))
        if record.get("summary_text"):
            candidates.append(self._build_file_virtual_message(record, "summary", self._file_summary_text(record)))
        candidates.append(self._build_file_virtual_message(
            record,
            "metadata",
            f"Attached file metadata\nfilename: {record.get('filename', '')}\ncontent_type: {record.get('content_type', '') or 'unknown'}\nmode: omitted\ncontent omitted",
        ))
        return candidates

    def _file_virtual_role(self) -> str:
        role = str(self.valves.file_virtual_message_role or "system").strip().lower()
        if role not in {"system", "user", "assistant"}:
            return "system"
        return role

    def _file_summary_text(self, record: Dict[str, Any]) -> str:
        summary = str(record.get("summary_text", "") or "").strip()
        if summary:
            return summary
        inline_text = str(record.get("inline_text", "") or "").strip()
        if inline_text:
            return trim_text(inline_text, self.valves.file_excerpt_chars, self.valves.file_excerpt_chars)
        return "Content omitted."

    def _extract_file_meta(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        file_obj = raw.get("file") if isinstance(raw, dict) else {}
        file_obj = file_obj if isinstance(file_obj, dict) else {}
        meta = file_obj.get("meta") if isinstance(file_obj.get("meta"), dict) else {}
        return {
            "file_id": self._pick_first_value(raw, file_obj, keys=["id", "url", "itemId"]),
            "filename": self._pick_first_value(raw, file_obj, meta, keys=["filename", "name"]),
            "content_type": self._pick_first_value(raw, file_obj, meta, keys=["content_type"]),
            "path": self._pick_first_value(raw, file_obj, keys=["path"]),
        }

    def _pick_first_value(self, *containers: Any, keys: List[str]) -> str:
        for container in containers:
            if not isinstance(container, dict):
                continue
            for key in keys:
                value = container.get(key)
                if value not in (None, ""):
                    return str(value)
        return ""

    def _resolve_safe_file_path(self, meta: Dict[str, Any]) -> str:
        path_str = str(meta.get("path", "") or "").strip()
        if not path_str:
            return ""
        if not self.valves.enforce_upload_root:
            return path_str
        root = str(self.valves.uploads_root or "").strip()
        if not root:
            return path_str
        try:
            path = Path(path_str).resolve()
            upload_root = Path(root).resolve()
            path.relative_to(upload_root)
            return str(path)
        except Exception:
            return ""

    def _compute_file_fingerprint(self, meta: Dict[str, Any], path_str: str) -> str:
        if path_str and Path(path_str).exists():
            stat = Path(path_str).stat()
            base = f"{meta.get('file_id', '')}|{stat.st_size}|{int(stat.st_mtime)}"
            return sha1_text(base)[:12]
        base = f"{meta.get('file_id', '')}|{meta.get('filename', '')}|{meta.get('content_type', '')}"
        return sha1_text(base)[:12]

    def _load_supported_file_text(self, path_str: str, meta: Dict[str, Any]) -> str:
        if not path_str:
            return ""
        path = Path(path_str)
        if not path.exists() or not path.is_file():
            return ""
        suffix = path.suffix.lower()
        if suffix == ".docx":
            return self._load_docx_text(path)
        max_chars = self.valves.file_summary_source_max_chars
        if suffix not in SUPPORTED_TEXT_EXTENSIONS and suffix != ".doc":
            content_type = str(meta.get("content_type", "") or "").lower()
            if not content_type.startswith("text/") and not any(tag in content_type for tag in ["json", "xml", "yaml", "csv", "javascript"]):
                return ""
        for encoding in ["utf-8", "utf-8-sig", "gb18030", "latin-1"]:
            try:
                text = path.read_text(encoding=encoding)
                return text[:max_chars]
            except Exception:
                continue
        try:
            data = path.read_bytes()[: max_chars * 4]
            return data.decode("utf-8", errors="ignore")[:max_chars]
        except Exception:
            return ""

    def _load_docx_text(self, path: Path) -> str:
        try:
            with zipfile.ZipFile(path, "r") as zf:
                xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
        except Exception:
            return ""
        xml = re.sub(r"</w:p>", "\n", xml)
        xml = re.sub(r"<[^>]+>", "", xml)
        text = unescape(xml)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text[: self.valves.file_summary_source_max_chars]

    def _build_file_virtual_message(self, record: Dict[str, Any], mode: str, content: str) -> Dict[str, Any]:
        return {
            "role": self._file_virtual_role(),
            "content": f"[Attached file context]\nfilename: {record.get('filename', '')}\ncontent_type: {record.get('content_type', '') or 'unknown'}\nmode: {mode}\n\n{content.strip()}",
            "_kind": "file_context",
            "_source_name": str(record.get("filename", "") or record.get("file_id", "")),
            "_file_id": str(record.get("file_id", "") or ""),
        }

    async def _summarize_file_once(self, meta: Dict[str, Any], text: str, request_model: Optional[str]) -> str:
        excerpt = text[: self.valves.file_summary_source_max_chars]
        try:
            raw = await self._call_summary_model(
                self._build_summary_prompt(
                    [
                        {
                            "role": "user",
                            "content": f"Please summarize the following file.\nfilename: {meta.get('filename', '')}\ncontent_type: {meta.get('content_type', '') or 'unknown'}\n\n{excerpt}",
                        }
                    ]
                ),
                request_model,
            )
            summary = self._extract_summary(raw)
            if summary:
                return summary
        except Exception:
            logger.exception(build_log_message("file", "summary_failed", filename=meta.get("filename")))
        return self._fallback_file_summary(meta, excerpt)

    def _fallback_file_summary(self, meta: Dict[str, Any], text: str) -> str:
        excerpt = trim_text(text.strip(), self.valves.file_excerpt_chars, self.valves.file_excerpt_chars)
        return "\n".join(
            part
            for part in [
                f"File: {meta.get('filename', '')}",
                f"Type: {meta.get('content_type', '') or 'unknown'}",
                "Summary unavailable. Excerpt:",
                excerpt,
            ]
            if part
        )

    async def _merge_summary(self, previous_summary: str, delta_messages: List[Dict[str, Any]], transcript_path: str, request_model: Optional[str]) -> str:
        strict_prompt = self._build_summary_prompt(delta_messages, previous_summary, transcript_path)
        try:
            raw = await self._call_summary_model(strict_prompt, request_model)
            summary = self._extract_summary(raw)
            if self._validate_summary_shape(summary):
                return summary
        except Exception:
            logger.exception(build_log_message("summary", "strict_failed", transcript=display_path_name(transcript_path)))

        try:
            raw = await self._call_summary_model(self._build_loose_summary_prompt(delta_messages, previous_summary, transcript_path), request_model)
            summary = self._extract_plain_summary(raw)
            if summary:
                return summary
        except Exception:
            logger.exception(build_log_message("summary", "loose_failed", transcript=display_path_name(transcript_path)))

        return self._fallback_delta_summary(previous_summary, delta_messages, transcript_path)

    def _build_summary_prompt(self, delta_messages: List[Dict[str, Any]], previous_summary: str = "", transcript_path: str = "") -> str:
        rendered = "\n\n".join(render_message(message) for message in delta_messages[-24:])
        instructions = [
            "Merge the prior summary and the new conversation delta into a compact, structured memory.",
            "Use exactly these headers:",
            "Goals:",
            "Decisions:",
            "Open Items:",
            "Current State:",
            "Keep facts concrete and omit filler.",
        ]
        if transcript_path:
            instructions.append(f"Transcript: {display_path_name(transcript_path)}")
        return "\n".join(
            [
                "\n".join(instructions),
                "",
                "Prior Summary:",
                previous_summary.strip() or "(none)",
                "",
                "Delta:",
                rendered or "(empty)",
            ]
        )

    def _build_loose_summary_prompt(self, delta_messages: List[Dict[str, Any]], previous_summary: str = "", transcript_path: str = "") -> str:
        rendered = "\n\n".join(render_message(message) for message in delta_messages[-20:])
        prompt_parts = [
            "Summarize the new delta and merge it with the prior summary.",
            "Return plain text only.",
            "Cover the user's goals, decisions, open questions, and the latest working state.",
            "",
            "Prior Summary:",
            previous_summary.strip() or "(none)",
        ]
        if transcript_path:
            prompt_parts.extend(["", f"Transcript: {display_path_name(transcript_path)}"])
        prompt_parts.extend(["", "Delta:", rendered or "(empty)"])
        return "\n".join(prompt_parts)

    def _fallback_delta_summary(self, previous_summary: str, delta_messages: List[Dict[str, Any]], transcript_path: str) -> str:
        head = [trim_text(render_message(message), 400, 200) for message in delta_messages[:3]]
        tail = [trim_text(render_message(message), 400, 200) for message in delta_messages[-3:]]
        sections = [
            "Goals:",
            self._extract_summary_section(previous_summary, "Goals") or "- Preserve prior goals.",
            "",
            "Decisions:",
            "- Summary model unavailable; using transcript excerpt fallback.",
            "",
            "Open Items:",
            "- Verify details from transcript excerpt if needed.",
            "",
            "Current State:",
            "\n".join(head + tail) or "- No delta captured.",
        ]
        if transcript_path:
            sections.extend(["", f"Transcript: {display_path_name(transcript_path)}"])
        return "\n".join(sections).strip()

    async def _call_summary_model(self, prompt: str, request_model: Optional[str]) -> str:
        model = str(self.valves.summary_model or request_model or "").strip()
        if not model:
            raise RuntimeError("summary_model_missing")
        base_url = str(self.valves.internal_base_url or "").rstrip("/")
        if not base_url:
            raise RuntimeError("internal_base_url_missing")
        url = f"{base_url}/api/chat/completions"
        headers = {"Content-Type": "application/json"}
        api_key = str(self.valves.internal_api_key or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "stream": False,
            "temperature": self.valves.summary_temperature,
            "max_tokens": self.valves.summary_max_tokens,
            "messages": [{"role": "system", "content": "You produce compact conversation memory."}, {"role": "user", "content": prompt}],
        }
        timeout = aiohttp.ClientTimeout(total=self.valves.request_timeout_sec)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                text = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"summary_http_{response.status}: {self._response_detail_text(text, response.status)}")
                data = json.loads(text)
        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("summary_empty_choices")
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        return flatten_content(content)

    def _extract_summary(self, raw: Any) -> str:
        if isinstance(raw, str):
            text = raw.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    payload = json.loads(text)
                    summary = payload.get("summary")
                    if isinstance(summary, str) and summary.strip():
                        return summary.strip()
                except Exception:
                    pass
            return text
        if isinstance(raw, dict):
            summary = raw.get("summary")
            if isinstance(summary, str):
                return summary.strip()
        return ""

    def _extract_plain_summary(self, raw: Any) -> str:
        return self._extract_summary(raw)

    def _summary_headers(self) -> List[str]:
        return ["Goals:", "Decisions:", "Open Items:", "Current State:"]

    def _extract_summary_section(self, summary: str, header: str) -> str:
        pattern = re.escape(header.rstrip(":")) + r":\s*(.*?)(?=\n[A-Z][^:\n]+:\s|\Z)"
        match = re.search(pattern, summary, flags=re.S)
        return match.group(1).strip() if match else ""

    def _validate_summary_shape(self, summary: str) -> bool:
        if not summary.strip():
            return False
        return all(header in summary for header in self._summary_headers())

    async def _reconcile_current_chat_state(self, chat_id: str) -> None:
        if not self.valves.sync_deleted_chats or not chat_id or chat_id == "ephemeral":
            return
        exists = await self._resolve_chat_existence(chat_id)
        if self.valves.debug:
            logger.info(build_log_message("cleanup", "current", chat_id=chat_id, exists=exists))
        if exists is False:
            self.store.delete_chat_state(chat_id)

    async def _maybe_cleanup_states(self, chat_id: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        if not self.valves.sync_deleted_chats:
            return snapshot
        now = now_ts()
        if not self.valves.force_cleanup_on_every_request and now - self._last_cleanup_ts < self.valves.cleanup_check_interval_sec:
            return snapshot
        deleted = 0
        scanned = 0
        for path in self.store.iter_chat_dirs()[: self.valves.cleanup_batch_size]:
            state_chat_id = path.name
            if state_chat_id == chat_id:
                continue
            scanned += 1
            exists = await self._resolve_chat_existence(state_chat_id)
            if exists is False:
                self.store.delete_chat_state(state_chat_id)
                deleted += 1
        self._last_cleanup_ts = now
        snapshot = dict(snapshot)
        snapshot["last_cleanup_scan_at"] = now
        if self.valves.debug:
            logger.info(build_log_message("cleanup", "batch", current=chat_id, scanned=scanned, deleted=deleted, last_cleanup_ts=now))
        return snapshot

    async def _resolve_chat_existence(self, chat_id: str) -> Optional[bool]:
        base_url = str(self.valves.chat_reconcile_base_url or self.valves.internal_base_url or "").rstrip("/")
        if not base_url:
            return None
        headers: Dict[str, str] = {}
        api_key = str(self.valves.chat_reconcile_api_key or self.valves.internal_api_key or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = f"{base_url}/api/v1/chats/{chat_id}"
        timeout = aiohttp.ClientTimeout(total=self.valves.chat_reconcile_timeout_sec)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    text = await response.text()
                    detail = self._response_detail_text(text, response.status)
                    if response.status == 200:
                        if self.valves.debug:
                            logger.info(build_log_message("cleanup", "check", chat_id=chat_id, status=200, exists=True, endpoint="/api/v1/chats/{chat_id}"))
                        return True
                    if response.status == 404 or (response.status in {401, 403} and self._detail_indicates_missing(detail)):
                        if self.valves.debug:
                            logger.info(build_log_message("cleanup", "check", chat_id=chat_id, status=response.status, exists=False, detail=detail, endpoint="/api/v1/chats/{chat_id}"))
                        return False
                    if self.valves.debug:
                        logger.info(build_log_message("cleanup", "check", chat_id=chat_id, status=response.status, exists=None, detail=detail, endpoint="/api/v1/chats/{chat_id}"))
                    return None
        except Exception as exc:
            if self.valves.debug:
                logger.warning(build_log_message("cleanup", "error", chat_id=chat_id, error=type(exc).__name__))
            return None

    def _cleanup_ephemeral(self) -> None:
        for path in self.store.iter_chat_dirs():
            if path.name != "ephemeral":
                continue
            if self._state_age_sec(path.name) >= self.valves.cleanup_stale_ephemeral_after_sec:
                self.store.delete_chat_state(path.name)

    def _state_age_sec(self, chat_id: str) -> int:
        snapshot_path = self.store.snapshot_path(chat_id)
        target = snapshot_path if snapshot_path.exists() else self.store.chat_dir(chat_id)
        return max(0, now_ts() - int(target.stat().st_mtime))

    def _response_detail_text(self, response_text: str, status: int) -> str:
        if status == 200:
            return "ok"
        try:
            payload = json.loads(response_text)
            detail = payload.get("detail")
            if isinstance(detail, str) and detail.strip():
                return detail.strip().lower()
        except Exception:
            pass
        return str(status)

    def _detail_indicates_missing(self, detail: str) -> bool:
        lowered = str(detail or "").lower()
        return "could not find" in lowered or "not found" in lowered

    def _normalize_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        role = str(message.get("role", "user") or "user")
        content = message.get("content")
        if content in (None, "") and isinstance(message.get("output"), list):
            content = flatten_content(message.get("output"))
        normalized: Dict[str, Any] = {"role": role, "content": self._truncate_message(content)}
        for key in ["name", "tool_call_id", "tool_calls"]:
            if message.get(key) not in (None, "", []):
                normalized[key] = message.get(key)
        if isinstance(message.get("files"), list) and message["files"]:
            normalized["files"] = list(message["files"])
        normalized["_kind"] = "system" if role == "system" else "chat_message"
        return normalized

    def _truncate_message(self, content: Any) -> Any:
        if isinstance(content, str):
            if len(content) > self.valves.max_inline_chars * 2:
                return trim_text(content, self.valves.max_inline_chars, self.valves.file_excerpt_chars)
            return content
        if isinstance(content, list):
            return self._truncate_multimodal_content(content)
        return content

    def _sanitize_message(self, message: Dict[str, Any], passthrough_ids: Optional[set] = None) -> Dict[str, Any]:
        passthrough_ids = passthrough_ids or set()
        sanitized: Dict[str, Any] = {"role": message.get("role", "user"), "content": message.get("content", "")}
        for key in ["name", "tool_call_id", "tool_calls"]:
            if message.get(key) not in (None, "", []):
                sanitized[key] = message.get(key)
        if isinstance(message.get("files"), list):
            kept_files = self._filter_raw_files_by_ids(message.get("files"), passthrough_ids)
            if kept_files:
                sanitized["files"] = kept_files
        return sanitized

    def _apply_final_context(self, body: Dict[str, Any], effective: Dict[str, Any], final_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        body = dict(body)
        passthrough_ids = set(effective.get("passthrough_file_ids", []) or [])
        sanitized_messages = [self._sanitize_message(message, passthrough_ids) for message in final_messages]
        body["messages"] = sanitized_messages

        original_files = body.get("files", [])
        dropped = False
        if self.valves.takeover_file_context and self.valves.drop_original_files_on_takeover:
            body["files"] = self._filter_raw_files_by_ids(original_files, passthrough_ids)
            dropped = bool(original_files) and len(body["files"]) != len(original_files)
            if self.valves.strip_file_related_features_on_takeover:
                body["features"] = self._strip_file_related_features(body.get("features"))
        elif isinstance(original_files, list):
            body["files"] = original_files

        self._mark_body_as_file_context_taken_over(body, effective, dropped)
        if self.valves.debug:
            logger.info(build_log_message(
                "takeover", "apply",
                dropped=dropped,
                passthrough_file_count=effective.get("passthrough_file_count", 0),
                files_after=len(body.get("files", []) or []),
                metadata=body.get("metadata", {}),
                features=body.get("features", {}),
            ))
        return body

    def _mark_body_as_file_context_taken_over(self, body: Dict[str, Any], effective: Dict[str, Any], dropped: bool) -> None:
        metadata = dict(body.get("metadata", {}) or {})
        metadata.update(
            {
                "file_context_taken_over": bool(self.valves.takeover_file_context),
                "dropped_original_files": bool(dropped),
                "file_count": int(effective.get("request_file_count", 0) or 0),
                "compressed_file_count": len(effective.get("file_contexts", []) or []),
                "passthrough_file_count": int(effective.get("passthrough_file_count", 0) or 0),
                "selected_file_message_count": len(effective.get("file_virtual_messages", []) or []),
                "passthrough_token_cost": int(effective.get("passthrough_token_cost", 0) or 0),
                "message_image_count": int(effective.get("message_image_count", 0) or 0),
                "message_media_token_cost": int(effective.get("message_media_token_cost", 0) or 0),
            }
        )
        body["metadata"] = metadata

    def _strip_file_related_features(self, features: Any) -> Dict[str, Any]:
        cleaned = dict(features or {})
        for key in ["file_search", "rag", "knowledge", "file_context"]:
            cleaned.pop(key, None)
        return cleaned

    def _enforce_hard_limit(self, messages: List[Dict[str, Any]], extra_token_cost: int = 0) -> List[Dict[str, Any]]:
        limit = self._effective_context_window()
        total = self.token_counter.count_messages(messages) + extra_token_cost
        if total <= limit:
            return messages
        pinned = [message for message in messages if message.get("_kind") in {"system", "memory_snapshot", "file_context"}]
        rolling = [message for message in messages if message.get("_kind") not in {"system", "memory_snapshot", "file_context"}]
        kept = pinned + rolling[-self.valves.emergency_tail_keep_messages :]
        if self.valves.debug:
            logger.warning(build_log_message("context", "hard_limit", before=total, after=self.token_counter.count_messages(kept) + extra_token_cost, limit=limit))
        return self._dedupe_messages(kept)

    def _dedupe_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        last_digest = ""
        for message in messages:
            digest = self._message_digest(message)
            if digest and digest == last_digest:
                continue
            deduped.append(message)
            last_digest = digest
        return deduped

    def _message_digest(self, message: Dict[str, Any]) -> str:
        payload = {
            "role": message.get("role"),
            "content": message.get("content"),
            "name": message.get("name"),
            "tool_call_id": message.get("tool_call_id"),
            "tool_calls": message.get("tool_calls"),
            "files": self._raw_file_ids(message.get("files")),
            "_kind": message.get("_kind"),
        }
        try:
            return sha1_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        except Exception:
            return sha1_text(str(payload))

    def _effective_context_window(self) -> int:
        return max(1024, self.valves.max_context_tokens - self.valves.reserved_output_tokens)

    def _compact_threshold(self) -> int:
        return max(512, self._effective_context_window() - self.valves.autocompact_buffer_tokens)

    def _resolve_chat_id(self, body: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> str:
        for container in [metadata or {}, body.get("metadata", {}) or {}, body]:
            if not isinstance(container, dict):
                continue
            for key in ["chat_id", "chatId", "conversation_id", "conversationId", "id"]:
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return "ephemeral"

    async def _emit_status(self, emitter: Any, description: str, done: bool) -> None:
        await self._emit(
            emitter,
            {
                "type": "status",
                "data": {
                    "description": description,
                    "done": done,
                },
            },
        )

    async def _emit(self, emitter: Any, payload: Dict[str, Any]) -> None:
        if emitter is None:
            return
        try:
            result = emitter(payload)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception(build_log_message("event", "emit_failed", payload=payload))

    def _last_transcript(self, snapshot: Dict[str, Any]) -> str:
        transcripts = snapshot.get("transcript_paths", []) if isinstance(snapshot, dict) else []
        if isinstance(transcripts, list) and transcripts:
            return str(transcripts[-1])
        return ""

    def _find_last_message(self, messages: Any, role: str) -> Optional[Dict[str, Any]]:
        if not isinstance(messages, list):
            return None
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == role:
                return message
        return None

    def _collect_request_files(self, body: Dict[str, Any]) -> Dict[str, Any]:
        raw_files: List[Dict[str, Any]] = []
        message_file_count = 0
        for raw in body.get("files", []) if isinstance(body.get("files"), list) else []:
            if isinstance(raw, dict):
                clone = dict(raw)
                clone["_origin"] = "body"
                raw_files.append(clone)
        for index, message in enumerate(body.get("messages", []) if isinstance(body.get("messages"), list) else []):
            if not isinstance(message, dict):
                continue
            files = message.get("files")
            if not isinstance(files, list):
                continue
            for raw in files:
                if not isinstance(raw, dict):
                    continue
                clone = dict(raw)
                clone["_origin"] = "message"
                clone["_message_index"] = index
                raw_files.append(clone)
                message_file_count += 1

        deduped: List[Dict[str, Any]] = []
        seen: set = set()
        for raw in raw_files:
            identity = self._raw_file_identity(raw)
            if not identity or identity in seen:
                continue
            seen.add(identity)
            deduped.append(raw)
        return {
            "raw_files": deduped,
            "request_file_count": len(deduped),
            "message_file_count": message_file_count,
        }

    def _raw_file_identity(self, raw: Any) -> str:
        if not isinstance(raw, dict):
            return ""
        file_obj = raw.get("file") if isinstance(raw.get("file"), dict) else {}
        for key in ["id", "url", "itemId", "path", "name", "filename"]:
            value = raw.get(key)
            if value not in (None, ""):
                return str(value)
            value = file_obj.get(key)
            if value not in (None, ""):
                return str(value)
        return ""

    def _raw_file_ids(self, raw_files: Any) -> List[str]:
        if not isinstance(raw_files, list):
            return []
        ids: List[str] = []
        for raw in raw_files:
            file_id = self._extract_file_meta(raw).get("file_id", "")
            if file_id:
                ids.append(str(file_id))
        return ids

    def _filter_raw_files_by_ids(self, raw_files: Any, keep_ids: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_files, list):
            return []
        keep = set(keep_ids or [])
        return [raw for raw in raw_files if self._extract_file_meta(raw).get("file_id") in keep]

    def _message_image_count(self, messages: List[Dict[str, Any]]) -> int:
        return sum(self._content_image_count(message.get("content")) for message in messages)

    def _content_image_count(self, content: Any) -> int:
        if not isinstance(content, list):
            return 0
        count = 0
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "") or "")
            if item_type in {"image_url", "input_image"}:
                count += 1
        return count

    def _estimate_message_media_tokens(self, messages: List[Dict[str, Any]]) -> int:
        return self._message_image_count(messages) * self.valves.image_passthrough_token_cost

    def _content_has_non_text_parts(self, content: Any) -> bool:
        if not isinstance(content, list):
            return False
        for item in content:
            if not isinstance(item, dict):
                continue
            if str(item.get("type", "") or "") not in {"text", "input_text", "output_text"}:
                return True
        return False

    def _truncate_multimodal_content(self, content: List[Any]) -> List[Any]:
        normalized: List[Any] = []
        for item in content:
            if not isinstance(item, dict):
                normalized.append(item)
                continue
            item_type = str(item.get("type", "") or "")
            if item_type in {"text", "input_text", "output_text"}:
                cloned = dict(item)
                text = str(cloned.get("text", "") or "")
                if len(text) > self.valves.max_inline_chars:
                    cloned["text"] = trim_text(text, self.valves.max_inline_chars, self.valves.file_excerpt_chars)
                normalized.append(cloned)
            else:
                normalized.append(dict(item))
        return normalized
