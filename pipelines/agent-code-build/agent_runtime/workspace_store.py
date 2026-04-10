"""Persistent workspace storage for session state, attachments, and transcripts.

Handles safe file registration, prompt-history persistence, transcript/artifact
I/O, and text-oriented file access helpers used by runtime tools.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .common import build_log_message, ensure_dir, flatten_content, get_logger, now_ts, render_message, sha1_text
from .message_types import SessionState, default_session_state


SUPPORTED_TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
    ".scss",
    ".sql",
    ".toml",
    ".ini",
    ".cfg",
    ".env",
    ".sh",
    ".ps1",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".xml",
    ".log",
}


class WorkspaceStore:
    """Filesystem-backed session store used by the runtime."""

    def __init__(
        self,
        root_dir: str,
        max_file_chars: int = 200000,
        max_transcript_chars: int = 120000,
        openwebui_uploads_root: str = "",
        mounted_uploads_root: str = "",
        enforce_upload_root: bool = False,
        file_registration_mode: str = "copy",
    ) -> None:
        """Initialize storage roots, limits, and upload path security settings."""
        self.root = Path(root_dir)
        self.max_file_chars = max_file_chars
        self.max_transcript_chars = max_transcript_chars
        self.openwebui_uploads_root = str(openwebui_uploads_root or "").strip()
        self.mounted_uploads_root = str(mounted_uploads_root or "").strip()
        self.enforce_upload_root = enforce_upload_root
        self.file_registration_mode = str(file_registration_mode or "copy").strip().lower()
        self.logger = get_logger("workspace")
        ensure_dir(self.root)
        self.logger.info(
            build_log_message(
                "workspace",
                "init",
                root_dir=self.root,
                max_file_chars=self.max_file_chars,
                max_transcript_chars=self.max_transcript_chars,
                openwebui_uploads_root=self.openwebui_uploads_root,
                mounted_uploads_root=self.mounted_uploads_root,
                enforce_upload_root=self.enforce_upload_root,
                file_registration_mode=self.file_registration_mode,
            )
        )

    def _safe_chat_id(self, chat_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(chat_id or "ephemeral"))
        return safe or "ephemeral"

    def session_dir(self, chat_id: str) -> Path:
        path = self.root / "sessions" / self._safe_chat_id(chat_id)
        ensure_dir(path)
        return path

    def attachment_dir(self, chat_id: str) -> Path:
        path = self.session_dir(chat_id) / "attachments"
        ensure_dir(path)
        return path

    def transcript_dir(self, chat_id: str) -> Path:
        path = self.session_dir(chat_id) / "transcripts"
        ensure_dir(path)
        return path

    def artifact_dir(self, chat_id: str) -> Path:
        path = self.session_dir(chat_id) / "artifacts"
        ensure_dir(path)
        return path

    def scratch_dir(self, chat_id: str) -> Path:
        path = self.session_dir(chat_id) / "scratch"
        ensure_dir(path)
        return path

    def state_path(self, chat_id: str) -> Path:
        return self.session_dir(chat_id) / "state.json"

    def manifest_path(self, chat_id: str) -> Path:
        return self.session_dir(chat_id) / "attachments.json"

    def snapshot_path(self, chat_id: str) -> Path:
        return self.session_dir(chat_id) / "snapshot.json"

    def prompt_history_path(self, chat_id: str) -> Path:
        return self.session_dir(chat_id) / "prompt_history.json"

    def default_snapshot(self) -> Dict[str, Any]:
        return {
            "summary": "",
            "last_effective_tokens": 0,
            "last_compaction_reason": "",
            "last_compacted_group_anchor": "",
            "last_compacted_group_count": 0,
            "consecutive_failures": 0,
            "updated_at": 0,
            "transcript_paths": [],
        }

    def default_prompt_history(self) -> Dict[str, Any]:
        return {
            "entries": [],
            "updated_at": 0,
        }

    def load_state(self, chat_id: str) -> SessionState:
        """Load a session state from disk and merge defaults."""
        path = self.state_path(chat_id)
        state = default_session_state(chat_id, now_ts())
        if path.exists():
            try:
                data = json.loads(path.read_text("utf-8"))
                if isinstance(data, dict):
                    state.update(data)
            except Exception:
                pass
        state["attachment_manifest"] = self.load_attachment_manifest(chat_id, state.get("attachment_manifest"))
        snapshot = self.load_snapshot(chat_id)
        state["compaction_snapshot"] = state.get("compaction_snapshot") or snapshot
        state["transcript_paths"] = list(snapshot.get("transcript_paths", []) or [])
        state["summary_message"] = self._latest_summary_message(state.get("history", []))
        self.logger.debug(
            build_log_message(
                "workspace",
                "load_state",
                chat_id=chat_id,
                history_count=len(state.get("history", [])),
                transcript_count=len(state.get("transcript_paths", []) or []),
            )
        )
        return state

    def save_state(self, chat_id: str, state: SessionState) -> None:
        """Persist main session state and attachment manifest atomically."""
        payload = dict(default_session_state(chat_id, now_ts()))
        payload.update(state)
        self._write_json(self.state_path(chat_id), payload)
        self.save_attachment_manifest(chat_id, payload.get("attachment_manifest", {"files": []}))
        self.logger.debug(
            build_log_message(
                "workspace",
                "save_state",
                chat_id=chat_id,
                history_count=len(payload.get("history", [])),
            )
        )

    def load_attachment_manifest(
        self, chat_id: str, fallback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        path = self.manifest_path(chat_id)
        manifest = {"files": []}
        if isinstance(fallback, dict):
            manifest.update(fallback)
        if path.exists():
            try:
                data = json.loads(path.read_text("utf-8"))
                if isinstance(data, dict):
                    manifest.update(data)
            except Exception:
                pass
        manifest["files"] = list(manifest.get("files", []) or [])
        return manifest

    def save_attachment_manifest(self, chat_id: str, manifest: Dict[str, Any]) -> None:
        self._write_json(self.manifest_path(chat_id), manifest)

    def load_snapshot(self, chat_id: str) -> Dict[str, Any]:
        path = self.snapshot_path(chat_id)
        snapshot = self.default_snapshot()
        if path.exists():
            try:
                data = json.loads(path.read_text("utf-8"))
                if isinstance(data, dict):
                    snapshot.update(data)
            except Exception:
                pass
        return snapshot

    def save_snapshot(self, chat_id: str, snapshot: Dict[str, Any]) -> None:
        snapshot["updated_at"] = int(snapshot.get("updated_at") or now_ts())
        self._write_json(self.snapshot_path(chat_id), snapshot)
        self.logger.info(
            build_log_message(
                "workspace",
                "save_snapshot",
                chat_id=chat_id,
                summary_chars=len(str(snapshot.get("summary", "") or "")),
                transcript_count=len(snapshot.get("transcript_paths", []) or []),
                last_effective_tokens=snapshot.get("last_effective_tokens"),
                last_compaction_reason=snapshot.get("last_compaction_reason"),
                updated_at=snapshot.get("updated_at"),
            )
        )

    def load_prompt_history(self, chat_id: str) -> Dict[str, Any]:
        path = self.prompt_history_path(chat_id)
        payload = self.default_prompt_history()
        if path.exists():
            try:
                data = json.loads(path.read_text("utf-8"))
                if isinstance(data, dict):
                    payload.update(data)
            except Exception:
                pass
        payload["entries"] = list(payload.get("entries", []) or [])
        return payload

    def save_prompt_history(self, chat_id: str, payload: Dict[str, Any]) -> None:
        clone = self.default_prompt_history()
        clone.update(payload or {})
        clone["updated_at"] = int(clone.get("updated_at") or now_ts())
        clone["entries"] = list(clone.get("entries", []) or [])
        self._write_json(self.prompt_history_path(chat_id), clone)
        self.logger.debug(
            build_log_message(
                "workspace",
                "save_prompt_history",
                chat_id=chat_id,
                entry_count=len(clone["entries"]),
                updated_at=clone["updated_at"],
            )
        )

    def sync_prompt_history(self, chat_id: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Append newly observed user prompts into persisted prompt history."""
        payload = self.load_prompt_history(chat_id)
        entries = list(payload.get("entries", []) or [])
        extracted: List[str] = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "user") or "user") != "user":
                continue
            text = flatten_content(message.get("content")).strip()
            if text:
                extracted.append(text)

        existing_count = len(entries)
        if len(extracted) > existing_count:
            for index, text in enumerate(extracted[existing_count:], start=existing_count + 1):
                entries.append(
                    {
                        "index": index,
                        "ts": now_ts(),
                        "text": text,
                        "chars": len(text),
                        "digest": sha1_text(f"{index}:{text}"),
                    }
                )
        payload["entries"] = entries
        payload["updated_at"] = now_ts()
        self.save_prompt_history(chat_id, payload)
        self.logger.info(
            build_log_message(
                "workspace",
                "sync_prompt_history",
                chat_id=chat_id,
                extracted_count=len(extracted),
                stored_count=len(entries),
            )
        )
        return payload

    def render_prompt_history_text(self, chat_id: str, payload: Optional[Dict[str, Any]] = None) -> str:
        prompt_history = payload or self.load_prompt_history(chat_id)
        rendered: List[str] = ["# Prompt History"]
        for entry in list(prompt_history.get("entries", []) or []):
            if not isinstance(entry, dict):
                continue
            index = int(entry.get("index") or 0)
            text = str(entry.get("text") or "").strip()
            if not text:
                continue
            rendered.append(f"[user_prompt #{index}]")
            rendered.append(text)
            rendered.append("")
        return "\n".join(rendered).strip()

    def write_transcript(self, chat_id: str, messages: List[Dict[str, Any]]) -> str:
        """Write markdown transcript for compaction recovery."""
        path = self.transcript_dir(chat_id) / f"{now_ts()}_{uuid4().hex[:8]}.md"
        rendered: List[str] = []
        for message in messages:
            rendered.extend([render_message(message), "", "-" * 60, ""])
        payload = "\n".join(rendered).strip()
        path.write_text(payload, encoding="utf-8")
        self.logger.info(
            build_log_message(
                "workspace",
                "write_transcript",
                chat_id=chat_id,
                transcript_path=path,
                message_count=len(messages),
                transcript_chars=len(payload),
            )
        )
        return str(path)

    def write_text_artifact(self, chat_id: str, prefix: str, text: str) -> str:
        safe_prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", str(prefix or "artifact"))[:48] or "artifact"
        path = self.artifact_dir(chat_id) / f"{safe_prefix}_{now_ts()}_{uuid4().hex[:8]}.md"
        path.write_text(str(text or ""), encoding="utf-8")
        self.logger.info(
            build_log_message(
                "workspace",
                "write_text_artifact",
                chat_id=chat_id,
                artifact_path=path,
                text_chars=len(str(text or "")),
            )
        )
        return str(path)

    def collect_request_files(self, body: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect and de-duplicate file descriptors from request body/messages."""
        raw_files: List[Dict[str, Any]] = []
        for raw in body.get("files", []) if isinstance(body.get("files"), list) else []:
            if isinstance(raw, dict):
                clone = dict(raw)
                clone["_origin"] = "body"
                raw_files.append(clone)
        for index, message in enumerate(body.get("messages", []) if isinstance(body.get("messages"), list) else []):
            if not isinstance(message, dict):
                continue
            for raw in message.get("files", []) if isinstance(message.get("files"), list) else []:
                if isinstance(raw, dict):
                    clone = dict(raw)
                    clone["_origin"] = "message"
                    clone["_message_index"] = index
                    raw_files.append(clone)

        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for raw in raw_files:
            identity = self._raw_file_identity(raw)
            if not identity or identity in seen:
                continue
            seen.add(identity)
            deduped.append(raw)
        self.logger.info(build_log_message("workspace", "collect_request_files", raw_count=len(raw_files), deduped_count=len(deduped)))
        return deduped

    def register_uploads(
        self,
        chat_id: str,
        raw_files: List[Dict[str, Any]],
        existing_manifest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolve, validate, and register request files into manifest."""
        existing_items = list((existing_manifest or {}).get("files", []) or [])
        existing_by_key = {
            self._manifest_identity(item): item
            for item in existing_items
            if self._manifest_identity(item)
        }

        for raw in raw_files:
            meta = self.extract_file_meta(raw)
            source_path = self.resolve_safe_file_path(meta)
            if not source_path:
                continue
            source = Path(source_path)
            if not source.exists() or not source.is_file():
                continue

            fingerprint = self.compute_file_fingerprint(meta, source)
            filename = meta.get("filename") or source.name or meta.get("file_id") or "attachment"
            target_path = source
            storage_mode = "reference"
            if self.file_registration_mode != "reference":
                target_name = self._build_attachment_filename(filename, fingerprint)
                target_path = self.attachment_dir(chat_id) / target_name
                if not target_path.exists():
                    shutil.copy2(str(source), str(target_path))
                storage_mode = "copy"

            item = {
                "file_id": meta.get("file_id") or "",
                "display_name": filename,
                "workspace_path": str(target_path),
                "source_path": str(source),
                "origin_path": str(meta.get("path") or ""),
                "content_type": meta.get("content_type") or "",
                "fingerprint": fingerprint,
                "size_bytes": int(source.stat().st_size),
                "storage_mode": storage_mode,
            }
            identity = self._manifest_identity(item)
            if identity:
                existing_by_key[identity] = item

        manifest = {"files": list(existing_by_key.values())}
        self.save_attachment_manifest(chat_id, manifest)
        self.logger.info(
            build_log_message(
                "workspace",
                "register_uploads",
                chat_id=chat_id,
                registered_count=len(manifest["files"]),
                file_registration_mode=self.file_registration_mode,
            )
        )
        return manifest

    def resolve_attachment_path(
        self, chat_id: str, reference: str, manifest: Optional[Dict[str, Any]] = None
    ) -> Optional[Path]:
        """Resolve user/tool references to a safe, existing attachment path."""
        if not reference:
            return None
        ref = str(reference).strip()
        manifest = manifest or self.load_attachment_manifest(chat_id)
        candidate = Path(ref)
        if candidate.is_absolute() and candidate.exists():
            candidate_resolved = candidate.resolve()
            try:
                candidate_resolved.relative_to(self.attachment_dir(chat_id).resolve())
                self.logger.debug(build_log_message("workspace", "resolve_attachment_absolute", chat_id=chat_id, reference=ref))
                return candidate_resolved
            except Exception:
                pass
            for item in manifest.get("files", []) or []:
                workspace_path = str(item.get("workspace_path", "") or "")
                source_path = str(item.get("source_path", "") or "")
                if ref in {workspace_path, source_path}:
                    self.logger.debug(
                        build_log_message(
                            "workspace",
                            "resolve_attachment_absolute_manifest",
                            chat_id=chat_id,
                            reference=ref,
                            path=candidate_resolved,
                        )
                    )
                    return candidate_resolved
            self.logger.warning(build_log_message("workspace", "resolve_attachment_denied", chat_id=chat_id, reference=ref))
            return None

        for item in manifest.get("files", []) or []:
            workspace_path = str(item.get("workspace_path", "") or "")
            source_path = str(item.get("source_path", "") or "")
            display_name = str(item.get("display_name", "") or "")
            file_id = str(item.get("file_id", "") or "")
            if ref in {workspace_path, source_path, display_name, Path(workspace_path).name, Path(source_path).name, file_id}:
                path = Path(workspace_path)
                if path.exists():
                    self.logger.debug(build_log_message("workspace", "resolve_attachment_manifest", chat_id=chat_id, reference=ref, path=path))
                    return path
        self.logger.warning(build_log_message("workspace", "resolve_attachment_missing", chat_id=chat_id, reference=ref))
        return None

    def resolve_transcript_path(self, chat_id: str, reference: str = "") -> Optional[Path]:
        transcripts = sorted(self.transcript_dir(chat_id).glob("*.md"))
        artifacts = sorted(self.artifact_dir(chat_id).glob("*.md"))
        if not transcripts and not artifacts:
            self.logger.debug(build_log_message("workspace", "resolve_transcript_none", chat_id=chat_id))
            return None
        if not reference:
            if not transcripts:
                self.logger.debug(build_log_message("workspace", "resolve_transcript_latest_missing_compaction", chat_id=chat_id))
                return None
            self.logger.debug(build_log_message("workspace", "resolve_transcript_latest", chat_id=chat_id, path=transcripts[-1]))
            return transcripts[-1]

        ref = str(reference).strip()
        candidate = Path(ref)
        if candidate.is_absolute() and candidate.exists():
            for allowed_root in [self.transcript_dir(chat_id), self.artifact_dir(chat_id)]:
                try:
                    candidate.resolve().relative_to(allowed_root.resolve())
                    self.logger.debug(
                        build_log_message("workspace", "resolve_transcript_absolute", chat_id=chat_id, reference=ref)
                    )
                    return candidate.resolve()
                except Exception:
                    continue
            self.logger.warning(build_log_message("workspace", "resolve_transcript_denied", chat_id=chat_id, reference=ref))
            return None

        for path in [*transcripts, *artifacts]:
            if ref in {str(path), path.name}:
                self.logger.debug(build_log_message("workspace", "resolve_transcript_named", chat_id=chat_id, reference=ref, path=path))
                return path
        self.logger.warning(build_log_message("workspace", "resolve_transcript_missing", chat_id=chat_id, reference=ref))
        return None

    def read_text_file(self, path: Path, max_chars: Optional[int] = None) -> Tuple[str, bool]:
        """Read a text-compatible file and report whether output was truncated."""
        limit = max_chars or self.max_file_chars
        self.logger.debug(build_log_message("workspace", "read_text_file", path=path, limit=limit))
        if path.suffix.lower() == ".docx":
            text = self._load_docx_text(path)
            return text[:limit], len(text) > limit

        if path.suffix.lower() not in SUPPORTED_TEXT_EXTENSIONS:
            self.logger.warning(build_log_message("workspace", "unsupported_extension", path=path, suffix=path.suffix.lower()))
            return "", False

        for encoding in ["utf-8", "utf-8-sig", "gb18030", "latin-1"]:
            try:
                text = path.read_text(encoding=encoding)
                self.logger.debug(build_log_message("workspace", "read_text_file_success", path=path, encoding=encoding, truncated=len(text) > limit))
                return text[:limit], len(text) > limit
            except Exception:
                continue

        try:
            data = path.read_bytes()[: limit * 4]
            text = data.decode("utf-8", errors="ignore")
            self.logger.debug(build_log_message("workspace", "read_bytes_fallback", path=path, truncated=len(text) > limit))
            return text[:limit], len(text) > limit
        except Exception:
            self.logger.exception(build_log_message("workspace", "read_text_file_failed", path=path))
            return "", False

    def read_file_line_range(
        self,
        path: Path,
        start_line: int,
        end_line: int,
    ) -> Dict[str, Any]:
        """Read inclusive line range from supported text-like files."""
        start = max(1, int(start_line))
        end = max(start, int(end_line))
        self.logger.debug(
            build_log_message(
                "workspace",
                "read_file_line_range",
                path=path,
                start_line=start,
                end_line=end,
            )
        )
        suffix = path.suffix.lower()
        if suffix == ".docx":
            lines = self._load_docx_text(path).splitlines()
            return self._build_line_range_result(lines, start, end, encoding="docx")
        if suffix not in SUPPORTED_TEXT_EXTENSIONS:
            self.logger.warning(build_log_message("workspace", "line_range_unsupported_extension", path=path, suffix=suffix))
            return {
                "supported": False,
                "encoding": "",
                "lines": [],
                "total_lines": 0,
                "actual_end": 0,
                "requested_end": end,
            }

        for encoding in ["utf-8", "utf-8-sig", "gb18030", "latin-1"]:
            try:
                with path.open("r", encoding=encoding) as handle:
                    excerpt: List[str] = []
                    total_lines = 0
                    for total_lines, raw_line in enumerate(handle, start=1):
                        if total_lines < start:
                            continue
                        if total_lines <= end:
                            excerpt.append(raw_line.rstrip("\r\n"))
                    return {
                        "supported": True,
                        "encoding": encoding,
                        "lines": excerpt,
                        "total_lines": total_lines,
                        "actual_end": min(end, total_lines) if total_lines else 0,
                        "requested_end": end,
                    }
            except Exception:
                continue

        try:
            data = path.read_bytes()
            text = data.decode("utf-8", errors="ignore")
            return self._build_line_range_result(text.splitlines(), start, end, encoding="utf-8-ignore")
        except Exception:
            self.logger.exception(build_log_message("workspace", "read_file_line_range_failed", path=path))
            return {
                "supported": False,
                "encoding": "",
                "lines": [],
                "total_lines": 0,
                "actual_end": 0,
                "requested_end": end,
            }

    def read_file_chunk(
        self,
        path: Path,
        start_line: int,
        max_chars: int,
    ) -> Dict[str, Any]:
        """Read consecutive lines from start line until char budget is reached."""
        start = max(1, int(start_line))
        char_limit = max(200, int(max_chars))
        self.logger.debug(
            build_log_message(
                "workspace",
                "read_file_chunk",
                path=path,
                start_line=start,
                max_chars=char_limit,
            )
        )
        suffix = path.suffix.lower()
        if suffix == ".docx":
            lines = self._load_docx_text(path).splitlines()
            return self._build_chunk_result(lines, start, char_limit, encoding="docx")
        if suffix not in SUPPORTED_TEXT_EXTENSIONS:
            self.logger.warning(build_log_message("workspace", "chunk_unsupported_extension", path=path, suffix=suffix))
            return {
                "supported": False,
                "encoding": "",
                "lines": [],
                "total_lines": 0,
                "actual_end": 0,
                "used_chars": 0,
                "requested_chars": char_limit,
            }

        for encoding in ["utf-8", "utf-8-sig", "gb18030", "latin-1"]:
            try:
                with path.open("r", encoding=encoding) as handle:
                    excerpt: List[str] = []
                    total_lines = 0
                    used_chars = 0
                    for total_lines, raw_line in enumerate(handle, start=1):
                        if total_lines < start:
                            continue
                        line = raw_line.rstrip("\r\n")
                        line_cost = max(1, len(line) + 1)
                        if excerpt and used_chars + line_cost > char_limit:
                            break
                        excerpt.append(line)
                        used_chars += line_cost
                    actual_end = start + len(excerpt) - 1 if excerpt else 0
                    return {
                        "supported": True,
                        "encoding": encoding,
                        "lines": excerpt,
                        "total_lines": total_lines,
                        "actual_end": actual_end,
                        "used_chars": used_chars,
                        "requested_chars": char_limit,
                    }
            except Exception:
                continue

        try:
            data = path.read_bytes()
            text = data.decode("utf-8", errors="ignore")
            return self._build_chunk_result(text.splitlines(), start, char_limit, encoding="utf-8-ignore")
        except Exception:
            self.logger.exception(build_log_message("workspace", "read_file_chunk_failed", path=path))
            return {
                "supported": False,
                "encoding": "",
                "lines": [],
                "total_lines": 0,
                "actual_end": 0,
                "used_chars": 0,
                "requested_chars": char_limit,
            }

    def search_text_file(
        self,
        path: Path,
        query: str,
        case_sensitive: bool = False,
        max_results: int = 20,
    ) -> Dict[str, Any]:
        """Search substring hits with bounded result count."""
        needle = str(query or "")
        max_hits = max(1, min(int(max_results or 20), 50))
        self.logger.debug(
            build_log_message(
                "workspace",
                "search_text_file",
                path=path,
                query=needle,
                case_sensitive=case_sensitive,
                max_results=max_hits,
            )
        )
        suffix = path.suffix.lower()
        if suffix == ".docx":
            return self._search_lines(
                lines=self._load_docx_text(path).splitlines(),
                query=needle,
                case_sensitive=case_sensitive,
                max_results=max_hits,
                encoding="docx",
            )
        if suffix not in SUPPORTED_TEXT_EXTENSIONS:
            self.logger.warning(build_log_message("workspace", "search_unsupported_extension", path=path, suffix=suffix))
            return {
                "supported": False,
                "encoding": "",
                "results": [],
                "hit_limit_reached": False,
            }

        for encoding in ["utf-8", "utf-8-sig", "gb18030", "latin-1"]:
            try:
                with path.open("r", encoding=encoding) as handle:
                    results: List[str] = []
                    lowered_needle = needle if case_sensitive else needle.lower()
                    for line_number, raw_line in enumerate(handle, start=1):
                        line = raw_line.rstrip("\r\n")
                        haystack = line if case_sensitive else line.lower()
                        if lowered_needle in haystack:
                            results.append(f"{line_number}: {line.strip()}")
                            if len(results) >= max_hits:
                                return {
                                    "supported": True,
                                    "encoding": encoding,
                                    "results": results,
                                    "hit_limit_reached": True,
                                }
                    return {
                        "supported": True,
                        "encoding": encoding,
                        "results": results,
                        "hit_limit_reached": False,
                    }
            except Exception:
                continue

        try:
            data = path.read_bytes()
            text = data.decode("utf-8", errors="ignore")
            return self._search_lines(
                lines=text.splitlines(),
                query=needle,
                case_sensitive=case_sensitive,
                max_results=max_hits,
                encoding="utf-8-ignore",
            )
        except Exception:
            self.logger.exception(build_log_message("workspace", "search_text_file_failed", path=path))
            return {
                "supported": False,
                "encoding": "",
                "results": [],
                "hit_limit_reached": False,
            }

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
        return text

    def _build_line_range_result(
        self,
        lines: List[str],
        start: int,
        end: int,
        encoding: str,
    ) -> Dict[str, Any]:
        total_lines = len(lines)
        actual_end = min(end, total_lines) if total_lines else 0
        excerpt = lines[start - 1:actual_end] if start <= total_lines else []
        return {
            "supported": True,
            "encoding": encoding,
            "lines": excerpt,
            "total_lines": total_lines,
            "actual_end": actual_end,
            "requested_end": end,
        }

    def _build_chunk_result(
        self,
        lines: List[str],
        start: int,
        char_limit: int,
        encoding: str,
    ) -> Dict[str, Any]:
        total_lines = len(lines)
        if start > total_lines:
            return {
                "supported": True,
                "encoding": encoding,
                "lines": [],
                "total_lines": total_lines,
                "actual_end": 0,
                "used_chars": 0,
                "requested_chars": char_limit,
            }
        excerpt: List[str] = []
        used_chars = 0
        actual_end = 0
        for line_number in range(start, total_lines + 1):
            line = lines[line_number - 1]
            line_cost = max(1, len(line) + 1)
            if excerpt and used_chars + line_cost > char_limit:
                break
            excerpt.append(line)
            used_chars += line_cost
            actual_end = line_number
        return {
            "supported": True,
            "encoding": encoding,
            "lines": excerpt,
            "total_lines": total_lines,
            "actual_end": actual_end,
            "used_chars": used_chars,
            "requested_chars": char_limit,
        }

    def _search_lines(
        self,
        lines: List[str],
        query: str,
        case_sensitive: bool,
        max_results: int,
        encoding: str,
    ) -> Dict[str, Any]:
        lowered_needle = query if case_sensitive else query.lower()
        results: List[str] = []
        for line_number, line in enumerate(lines, start=1):
            haystack = line if case_sensitive else line.lower()
            if lowered_needle in haystack:
                results.append(f"{line_number}: {line.strip()}")
                if len(results) >= max_results:
                    return {
                        "supported": True,
                        "encoding": encoding,
                        "results": results,
                        "hit_limit_reached": True,
                    }
        return {
            "supported": True,
            "encoding": encoding,
            "results": results,
            "hit_limit_reached": False,
        }

    def extract_file_meta(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        file_obj = raw.get("file") if isinstance(raw, dict) else {}
        file_obj = file_obj if isinstance(file_obj, dict) else {}
        meta = file_obj.get("meta") if isinstance(file_obj.get("meta"), dict) else {}
        return {
            "file_id": self._pick_first_value(raw, file_obj, keys=["id", "url", "itemId"]),
            "filename": self._pick_first_value(raw, file_obj, meta, keys=["filename", "name"]),
            "content_type": self._pick_first_value(raw, file_obj, meta, keys=["content_type"]),
            "path": self._pick_first_value(raw, file_obj, keys=["path"]),
        }

    def resolve_safe_file_path(self, meta: Dict[str, Any]) -> str:
        """Resolve uploaded path with optional remapping and root enforcement."""
        path_str = str(meta.get("path", "") or "").strip()
        if not path_str:
            self.logger.debug(build_log_message("workspace", "resolve_safe_file_empty_path"))
            return ""

        candidates = [path_str]
        remapped = self._remap_openwebui_upload_path(path_str)
        if remapped and remapped not in candidates:
            candidates.append(remapped)
        self.logger.debug(build_log_message("workspace", "resolve_safe_file_candidates", candidates=candidates))

        for candidate_str in candidates:
            path = Path(candidate_str)
            if not path.exists():
                continue
            try:
                resolved = path.resolve()
            except Exception:
                continue
            if self.enforce_upload_root and not self._is_allowed_upload_path(resolved):
                self.logger.warning(build_log_message("workspace", "resolve_safe_file_outside_root", path=resolved))
                continue
            self.logger.info(build_log_message("workspace", "resolve_safe_file_success", path=resolved))
            return str(resolved)
        self.logger.warning(build_log_message("workspace", "resolve_safe_file_missing", original_path=path_str, remapped=remapped))
        return ""

    def compute_file_fingerprint(self, meta: Dict[str, Any], path: Path) -> str:
        stat = path.stat()
        base = f"{meta.get('file_id', '')}|{path.name}|{stat.st_size}|{int(stat.st_mtime)}"
        return sha1_text(base)[:12]

    def _raw_file_identity(self, raw: Dict[str, Any]) -> str:
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

    def _pick_first_value(self, *containers: Any, keys: List[str]) -> str:
        for container in containers:
            if not isinstance(container, dict):
                continue
            for key in keys:
                value = container.get(key)
                if value not in (None, ""):
                    return str(value)
        return ""

    def _build_attachment_filename(self, filename: str, fingerprint: str) -> str:
        original = Path(filename)
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", original.stem)[:80] or "attachment"
        suffix = re.sub(r"[^A-Za-z0-9.]+", "", original.suffix)[:16]
        return f"{stem}_{fingerprint}{suffix}"

    def _manifest_identity(self, item: Dict[str, Any]) -> str:
        for key in ["file_id", "workspace_path", "source_path"]:
            value = str(item.get(key, "") or "")
            if value:
                return value
        return ""

    def _remap_openwebui_upload_path(self, path_str: str) -> str:
        if not path_str:
            return ""
        if not self.openwebui_uploads_root or not self.mounted_uploads_root:
            return ""
        try:
            source_root = Path(self.openwebui_uploads_root).resolve()
            source_path = Path(path_str)
            relative = source_path.resolve().relative_to(source_root)
        except Exception:
            try:
                normalized_root = self._normalize_for_prefix(self.openwebui_uploads_root)
                normalized_path = self._normalize_for_prefix(path_str)
                if not normalized_path.startswith(normalized_root + "/"):
                    return ""
                relative = Path(normalized_path[len(normalized_root) + 1 :])
            except Exception:
                return ""
        remapped = str((Path(self.mounted_uploads_root) / relative).resolve())
        self.logger.debug(build_log_message("workspace", "remap_upload_path", source_path=path_str, remapped_path=remapped))
        return remapped

    def _is_allowed_upload_path(self, path: Path) -> bool:
        allowed_roots: List[Path] = []
        for root in [self.openwebui_uploads_root, self.mounted_uploads_root]:
            if not root:
                continue
            try:
                allowed_roots.append(Path(root).resolve())
            except Exception:
                continue
        if not allowed_roots:
            return True
        for root in allowed_roots:
            try:
                path.relative_to(root)
                return True
            except Exception:
                continue
        return False

    def _normalize_for_prefix(self, path_str: str) -> str:
        return str(path_str or "").replace("\\", "/").rstrip("/")

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        ensure_dir(path.parent)
        tmp = path.with_suffix(path.suffix + f".{uuid4().hex}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(str(tmp), str(path))
        self.logger.debug(build_log_message("workspace", "write_json", path=path))

    def _latest_summary_message(
        self, history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        for message in reversed(history or []):
            if isinstance(message, dict) and message.get("_kind") == "memory_summary":
                return message
        return None
