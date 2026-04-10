"""Provider adapter for OpenAI-compatible chat/completions APIs.

Wraps sync/stream calls and normalizes response parsing.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional

import requests

from .common import build_log_message, flatten_content, get_logger


class ProviderAdapter:
    """Model API adapter used by runtime."""

    def __init__(self, base_url: str, api_key: str, timeout_sec: int = 120) -> None:
        """Initialize endpoint settings and request timeout."""
        self.base_url = str(base_url or "").rstrip("/")
        self.api_key = str(api_key or "").strip()
        self.timeout_sec = timeout_sec
        self.logger = get_logger("provider")
        self.logger.info(
            build_log_message(
                "provider",
                "init",
                base_url=self.base_url,
                timeout_sec=self.timeout_sec,
                has_api_key=bool(self.api_key),
            )
        )

    def close(self) -> None:
        """No-op close hook."""
        self.logger.debug(build_log_message("provider", "close"))
        return None

    def call_model(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run one non-streaming chat completion call."""
        if not self.base_url:
            raise RuntimeError("MODEL_API_BASE_URL is empty")
        if not model_id:
            raise RuntimeError("TARGET_MODEL is empty")

        payload = self._build_payload(
            model_id=model_id,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
            stream=False,
        )
        self.logger.info(
            build_log_message(
                "provider",
                "call_model",
                model_id=model_id,
                message_count=len(messages),
                tool_count=len(tools or []),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._request_headers(),
            timeout=self.timeout_sec,
        )
        if response.status_code >= 400:
            self.logger.error(
                build_log_message(
                    "provider",
                    "http_error",
                    status_code=response.status_code,
                    body=response.text[:200],
                )
            )
            raise RuntimeError(f"model_http_{response.status_code}: {response.text.strip()}")
        data = response.json()
        if not isinstance(data, dict):
            self.logger.error(build_log_message("provider", "invalid_response_type", response_type=type(data).__name__))
            raise RuntimeError("invalid_model_response")
        choices = data.get("choices") if isinstance(data, dict) else []
        self.logger.info(
            build_log_message(
                "provider",
                "call_model_complete",
                model_id=model_id,
                choice_count=len(choices or []),
            )
        )
        return data

    def call_model_streaming(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        on_text_delta: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run streaming completion and reconstruct a final assistant payload."""
        if not self.base_url:
            raise RuntimeError("MODEL_API_BASE_URL is empty")
        if not model_id:
            raise RuntimeError("TARGET_MODEL is empty")

        payload = self._build_payload(
            model_id=model_id,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
            stream=True,
        )
        self.logger.info(
            build_log_message(
                "provider",
                "call_model_streaming",
                model_id=model_id,
                message_count=len(messages),
                tool_count=len(tools or []),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        finish_reason = ""
        with requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._request_headers(),
            timeout=self.timeout_sec,
            stream=True,
        ) as response:
            if response.status_code >= 400:
                self.logger.error(
                    build_log_message(
                        "provider",
                        "stream_http_error",
                        status_code=response.status_code,
                        body=response.text[:200],
                    )
                )
                raise RuntimeError(f"model_http_{response.status_code}: {response.text.strip()}")

            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = str(raw_line).strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except Exception:
                    continue
                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0] if isinstance(choices[0], dict) else {}
                delta = choice.get("delta") if isinstance(choice, dict) else {}
                delta = delta if isinstance(delta, dict) else {}

                text_delta = self._extract_delta_text(delta)
                if text_delta:
                    content_parts.append(text_delta)
                    if callable(on_text_delta):
                        on_text_delta(text_delta)

                streamed_tool_calls = delta.get("tool_calls")
                if isinstance(streamed_tool_calls, list):
                    self._merge_streamed_tool_calls(tool_calls, streamed_tool_calls)

                if choice.get("finish_reason") not in (None, ""):
                    finish_reason = str(choice.get("finish_reason") or "")

        message: Dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts),
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        response_payload = {
            "choices": [
                {
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ]
        }
        self.logger.info(
            build_log_message(
                "provider",
                "call_model_streaming_complete",
                model_id=model_id,
                content_chars=len(message.get("content", "")),
                tool_call_count=len(tool_calls),
                finish_reason=finish_reason,
            )
        )
        return response_payload

    def _build_payload(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        extra_body: Optional[Dict[str, Any]],
        stream: bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra_body:
            payload.update(extra_body)
        return payload

    def _request_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _extract_delta_text(self, delta: Dict[str, Any]) -> str:
        content = delta.get("content")
        if content in (None, ""):
            return ""
        if isinstance(content, str):
            return content
        return flatten_content(content)

    def _merge_streamed_tool_calls(
        self,
        aggregated: List[Dict[str, Any]],
        streamed_tool_calls: List[Dict[str, Any]],
    ) -> None:
        for index, chunk in enumerate(streamed_tool_calls):
            if not isinstance(chunk, dict):
                continue
            target_index = int(chunk.get("index", index) or 0)
            while len(aggregated) <= target_index:
                aggregated.append(
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                )
            target = aggregated[target_index]
            if chunk.get("id"):
                target["id"] = str(chunk.get("id") or "")
            if chunk.get("type"):
                target["type"] = str(chunk.get("type") or "function")
            function = chunk.get("function") if isinstance(chunk.get("function"), dict) else {}
            target_function = target.setdefault("function", {"name": "", "arguments": ""})
            if function.get("name"):
                target_function["name"] = str(target_function.get("name") or "") + str(function.get("name") or "")
            if function.get("arguments"):
                target_function["arguments"] = str(target_function.get("arguments") or "") + str(function.get("arguments") or "")

    def stream_text(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        """Yield only text deltas from streaming responses."""
        if not self.base_url:
            raise RuntimeError("MODEL_API_BASE_URL is empty")
        if not model_id:
            raise RuntimeError("TARGET_MODEL is empty")

        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra_body:
            payload.update(extra_body)

        self.logger.info(
            build_log_message(
                "provider",
                "stream_text_start",
                model_id=model_id,
                message_count=len(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout_sec,
            stream=True,
        ) as response:
            if response.status_code >= 400:
                self.logger.error(
                    build_log_message(
                        "provider",
                        "stream_http_error",
                        status_code=response.status_code,
                        body=response.text[:200],
                    )
                )
                raise RuntimeError(f"model_http_{response.status_code}: {response.text.strip()}")

            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = str(raw_line).strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except Exception:
                    continue
                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                delta = choices[0].get("delta") if isinstance(choices[0], dict) else {}
                delta = delta if isinstance(delta, dict) else {}
                text = delta.get("content")
                if text in (None, ""):
                    continue
                if isinstance(text, str):
                    yield text
                else:
                    flattened = flatten_content(text)
                    if flattened:
                        yield flattened

        self.logger.info(build_log_message("provider", "stream_text_complete", model_id=model_id))

    def message_from_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            self.logger.error(build_log_message("provider", "missing_choices"))
            raise RuntimeError("model_response_missing_choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            self.logger.error(build_log_message("provider", "missing_message"))
            raise RuntimeError("model_response_missing_message")
        return message

    def has_tool_calls(self, response: Dict[str, Any]) -> bool:
        message = self.message_from_response(response)
        tool_calls = message.get("tool_calls")
        return isinstance(tool_calls, list) and len(tool_calls) > 0

    def extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse provider tool-calls into runtime's normalized tool_call shape."""
        message = self.message_from_response(response)
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            self.logger.debug(build_log_message("provider", "extract_tool_calls_empty"))
            return []

        parsed: List[Dict[str, Any]] = []
        for index, tool_call in enumerate(tool_calls):
            function = tool_call.get("function") if isinstance(tool_call, dict) else {}
            function = function if isinstance(function, dict) else {}
            raw_arguments = function.get("arguments", "{}")
            if isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                try:
                    arguments = json.loads(str(raw_arguments or "{}"))
                except Exception:
                    arguments = {"_raw_arguments": str(raw_arguments or "")}
            parsed.append(
                {
                    "id": str(tool_call.get("id") or f"tool_{index}"),
                    "name": str(function.get("name") or ""),
                    "input": arguments,
                    "raw": tool_call,
                }
            )
        self.logger.info(build_log_message("provider", "extract_tool_calls", tool_call_count=len(parsed)))
        return parsed

    def extract_text(self, response: Dict[str, Any]) -> str:
        message = self.message_from_response(response)
        text = flatten_content(message.get("content"))
        self.logger.debug(build_log_message("provider", "extract_text", text_preview=text[:160]))
        return text

    def to_assistant_history_item(self, response: Dict[str, Any]) -> Dict[str, Any]:
        message = self.message_from_response(response)
        tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
        return {
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": tool_calls,
            "_kind": "assistant_tool_use" if tool_calls else "assistant_text",
        }

    def summarize(
        self,
        model_id: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Run a dedicated summary prompt using the same completion backend."""
        self.logger.info(
            build_log_message(
                "provider",
                "summarize",
                model_id=model_id,
                prompt_preview=prompt[:160],
                max_tokens=max_tokens,
            )
        )
        response = self.call_model(
            model_id=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You produce compact, structured session continuation memory.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self.extract_text(response)
