"""Optional typed-decision integration for automatic queue priority."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from .config import DecisionRouterConfig

logger = logging.getLogger(__name__)
_PRIORITIES = frozenset({"high", "normal", "low"})
_MAX_CLASSIFICATION_BYTES = 8 * 1024
_MAX_CLASSIFICATION_MESSAGES = 6
_MAX_MESSAGE_CONTENT_CHARS = 1_200
_MAX_ROLE_CHARS = 64
_MAX_MODEL_CHARS = 256


def _clip_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "\n...[truncated]...\n"
    available = max(0, limit - len(marker))
    head = available // 2
    return value[:head] + marker + value[-(available - head):]


def _compact_message(message: Any) -> Any:
    if not isinstance(message, dict):
        return _clip_text(str(message), _MAX_MESSAGE_CONTENT_CHARS)
    compact: dict[str, Any] = {}
    if "role" in message:
        compact["role"] = _clip_text(str(message["role"]), _MAX_ROLE_CHARS)
    content = message.get("content")
    if isinstance(content, str):
        compact["content"] = _clip_text(content, _MAX_MESSAGE_CONTENT_CHARS)
    elif content is not None:
        encoded = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        compact["content"] = _clip_text(encoded, _MAX_MESSAGE_CONTENT_CHARS)
    return compact


def _compact_body(body: bytes) -> Any:
    try:
        parsed = json.loads(body) if body else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _clip_text(body.decode("utf-8", errors="replace"), _MAX_CLASSIFICATION_BYTES)

    if not isinstance(parsed, dict):
        return parsed

    compact: dict[str, Any] = {}
    if isinstance(parsed.get("model"), str):
        compact["model"] = _clip_text(parsed["model"], _MAX_MODEL_CHARS)
    for key in ("stream", "think"):
        if key in parsed:
            compact[key] = parsed[key]
    messages = parsed.get("messages")
    if isinstance(messages, list):
        tail_indices = set(
            range(
                max(0, len(messages) - (_MAX_CLASSIFICATION_MESSAGES - 1)),
                len(messages),
            )
        )
        last_user_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], dict)
                and messages[index].get("role") == "user"
            ),
            None,
        )
        if last_user_index is not None:
            tail_indices.add(last_user_index)
        compact["messages"] = [
            _compact_message(messages[index]) for index in sorted(tail_indices)
        ]
    for key in ("prompt", "input", "inputs", "text", "query"):
        value = parsed.get(key)
        if isinstance(value, str):
            compact[key] = _clip_text(value, _MAX_MESSAGE_CONTENT_CHARS)
        elif value is not None:
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            compact[key] = _clip_text(encoded, _MAX_MESSAGE_CONTENT_CHARS)
    return compact


def _bounded_body(body: Any) -> Any:
    def encoded_size(value: Any) -> int:
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    if encoded_size(body) <= _MAX_CLASSIFICATION_BYTES:
        return body
    if isinstance(body, dict) and isinstance(body.get("messages"), list):
        latest_user = next(
            (
                message
                for message in reversed(body["messages"])
                if isinstance(message, dict) and message.get("role") == "user"
            ),
            None,
        )
        latest = latest_user or (body["messages"][-1] if body["messages"] else {})
        minimal = {"messages": [_compact_message(latest)]}
        if isinstance(body.get("model"), str):
            minimal["model"] = _clip_text(body["model"], _MAX_MODEL_CHARS)
        if encoded_size(minimal) <= _MAX_CLASSIFICATION_BYTES:
            return minimal
        return {"messages": [{"role": "user", "content": "[truncated]"}]}
    return _clip_text(
        json.dumps(body, ensure_ascii=False, separators=(",", ":")),
        _MAX_CLASSIFICATION_BYTES // 2,
    )


class DecisionRouterUnavailable(RuntimeError):
    """Raised when priority classification is required but unavailable."""


class DecisionRouter:
    """Call a Laya-compatible `/v1/systemone` decision service.

    The service is deliberately HTTP-based: the proxy stays small and responsive,
    while Laya's PyTorch runtime can live in its own CPU/GPU container. A failure
    normally leaves the caller's explicit/header priority unchanged.
    """

    def __init__(self, config: DecisionRouterConfig, client: httpx.AsyncClient) -> None:
        self._config = config
        self._client = client

    async def classify_priority(self, path: str, body: bytes) -> str | None:
        """Return a confident Laya priority, or None when classification abstains."""
        if not self._config.enabled:
            return None

        state: dict[str, Any] = {"path": path}
        state["body"] = _bounded_body(_compact_body(body))

        payload = {
            "state": state,
            "questions": {
                "priority": {
                    "type": "choice",
                    "instructions": (
                        "Which queue priority should handle this inference request? "
                        "Choose high only for interactive or urgent work, low for bulk or "
                        "background work, and normal otherwise."
                    ),
                    "criteria": {
                        "high": "interactive, urgent, or user-facing request",
                        "normal": "ordinary inference request",
                        "low": "bulk, batch, indexing, or background request",
                    },
                }
            },
        }

        try:
            headers = {}
            if self._config.api_key is not None:
                headers["Authorization"] = (
                    f"Bearer {self._config.api_key.get_secret_value()}"
                )
            response = await self._client.post(
                self._config.url,
                json=payload,
                headers=headers,
                timeout=self._config.timeout_ms / 1000,
            )
            response.raise_for_status()
            result = response.json()
            answer = result.get("answers", {}).get("priority", {})
            choice = answer.get("choice")
            confidence = float(answer.get("confidence", 0.0))
            if choice not in _PRIORITIES or confidence < self._config.min_confidence:
                logger.info(
                    "decision_router.abstain choice=%s confidence=%.3f threshold=%.3f",
                    choice,
                    confidence,
                    self._config.min_confidence,
                )
                return None
            logger.debug(
                "decision_router.priority path=%s choice=%s confidence=%.3f",
                path,
                choice,
                confidence,
            )
            return choice
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            if self._config.fail_open:
                logger.warning("decision_router.unavailable error=%s fail_open=true", exc)
                return None
            raise DecisionRouterUnavailable("decision router unavailable") from exc
