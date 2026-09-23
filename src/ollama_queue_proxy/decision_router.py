"""Optional typed-decision integration for automatic queue priority."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from .config import DecisionRouterConfig

logger = logging.getLogger(__name__)
_PRIORITIES = frozenset({"high", "normal", "low"})
_MAX_CLASSIFICATION_BYTES = 64 * 1024


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
        sample = body[:_MAX_CLASSIFICATION_BYTES]
        try:
            state["body"] = json.loads(sample) if sample else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            state["body"] = sample.decode("utf-8", errors="replace")

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
            response = await self._client.post(
                self._config.url,
                json=payload,
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
