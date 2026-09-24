from __future__ import annotations

import json

import httpx
import pytest

from ollama_queue_proxy.config import DecisionRouterConfig
from ollama_queue_proxy.decision_router import DecisionRouter, DecisionRouterUnavailable


@pytest.mark.asyncio
async def test_laya_choice_sets_priority_when_confident():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = request.read()
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={"answers": {"priority": {"choice": "high", "confidence": 0.94}}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = DecisionRouter(
            DecisionRouterConfig(enabled=True, api_key="sidecar-secret"), client
        )
        result = await router.classify_priority(
            "/api/chat", b'{"model":"llama3","messages":[]}'
        )

    assert result == "high"
    assert seen["authorization"] == "Bearer sidecar-secret"
    assert b'"priority"' in seen["payload"]


@pytest.mark.asyncio
async def test_large_chat_payload_keeps_latest_message_and_stays_bounded():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        seen["size"] = len(request.content)
        return httpx.Response(
            200,
            json={"answers": {"priority": {"choice": "high", "confidence": 0.94}}},
        )

    messages = [
        {"role": "user", "content": "old context " * 5_000}
        for _ in range(10)
    ]
    messages.append({"role": "user", "content": "URGENT latest request"})
    messages.append({"role": "assistant", "content": "trailing response"})
    body = json.dumps(
        {
            "model": "qwen3:8b " * 1_000,
            "messages": messages,
            "input": "batch context",
        }
    ).encode()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = DecisionRouter(DecisionRouterConfig(enabled=True), client)
        result = await router.classify_priority("/v1/chat/completions", body)

    assert result == "high"
    compact_body = seen["payload"]["state"]["body"]
    assert any(
        message.get("content") == "URGENT latest request"
        for message in compact_body["messages"]
    )
    assert compact_body["input"] == "batch context"
    assert len(compact_body["model"]) <= 256
    assert seen["size"] < 12 * 1024


@pytest.mark.asyncio
async def test_laya_abstains_below_confidence_threshold():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"answers": {"priority": {"choice": "low", "confidence": 0.5}}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = DecisionRouter(DecisionRouterConfig(enabled=True), client)
        assert await router.classify_priority("/api/generate", b"{}") is None


@pytest.mark.asyncio
async def test_laya_failure_fails_open_by_default():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sidecar down")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = DecisionRouter(DecisionRouterConfig(enabled=True), client)
        assert await router.classify_priority("/api/chat", b"{}") is None


@pytest.mark.asyncio
async def test_laya_failure_can_fail_closed():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sidecar down")

    config = DecisionRouterConfig(enabled=True, fail_open=False)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        router = DecisionRouter(config, client)
        with pytest.raises(DecisionRouterUnavailable):
            await router.classify_priority("/api/chat", b"{}")
