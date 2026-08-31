from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from ollama_queue_proxy.concurrency import ClientConcurrencyManager
from ollama_queue_proxy.config import (
    ApiKeyConfig,
    ConcurrencyConfig,
    RoutingConfig,
    _expand_env_references,
)


def test_router_policy_defaults_are_safe():
    cfg = RoutingConfig()
    assert cfg.fallback == "none"
    assert cfg.retry is False
    assert cfg.max_retries == 0


def test_router_policy_alias_and_limits():
    cfg = RoutingConfig(
        aliases={"v100-chat": "llama3"},
        per_model_concurrency={"llama3": 1},
        v100_models=["llama3"],
        v100_concurrency=1,
        exclusive_models=["llama3"],
    )
    assert cfg.aliases["v100-chat"] == "llama3"


@pytest.mark.asyncio
async def test_model_limit_serializes_aliases():
    manager = ClientConcurrencyManager([], RoutingConfig(
        aliases={"alias": "model"}, per_model_concurrency={"model": 1}
    ))
    await manager.acquire(None, model="alias")
    acquired = asyncio.Event()

    async def waiter():
        await manager.acquire(None, model="model")
        acquired.set()

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    assert not acquired.is_set()
    manager.release(None, model="alias")
    await asyncio.wait_for(acquired.wait(), timeout=1)
    manager.release(None, model="model")
    await task


def test_policy_limits_are_positive():
    with pytest.raises(ValidationError):
        ConcurrencyConfig(per_model={"model": 0})
    with pytest.raises(ValidationError):
        RoutingConfig(per_model_concurrency={"model": -1})


def test_env_references_expand_and_reject_missing(monkeypatch):
    monkeypatch.setenv("ROUTER_SECRET", "secret-value")
    assert _expand_env_references({"key": "${ROUTER_SECRET}"}) == {"key": "secret-value"}
    with pytest.raises(ValueError, match="MISSING_SECRET"):
        _expand_env_references("${MISSING_SECRET}")


@pytest.mark.asyncio
async def test_cancelled_client_wait_releases_model_slot():
    key = ApiKeyConfig(key="secret", client_id="client", max_concurrent=1)
    manager = ClientConcurrencyManager(
        [key], RoutingConfig(per_model_concurrency={"model": 2})
    )
    await manager.acquire("client", model="model")
    waiter = asyncio.create_task(manager.acquire("client", model="model"))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert manager._model_semaphores["model"]._value == 1
    manager.release("client", model="model")


@pytest.mark.asyncio
async def test_exclusive_model_blocks_other_models():
    manager = ClientConcurrencyManager(
        [], RoutingConfig(exclusive_models=["large"], per_model_concurrency={"small": 2})
    )
    await manager.acquire(None, model="large")
    waiting = asyncio.create_task(manager.acquire(None, model="small"))
    await asyncio.sleep(0)
    assert not waiting.done()
    manager.release(None, model="large")
    await asyncio.wait_for(waiting, timeout=1)
    manager.release(None, model="small")
