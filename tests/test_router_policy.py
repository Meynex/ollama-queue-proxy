from __future__ import annotations

import asyncio

import pytest

from ollama_queue_proxy.config import RoutingConfig
from ollama_queue_proxy.concurrency import ClientConcurrencyManager


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
