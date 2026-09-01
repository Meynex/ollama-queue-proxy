"""Model-aware routing table with weighted round-robin and background polling."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import httpx

from .config import OllamaConfig, RoutingConfig

logger = logging.getLogger(__name__)


@dataclass
class HostRoutingState:
    """Per-host routing state. Structured for v0.3 VRAM field extension."""

    url: str
    name: str
    weight: int
    model_sync_interval: int
    loaded_models: set[str] = field(default_factory=set)
    active_models: set[str] = field(default_factory=set)
    reachable: bool = True
    # v0.3 placeholder fields (not yet populated)
    # vram_total_gb: float | None = None
    # vram_used_gb: float | None = None
    # ps_last_checked: datetime | None = None


class RoutingTable:
    """
    Maintains a live map of (host → loaded_models) via per-host background pollers.

    Weighted round-robin is deterministic: a counter advances on each pick call
    and wraps around the weight-expanded host list.
    """

    def __init__(
        self,
        ollama_config: OllamaConfig,
        routing_config: RoutingConfig,
        http_client: httpx.AsyncClient,
    ) -> None:
        self._routing_cfg = routing_config
        self._client = http_client
        self._poll_timeout = routing_config.model_poll_timeout
        self._lock = asyncio.Lock()

        self._states: dict[str, HostRoutingState] = {
            h.name: HostRoutingState(
                url=h.url,
                name=h.name,
                weight=h.weight,
                model_sync_interval=h.model_sync_interval,
            )
            for h in ollama_config.hosts
        }

        # Deterministic weighted round-robin: counter increments on every pick
        self._rr_counter: int = 0

        # Metrics counters
        self.routing_decisions: dict[str, int] = {
            "model_match": 0,
            "round_robin": 0,
            "fallback": 0,
        }

        self._poll_tasks: list[asyncio.Task] = []


    async def startup_probe(self) -> None:
        """
        Synchronous initial poll of all hosts. Fail-fast if no host responds.
        Called before accepting requests.
        """
        await asyncio.gather(
            *[self._poll_host(state) for state in self._states.values()],
            return_exceptions=True,
        )
        reachable_count = sum(
            1 for state in self._states.values() if state.reachable
        )
        if reachable_count == 0:
            import sys

            print(
                "FATAL: routing startup probe — no Ollama host responded to /api/tags. "
                "Check that at least one host in ollama.hosts is reachable.",
                file=sys.stderr,
            )
            sys.exit(1)

        for state in self._states.values():
            logger.info(
                "routing.startup_probe host=%s reachable=%s models=%d",
                state.name,
                state.reachable,
                len(state.loaded_models),
            )

    def start_background_pollers(self) -> None:
        for state in self._states.values():
            self._poll_tasks.append(asyncio.create_task(self._poll_loop(state)))
            self._poll_tasks.append(asyncio.create_task(self._active_poll_loop(state)))

    async def stop(self) -> None:
        for task in self._poll_tasks:
            task.cancel()
        if self._poll_tasks:
            await asyncio.gather(*self._poll_tasks, return_exceptions=True)
        self._poll_tasks.clear()

    async def _poll_loop(self, state: HostRoutingState) -> None:
        while True:
            await asyncio.sleep(state.model_sync_interval)
            await self._poll_models(state)

    async def _active_poll_loop(self, state: HostRoutingState) -> None:
        while True:
            await asyncio.sleep(self._routing_cfg.active_model_poll_interval)
            await self._poll_active_models(state)

    async def _poll_host(self, state: HostRoutingState) -> None:
        """Poll installed inventory and active inventory during startup/tests."""
        await self._poll_models(state)
        if state.reachable:
            await self._poll_active_models(state)

    async def _poll_models(self, state: HostRoutingState) -> None:
        try:
            resp = await self._client.get(f"{state.url}/api/tags", timeout=self._poll_timeout)
            resp.raise_for_status()
            data = resp.json()
            models = {m["name"] for m in data.get("models", []) if "name" in m}
            async with self._lock:
                state.loaded_models = models
                state.reachable = True
            logger.debug("routing.poll host=%s models=%d", state.name, len(models))
        except Exception as e:
            async with self._lock:
                state.reachable = False
            logger.warning("routing.poll_failed host=%s error=%s", state.name, e)

    async def _poll_active_models(self, state: HostRoutingState) -> None:
        try:
            resp = await self._client.get(f"{state.url}/api/ps", timeout=self._poll_timeout)
            resp.raise_for_status()
            data = resp.json()
            active = {m["name"] for m in data.get("models", []) if "name" in m}
            async with self._lock:
                state.active_models = active
            logger.debug("routing.active_poll host=%s models=%d", state.name, len(active))
        except Exception as e:
            # /api/ps is optional on older Ollama versions; tags reachability remains valid.
            async with self._lock:
                state.active_models = set()
            logger.debug("routing.active_poll_unavailable host=%s error=%s", state.name, e)

    def canonical_model(self, model: str | None) -> str | None:
        """Return the configured canonical name for a client-facing model."""
        if not model:
            return model
        canonical = model
        seen: set[str] = set()
        while canonical in self._routing_cfg.aliases and canonical not in seen:
            seen.add(canonical)
            canonical = self._routing_cfg.aliases[canonical]
        return canonical

    def invalidate(self, host_name: str, model: str) -> None:
        """
        Fast-path invalidation: remove model from host's loaded set immediately
        when upstream returns 'model not found'. No lock needed — set.discard is GIL-safe
        for small sets, but we use the lock for consistency.
        """
        model = self.canonical_model(model) or model
        state = self._states.get(host_name)
        if state:
            state.loaded_models.discard(model)
            state.active_models.discard(model)
            logger.debug(
                "routing.invalidated host=%s model=%s", host_name, model
            )

    def pick(self, model: str | None) -> HostRoutingState | None:
        """
        Pick a host using the configured strategy.

        - model_aware: prefer hosts with the model loaded; fall back per routing.fallback
        - round_robin: weighted round-robin across all reachable hosts (ignores model table)

        Returns None if no host is available.
        """
        strategy = self._routing_cfg.strategy
        model = self.canonical_model(model)

        if strategy == "model_aware" and model:
            return self._pick_model_aware(model)
        else:
            result = self._pick_round_robin(list(self._states.values()))
            if result:
                self.routing_decisions["round_robin"] += 1
            return result

    def _pick_model_aware(self, model: str) -> HostRoutingState | None:
        reachable = [s for s in self._states.values() if s.reachable]
        with_model = [s for s in reachable if model in s.loaded_models]

        if with_model:
            result = self._pick_preferred(model, with_model)
            if result:
                self.routing_decisions["model_match"] += 1
            return result

        # Fall back — no host has the model loaded. Disabled by default because
        # sending a model to an arbitrary GPU can cause an unexpected large load.
        fallback = self._routing_cfg.fallback
        if fallback == "any_healthy":
            result = self._pick_preferred(model, reachable)
            if result:
                self.routing_decisions["fallback"] += 1
            return result

        return None

    def _pick_preferred(
        self, model: str, candidates: list[HostRoutingState]
    ) -> HostRoutingState | None:
        """Pick the first configured preferred host, then balance the rest."""
        by_name = {state.name: state for state in candidates}
        # Explicit host preference remains authoritative, including for fallback.
        for host_name in self._routing_cfg.preferred_hosts.get(model, []):
            state = by_name.get(host_name)
            if state:
                return state

        if self._routing_cfg.active_model_preference:
            active = [s for s in candidates if model in s.active_models]
            if active:
                return self._pick_round_robin(
                    active,
                    weight_multiplier=self._routing_cfg.active_model_bonus,
                )
        return self._pick_round_robin(candidates)

    def _pick_round_robin(
        self, candidates: list[HostRoutingState], weight_multiplier: float = 1.0
    ) -> HostRoutingState | None:
        """
        Deterministic weighted round-robin over the given candidates.
        Builds a weight-expanded sequence and selects by counter modulo total weight.
        """
        if not candidates:
            return None

        # Build the weighted sequence (deterministic, not stochastic)
        weighted: list[HostRoutingState] = []
        for state in candidates:
            weighted.extend([state] * max(1, round(state.weight * weight_multiplier)))

        if not weighted:
            return None

        idx = self._rr_counter % len(weighted)
        self._rr_counter += 1
        return weighted[idx]

    def host_model_counts(self) -> dict[str, int]:
        """Return {host_name: loaded_model_count} for metrics."""
        return {name: len(s.loaded_models) for name, s in self._states.items()}

    def loaded_models_by_host(self) -> dict[str, set[str]]:
        """Return {host_name: set_of_model_names} snapshot for metrics."""
        return {name: set(s.loaded_models) for name, s in self._states.items()}
