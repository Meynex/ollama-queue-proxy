"""Configuration loading and validation for ollama-queue-proxy."""

from __future__ import annotations

import os
import re
import sys
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator, model_validator


class HostConfig(BaseModel):
    url: str
    name: str
    weight: int = 1
    model_sync_interval: int = 30

    @field_validator("weight")
    @classmethod
    def positive_weight(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"ollama.hosts[].weight must be a positive integer, got {v}")
        return v

    @field_validator("model_sync_interval")
    @classmethod
    def positive_sync_interval(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                f"ollama.hosts[].model_sync_interval must be >= 1 second, got {v}"
            )
        return v


class OllamaConfig(BaseModel):
    hosts: list[HostConfig]
    health_check_interval: int = 30
    request_timeout: int = 300


class TierConfig(BaseModel):
    max_depth: int = 100
    max_wait: int = 300
    high_watermark_pct: int = 80


class QueueConfig(BaseModel):
    high: TierConfig = TierConfig(max_depth=50, max_wait=120)
    normal: TierConfig = TierConfig(max_depth=100, max_wait=300)
    low: TierConfig = TierConfig(max_depth=200, max_wait=600)
    overflow_status_code: Literal[503, 429] = 503


class WebhookConfig(BaseModel):
    enabled: bool = False
    url: str = ""
    events: list[str] = [
        "queue.full",
        "queue.high_watermark",
        "queue.drained",
        "host.unhealthy",
        "host.recovered",
    ]
    allowed_hosts: list[str] = []  # hostnames exempt from SSRF check (for internal ntfy etc.)


class ApiKeyConfig(BaseModel):
    key: str
    client_id: str
    description: str | None = None
    max_priority: Literal["high", "normal", "low"] = "normal"
    management: bool = False
    max_concurrent: int = 0  # 0 = unlimited (subject to proxy.max_concurrent)

    @field_validator("max_concurrent")
    @classmethod
    def non_negative_concurrent(cls, v: int) -> int:
        if v < 0:
            raise ValueError(
                f"auth.keys[].max_concurrent must be a non-negative integer, got {v}"
            )
        return v


class RateLimitConfig(BaseModel):
    max_failures: int = 10
    window_seconds: int = 60


class AuthConfig(BaseModel):
    enabled: bool = False
    keys: list[ApiKeyConfig] = []
    rate_limit: RateLimitConfig = RateLimitConfig()

    @model_validator(mode="after")
    def keys_required_when_enabled(self) -> "AuthConfig":
        if self.enabled and len(self.keys) == 0:
            print(
                "FATAL: auth.enabled is true but no API keys are configured. "
                "Add at least one key to auth.keys or set auth.enabled: false.",
                file=sys.stderr,
            )
            sys.exit(1)
        return self


class LoggingConfig(BaseModel):
    level: str = "info"
    format: Literal["json", "text"] = "json"


class ProxyConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 11435
    max_concurrent: int = 2
    allow_model_management: bool = False
    drain_timeout: int = 30
    max_request_body_mb: int = 50

    @field_validator("port")
    @classmethod
    def valid_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"Invalid port: {v}")
        return v


# ---------------------------------------------------------------------------
# v0.2.0 — new config sections
# ---------------------------------------------------------------------------


class InjectionListenerConfig(BaseModel):
    listen_port: int
    inject_as: str  # must match an auth.keys[].client_id
    bind: str = "127.0.0.1"

    @field_validator("listen_port")
    @classmethod
    def valid_listen_port(cls, v: int) -> int:
        if not (1024 <= v <= 65535):
            raise ValueError(
                f"client_injection.listeners[].listen_port must be in 1024–65535, got {v}"
            )
        return v


class ClientInjectionConfig(BaseModel):
    listeners: list[InjectionListenerConfig] = []
    allow_public_injection: bool = False


class RoutingConfig(BaseModel):
    strategy: Literal["model_aware", "round_robin"] = "round_robin"
    # Safety first: never send a request to an arbitrary host or retry it unless
    # explicitly enabled.  This prevents duplicate generation side effects.
    fallback: Literal["none", "any_healthy"] = "none"
    retry: bool = False
    max_retries: int = 0
    model_poll_timeout: int = 3
    # Model names are deliberately config-driven (including aliases used by clients).
    aliases: dict[str, str] = {}
    exclusive_models: list[str] = []
    per_model_concurrency: dict[str, int] = {}
    v100_models: list[str] = []
    v100_concurrency: int = 0  # 0 disables the shared V100 semaphore
    openviking_clients: list[str] = []
    openviking_priority: Literal["high", "normal", "low"] = "high"
    openviking_header: str = "X-OpenViking"

    @field_validator("max_retries", "v100_concurrency")
    @classmethod
    def non_negative_limits(cls, v: int) -> int:
        if v < 0:
            raise ValueError("routing concurrency and retry limits must be non-negative")
        return v

    @field_validator("per_model_concurrency")
    @classmethod
    def valid_model_limits(cls, v: dict[str, int]) -> dict[str, int]:
        if any(limit < 1 for limit in v.values()):
            raise ValueError("routing.per_model_concurrency values must be positive")
        return v


class EmbeddingCacheConfig(BaseModel):
    enabled: bool = False
    backend: str = "redis://localhost:6379/0"
    ttl: int = 86400
    max_entry_bytes: int = 32768
    key_prefix: str = "oqp:embed:"
    connect_timeout: int = 2


class ConcurrencyConfig(BaseModel):
    """Optional policy spelling for deployments that keep limits separate."""
    per_model: dict[str, int] = {}
    v100_limit: int = 0
    exclusive_models: list[str] = []

    @field_validator("v100_limit")
    @classmethod
    def valid_v100_limit(cls, v: int) -> int:
        if v < 0:
            raise ValueError("concurrency.v100_limit must be non-negative")
        return v

    @field_validator("per_model")
    @classmethod
    def valid_model_limits(cls, v: dict[str, int]) -> dict[str, int]:
        if any(limit < 1 for limit in v.values()):
            raise ValueError("concurrency.per_model values must be positive")
        return v


class KeepAliveConfig(BaseModel):
    default: str = "5m"
    override: bool = False


class Config(BaseModel):
    proxy: ProxyConfig = ProxyConfig()
    ollama: OllamaConfig
    queue: QueueConfig = QueueConfig()
    webhooks: WebhookConfig = WebhookConfig()
    auth: AuthConfig = AuthConfig()
    logging: LoggingConfig = LoggingConfig()
    # v0.2.0 sections
    client_injection: ClientInjectionConfig = ClientInjectionConfig()
    routing: RoutingConfig = RoutingConfig()
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
    embedding_cache: EmbeddingCacheConfig = EmbeddingCacheConfig()
    keep_alive: KeepAliveConfig = KeepAliveConfig()

    @model_validator(mode="after")
    def validate_v2_constraints(self) -> "Config":
        # Accept both the compact routing section and the dedicated concurrency
        # section; routing remains the canonical internal representation.
        if self.concurrency.per_model:
            self.routing.per_model_concurrency.update(self.concurrency.per_model)
        if self.concurrency.v100_limit:
            self.routing.v100_concurrency = self.concurrency.v100_limit
        if self.concurrency.exclusive_models:
            self.routing.exclusive_models = list(dict.fromkeys(
                self.routing.exclusive_models + self.concurrency.exclusive_models
            ))
        # Normalize all model policy keys once aliases are known. This keeps
        # aliases and canonical names on the same semaphore and exclusive gate.
        self.routing.per_model_concurrency = {
            self.routing.aliases.get(model, model): limit
            for model, limit in self.routing.per_model_concurrency.items()
        }
        self.routing.exclusive_models = list(dict.fromkeys(
            self.routing.aliases.get(model, model)
            for model in self.routing.exclusive_models
        ))
        self._validate_injection_ports()
        self._validate_inject_as_refs()
        self._validate_client_max_concurrent()
        self._validate_public_injection_bind()
        self._warn_public_injection_no_auth()
        return self

    def _validate_injection_ports(self) -> None:
        seen: set[int] = {self.proxy.port}
        for listener in self.client_injection.listeners:
            if listener.listen_port in seen:
                print(
                    f"FATAL: client_injection.listeners[].listen_port {listener.listen_port} "
                    f"conflicts with another port (proxy.port or another injection listener).",
                    file=sys.stderr,
                )
                sys.exit(1)
            seen.add(listener.listen_port)

    def _validate_inject_as_refs(self) -> None:
        known_ids = {k.client_id for k in self.auth.keys}
        for listener in self.client_injection.listeners:
            if listener.inject_as not in known_ids:
                print(
                    f"FATAL: client_injection.listeners[].inject_as '{listener.inject_as}' "
                    f"does not match any auth.keys[].client_id. Known IDs: {sorted(known_ids)}",
                    file=sys.stderr,
                )
                sys.exit(1)

    def _validate_client_max_concurrent(self) -> None:
        global_cap = self.proxy.max_concurrent
        for key in self.auth.keys:
            if key.max_concurrent > global_cap:
                print(
                    f"FATAL: auth.keys[client_id={key.client_id}].max_concurrent "
                    f"({key.max_concurrent}) exceeds proxy.max_concurrent ({global_cap}). "
                    f"Set max_concurrent <= {global_cap} or increase proxy.max_concurrent.",
                    file=sys.stderr,
                )
                sys.exit(1)

    def _validate_public_injection_bind(self) -> None:
        loopback = {"127.0.0.1", "localhost", "::1"}
        for listener in self.client_injection.listeners:
            if listener.bind in loopback:
                continue
            if not self.client_injection.allow_public_injection:
                print(
                    f"FATAL: client_injection.listeners[listen_port={listener.listen_port}].bind "
                    f"is '{listener.bind}' (non-loopback) but allow_public_injection is false. "
                    "Set allow_public_injection: true to confirm exposing an unauthenticated "
                    "injection port on the network, or change bind to 127.0.0.1.",
                    file=sys.stderr,
                )
                sys.exit(1)

    def _warn_public_injection_no_auth(self) -> None:
        loopback = {"127.0.0.1", "localhost", "::1"}
        has_non_loopback = any(
            listener.bind not in loopback
            for listener in self.client_injection.listeners
        )
        if self.client_injection.allow_public_injection and not self.auth.enabled:
            print(
                "WARNING: allow_public_injection is true AND auth.enabled is false. "
                "Injection ports will bind on all interfaces with no credential check — "
                "any host on the network can consume queue slots under an injected identity. "
                "Set auth.enabled: true or restrict allow_public_injection: false.",
                file=sys.stderr,
            )
        elif has_non_loopback:
            print(
                "WARNING: one or more client_injection.listeners bind to a non-loopback "
                "address. Injection ports bypass Bearer auth by design — any host able to "
                "reach that port can consume queue slots under the injected client identity. "
                "Restrict access at the firewall / reverse proxy layer.",
                file=sys.stderr,
            )


_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_references(value: object) -> object:
    if isinstance(value, dict):
        return {key: _expand_env_references(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_references(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ValueError(f"environment variable {name} is not set")
        return os.environ[name]

    return _ENV_REFERENCE.sub(replace, value)


def _apply_env_overrides(data: dict, prefix: str = "OQP") -> dict:
    """Apply OQP_ env var overrides onto the raw config dict using __ nesting."""
    for key, value in os.environ.items():
        if not key.startswith(prefix + "_"):
            continue
        parts = key[len(prefix) + 1 :].lower().split("__")
        target: object = data
        for index, part in enumerate(parts[:-1]):
            next_part = parts[index + 1]
            if part.isdigit():
                if not isinstance(target, list):
                    break
                while len(target) <= int(part):
                    target.append({} if not next_part.isdigit() else [])
                target = target[int(part)]
            else:
                if not isinstance(target, dict):
                    break
                if part not in target:
                    target[part] = [] if next_part.isdigit() else {}
                target = target[part]
        else:
            leaf = parts[-1]
            try:
                # YAML scalar parsing also handles null, lists, and JSON maps.
                parsed = yaml.safe_load(value)
            except yaml.YAMLError:
                parsed = value
            if isinstance(target, list) and leaf.isdigit():
                while len(target) <= int(leaf):
                    target.append(None)
                target[int(leaf)] = parsed
            elif isinstance(target, dict):
                target[leaf] = parsed
    return data


def load_config(path: str | None = None) -> Config:
    """Load configuration from YAML file with env var overrides."""
    config_path = (
        path
        or os.environ.get("CONFIG_PATH")
        or os.environ.get("OQP_CONFIG", "./config.yml")
    )
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(
            f"FATAL: Config file not found: {config_path}. "
            "Copy config.example.yml to config.yml and edit it.",
            file=sys.stderr,
        )
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"FATAL: Config file parse error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        raw = _expand_env_references(raw)
        raw = _apply_env_overrides(raw)
        return Config.model_validate(raw)
    except Exception as e:
        print(f"FATAL: Config validation error: {e}", file=sys.stderr)
        sys.exit(1)
