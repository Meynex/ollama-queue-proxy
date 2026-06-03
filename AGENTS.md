# ollama-queue-proxy

FastAPI reverse proxy for Ollama with priority queuing, per-client API key auth, embedding response caching, OpenAI-compatible API translation, and multi-host routing.

## What it does

Sits in front of one or more Ollama instances and provides: authenticated access, request prioritization, per-client concurrency limits, embedding caching, and an OpenAI-compatible API surface.

## Structure

```
src/ollama_queue_proxy/
  main.py             FastAPI app + lifespan — wires together all managers
  config.py           Config dataclass, load_config() from YAML
  auth.py             AuthManager — per-client API key validation
  queue.py            PriorityQueueManager — priority queue, TTL, pause/resume
  proxy.py            dispatch_request() — forwards requests to selected Ollama host
  routing.py          RoutingTable — model-to-host assignment
  hosts.py            HostManager — Ollama host pool, health checks
  concurrency.py      ClientConcurrencyManager — per-client request limits
  cache.py            EmbeddingCache — response cache for /api/embeddings
  openai_compat.py    OpenAI → Ollama path rewriting and response wrapping
  middleware.py       RequestContextMiddleware — injects client_id, priority
  injection.py        AppState singleton for dependency injection
  webhooks.py         WebhookManager — posts queue events to a webhook URL
  routes/
    queue.py          Queue management endpoints (/queue/status, /queue/pause, etc.)
    status.py         Health and metrics endpoints
tests/                pytest tests
config.example.yml    Reference configuration
Dockerfile            Docker image
```

## Source files

| File              | Role                                                              |
|-------------------|-------------------------------------------------------------------|
| `main.py`         | App entrypoint, lifespan startup/shutdown, manager wiring        |
| `config.py`       | YAML config loading; all tunables defined here                    |
| `auth.py`         | Per-client key lookup and validation                              |
| `queue.py`        | Async priority queue, TTL expiry, pause/resume per client         |
| `proxy.py`        | Upstream dispatch, timeout, error handling                        |
| `routing.py`      | Maps model names to target host(s)                                |
| `hosts.py`        | Host pool management and health polling                           |
| `openai_compat.py`| Path rewriting and response schema wrapping for OpenAI clients    |
| `webhooks.py`     | SSRF-validated webhook delivery for queue events                  |

## Configuration

| Env var       | Default       | Purpose                           |
|---------------|---------------|-----------------------------------|
| `CONFIG_PATH` | `config.yaml` | Path to the YAML configuration    |

All other tunables (auth keys, hosts, routing, concurrency limits, webhook URL) live in the YAML config. See `config.example.yml`.

## Architecture decisions

- **Auth is optional but warned** — if `auth.enabled: false` and the listen host is `0.0.0.0`, a startup warning is emitted. Do not remove this warning.
- **Webhook SSRF validation** — `validate_webhook_url()` runs at startup and rejects private IP targets. Do not bypass this check.
- **OpenAI compat layer** — `rewrite_path()` in `openai_compat.py` translates `/v1/chat/completions` → `/api/chat` and wraps responses back to OpenAI schema. This allows any OpenAI SDK client to point at the proxy without changes.
- **Client-controlled priority** — `X-Queue-Priority` header is parsed by `RequestContextMiddleware`. Clients cannot self-elevate beyond their configured max priority.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## Git workflow

Branch before editing — do not commit directly to `main`.
