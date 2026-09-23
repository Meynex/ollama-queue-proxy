# Laya automatic priority classification

OQP can optionally call a local Laya HTTP server before an inference request enters the queue. Laya returns a typed `choice`; OQP accepts `high`, `normal`, or `low` only when its confidence meets the configured threshold.

The integration is fail-open by default: if Laya is unavailable, the request keeps the explicit `X-Queue-Priority` value (or `normal`). An authenticated key's `max_priority` is always enforced after Laya classification, so Laya cannot grant a client more priority than its credential allows. Metadata fast-path requests do not call Laya.

## Start Laya

Run Laya separately from the Ollama proxy. This keeps PyTorch/model memory out of the queue-proxy container. The published CPU sidecar is:

```text
ghcr.io/meynex/laya-decision-service:0.1.0
```

For Docker Compose, set a secret without committing it and start the optional service:

```bash
export LAYA_API_KEY='use-a-local-secret'
docker compose -f docker-compose.yml -f docker-compose.laya.yml up -d
```

The sidecar preloads the English and multilingual checkpoints and persists them in `oqp-laya-model-cache`. It listens on container port `8000`; the combined Compose network lets OQP use `http://laya:8000/v1/systemone`.

For a manual Unraid container, use the same image, map container port `8000`, set `LAYA_DEVICE=cpu`, `LAYA_PRELOAD=1`, `LAYA_MODELS=english,multilingual`, and set `LAYA_API_KEY` as a container secret/environment value. Do not print or commit that value. Configure OQP's `decision_router.api_key` through `OQP_DECISION_ROUTER__API_KEY`.

Prefer an internal Docker network or a private host address. Do not expose the Laya port publicly without its own authentication and network controls. Laya's self-hosted server implements the Jev-compatible `POST /v1/systemone` protocol.

## Configure OQP

```yaml
decision_router:
  enabled: true
  url: "http://laya:8000/v1/systemone"
  # Prefer OQP_DECISION_ROUTER__API_KEY for this secret.
  api_key: "replace-me"
  timeout_ms: 100
  fail_open: true
  min_confidence: 0.85
```

For an Unraid sidecar, use the Laya container's reachable address instead of `laya`, for example `http://192.168.178.10:8000/v1/systemone`, and recreate the OQP container after changing the configuration.

## Verify

Send one interactive request and one batch request with the same authenticated key. Confirm the normal OQP response and `/queue/status`; then stop Laya and confirm requests still work with the configured fail-open behavior. If `fail_open: false` is selected, Laya outages return HTTP 503 instead.

## Design boundary

Laya classifies queue priority; it does not select an Ollama model or GPU in this first integration. Existing model-aware routing, per-GPU limits, authentication scopes, and priority ceilings remain authoritative.
