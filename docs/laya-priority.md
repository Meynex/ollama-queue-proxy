# Laya automatic priority classification

OQP can optionally call a local Laya HTTP server before an inference request enters the queue. Laya returns a typed `choice`; OQP accepts `high`, `normal`, or `low` only when its confidence meets the configured threshold.

The integration is fail-open by default: if Laya is unavailable, the request keeps the explicit `X-Queue-Priority` value (or `normal`). An authenticated key's `max_priority` is always enforced after Laya classification, so Laya cannot grant a client more priority than its credential allows. Metadata fast-path requests do not call Laya.

## Start Laya

Run Laya separately from the Ollama proxy. This keeps PyTorch/model memory out of the queue-proxy container:

```bash
python3 -m venv /opt/laya/.venv
/opt/laya/.venv/bin/python -m pip install "laya[serve]"
LAYA_HOST=0.0.0.0 LAYA_PORT=8000 LAYA_PRELOAD=1 \
  /opt/laya/.venv/bin/laya-serve
```

Prefer an internal Docker network or a private host address. Do not expose the Laya port publicly without its own authentication and network controls. Laya's self-hosted server implements the Jev-compatible `POST /v1/systemone` protocol.

## Configure OQP

```yaml
decision_router:
  enabled: true
  url: "http://laya:8000/v1/systemone"
  timeout_ms: 100
  fail_open: true
  min_confidence: 0.85
```

For an Unraid sidecar, use the Laya container's reachable address instead of `laya`, for example `http://192.168.178.10:8000/v1/systemone`, and recreate the OQP container after changing the configuration.

## Verify

Send one interactive request and one batch request with the same authenticated key. Confirm the normal OQP response and `/queue/status`; then stop Laya and confirm requests still work with the configured fail-open behavior. If `fail_open: false` is selected, Laya outages return HTTP 503 instead.

## Design boundary

Laya classifies queue priority; it does not select an Ollama model or GPU in this first integration. Existing model-aware routing, per-GPU limits, authentication scopes, and priority ceilings remain authoritative.
