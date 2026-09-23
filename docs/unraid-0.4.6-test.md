# Unraid smoke test: `ollama-queue-proxy:0.4.6`

This runbook verifies the published image and per-GPU host limits without exposing API keys in shell history or logs.

## 1. Container configuration

Use the image:

```text
ghcr.io/meynex/ollama-queue-proxy:0.4.6
```

Keep the existing config and add these container environment variables. They address hosts by their configured list index:

```text
OQP_OLLAMA__HOSTS__0__MAX_CONCURRENT=1
OQP_OLLAMA__HOSTS__1__MAX_CONCURRENT=2
```

For the current deployment the indexes are:

```text
0 = v100
1 = rtx
```

Recreate the container after changing the template. Do not paste API keys into commands or reports.

## 2. Health and capacity check

From a client that can reach the router, set the existing key in the shell without printing it:

```bash
read -rsp 'Router key: ' OQP_KEY; echo
BASE_URL='http://192.168.178.10:11435'

curl -fsS "$BASE_URL/health"
curl -fsS "$BASE_URL/queue/status" \
  -H "Authorization: Bearer $OQP_KEY"
unset OQP_KEY
```

The status response must show:

```json
"concurrency": {"max": 3, "global_configured_max": 2}
```

and host entries equivalent to:

```json
{"name":"v100", "max_concurrent":1, "active_requests":0}
{"name":"rtx",  "max_concurrent":2, "active_requests":0}
```

Both hosts must be `healthy: true`.

## 3. Functional probes

Use the OpenAI-compatible endpoint with a small non-reasoning probe. Keep the key in an environment variable and do not print the response headers wholesale:

```bash
export OQP_KEY
curl -fsS --max-time 120 "$BASE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $OQP_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3:8b","messages":[{"role":"user","content":"Reply only RTX-OK"}],"think":false,"max_tokens":32,"stream":false}'
```

The response must be HTTP 200 with `RTX-OK`. The router status should increment one host's `requests_handled` counter and then return both `active_requests` values to zero.

## 4. Independent GPU concurrency

Run two requests concurrently: one model available on the V100 and one model available on the RTX host. Use a short Python probe so each request has an independent timeout and the output contains only status, elapsed time, and the selected host:

```bash
python3 - <<'PY'
import concurrent.futures, json, os, time, urllib.request

base = os.environ.get("BASE_URL", "http://192.168.178.10:11435")
key = os.environ["OQP_KEY"]
requests = {
    "v100": "qwen3.8:27b",
    "rtx": "qwen3:8b",
}

def call(item):
    name, model = item
    payload = {"model": model, "messages": [{"role": "user", "content": "Reply only OK"}],
               "think": False, "max_tokens": 32, "stream": False}
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=180) as response:
        body = json.loads(response.read())
        return {"label": name, "status": response.status,
                "elapsed_s": round(time.monotonic() - started, 2),
                "host": response.headers.get("X-Failover-Host"),
                "content": body.get("choices", [{}])[0].get("message", {}).get("content")}

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    for result in pool.map(call, requests.items()):
        print(result)
PY
```

Both requests must complete successfully. Then query `/queue/status` again and confirm no request remains active. A request sent to a saturated host may wait, but it must not increase the other host's `active_requests` above its configured limit.

## 5. Rollback

If health, authentication, or response validation fails, restore the previous image tag (currently `0.4.5`), recreate the container, and remove the two `OQP_OLLAMA__HOSTS__*__MAX_CONCURRENT` overrides. Capture the sanitized `/queue/status` response and container logs before changing anything else.
