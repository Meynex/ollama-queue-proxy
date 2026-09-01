"""OpenAI-compatible request and response translation for Ollama."""

from __future__ import annotations

_OPENAI_COMPAT_PATHS: frozenset[str] = frozenset({
    "/v1/embeddings", "/v1/chat/completions",
})


def _normalized(path: str) -> str:
    return "/" + path.lstrip("/")


def is_openai_compat_path(path: str) -> bool:
    """Return True if the path should be handled by the compat layer."""
    return _normalized(path) in _OPENAI_COMPAT_PATHS


def rewrite_path(path: str) -> str:
    """Rewrite an OpenAI-compat path to the native Ollama equivalent."""
    normalized = _normalized(path)
    if normalized == "/v1/embeddings":
        return "/api/embed"
    if normalized == "/v1/chat/completions":
        return "/api/chat"
    return path


def translate_chat_request(body: dict) -> dict:
    """Translate OpenAI chat fields while retaining Ollama-native options."""
    result = dict(body)
    # Ollama uses the same stream flag; retaining it selects NDJSON or JSON.
    result["stream"] = bool(body.get("stream", False))
    options = dict(result.pop("options", {}) or {})
    for source, target in (("max_tokens", "num_predict"), ("max_completion_tokens", "num_predict"),
                           ("top_p", "top_p"), ("temperature", "temperature"),
                           ("seed", "seed"), ("stop", "stop")):
        if source in result and target not in options:
            options[target] = result.pop(source)
    if options:
        result["options"] = options
    # OpenAI clients sometimes send unsupported bookkeeping fields.
    for field in ("n", "user", "presence_penalty", "frequency_penalty", "logit_bias"):
        result.pop(field, None)
    return result


def wrap_chat_response(body: dict, model: str | None = None) -> dict:
    """Wrap one Ollama non-streaming chat response as OpenAI ChatCompletion."""
    message = body.get("message") or {}
    prompt_tokens = body.get("prompt_eval_count", 0) or 0
    completion_tokens = body.get("eval_count", 0) or 0
    return {
        "id": "chatcmpl-ollama",
        "object": "chat.completion",
        "created": 0,
        "model": model or body.get("model", ""),
        "choices": [{"index": 0, "message": {
            "role": message.get("role", "assistant"),
            "content": message.get("content", ""),
        }, "finish_reason": body.get("done_reason", "stop") if body.get("done", True) else None}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


def wrap_chat_chunk(body: dict, model: str | None = None) -> dict:
    """Convert an Ollama streaming object to an OpenAI chunk object."""
    message = body.get("message") or {}
    done = body.get("done", False)
    delta = {"role": message["role"]} if message.get("role") else {}
    if message.get("content"):
        delta["content"] = message["content"]
    return {
        "id": "chatcmpl-ollama", "object": "chat.completion.chunk", "created": 0,
        "model": model or body.get("model", ""),
        "choices": [{"index": 0, "delta": delta,
                     "finish_reason": (body.get("done_reason") or "stop") if done else None}],
    }


def wrap_error(body: dict) -> dict:
    """Convert an Ollama error object to the standard OpenAI error envelope."""
    message = body.get("error", "upstream error") if isinstance(body, dict) else "upstream error"
    return {"error": {"message": message, "type": "server_error", "param": None, "code": None}}


def wrap_response(ollama_body: dict, model: str | None = None) -> dict:
    """Wrap an Ollama /api/embed response body in the OpenAI embeddings format.

    Args:
        ollama_body: Parsed JSON response from Ollama /api/embed.
        model: Model name override (falls back to ollama_body["model"]).

    Returns:
        Dict conforming to the OpenAI /v1/embeddings response schema.
    """
    embeddings: list[list[float]] = ollama_body.get("embeddings") or []
    resolved_model: str = model or ollama_body.get("model", "")

    data = [
        {
            "object": "embedding",
            "embedding": vec,
            "index": i,
        }
        for i, vec in enumerate(embeddings)
    ]

    return {
        "object": "list",
        "data": data,
        "model": resolved_model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }
