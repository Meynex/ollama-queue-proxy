"""OpenAI-compatible request and response translation for Ollama."""

from __future__ import annotations

import json

_OPENAI_COMPAT_PATHS: frozenset[str] = frozenset({
    "/v1/embeddings", "/v1/chat/completions",
})
_MIN_REASONING_NUM_PREDICT = 256


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


def _content_part_fallback(part: object) -> str:
    """Serialize an unsupported content part without silently dropping it."""
    try:
        return json.dumps(part, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(part)


def _content_to_ollama_string(content: object) -> object:
    """Convert OpenAI content parts to text accepted by Ollama /api/chat."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _content_part_fallback(content)

    rendered: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
            rendered.append(text if isinstance(text, str) else _content_part_fallback(part))
        elif isinstance(part, dict) and part.get("type") == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                rendered.append(f"[image_url: {image_url['url']}]")
            else:
                rendered.append(f"[image_url: {_content_part_fallback(image_url)}]")
        else:
            rendered.append(f"[unsupported_content_part: {_content_part_fallback(part)}]")
    return "\n".join(rendered)


def _ollama_tool_calls(tool_calls: object) -> list[dict]:
    """Translate OpenAI assistant tool calls to Ollama's native message shape."""
    if not isinstance(tool_calls, list):
        return []

    translated: list[dict] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue

        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError):
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        translated.append({
            "function": {"name": function["name"], "arguments": arguments},
        })
    return translated


def _translate_chat_message(message: object) -> object:
    """Convert one OpenAI chat-history message to Ollama's native format."""
    if not isinstance(message, dict):
        return message

    translated = dict(message)
    if "content" in translated:
        translated["content"] = _content_to_ollama_string(translated["content"])
    if "tool_calls" in translated:
        translated["tool_calls"] = _ollama_tool_calls(translated["tool_calls"])
    if translated.get("role") == "tool":
        # Ollama associates a tool response by message order, not OpenAI's ID.
        translated.pop("tool_call_id", None)
    return translated


def translate_chat_request(body: dict) -> dict:
    """Translate OpenAI chat fields while retaining Ollama-native options."""
    result = dict(body)
    messages = body.get("messages")
    if isinstance(messages, list):
        result["messages"] = [_translate_chat_message(message) for message in messages]
    # Ollama uses the same stream flag; retaining it selects NDJSON or JSON.
    result["stream"] = bool(body.get("stream", False))
    # Ollama supports native thinking while OpenAI clients commonly expose the
    # control as reasoning_effort. Preserve an explicit native `think` value.
    if "think" not in result and "reasoning_effort" in result:
        result["think"] = result["reasoning_effort"] != "off"
    result.pop("reasoning_effort", None)
    options = dict(result.pop("options", {}) or {})
    for source, target in (("max_tokens", "num_predict"), ("max_completion_tokens", "num_predict"),
                           ("top_p", "top_p"), ("temperature", "temperature"),
                           ("seed", "seed"), ("stop", "stop")):
        if source in result and target not in options:
            options[target] = result.pop(source)
    # Qwen thinking consumes the same Ollama prediction budget as the final
    # answer. Very small OpenAI max_tokens values can therefore terminate after
    # thinking and return an apparently empty answer. Reserve enough room for a
    # useful final response while preserving larger caller-provided budgets.
    if result.get("think") is True and "num_predict" in options:
        try:
            options["num_predict"] = max(
                int(options["num_predict"]), _MIN_REASONING_NUM_PREDICT
            )
        except (TypeError, ValueError):
            pass
    if options:
        result["options"] = options
    # OpenAI clients sometimes send unsupported bookkeeping fields.
    for field in ("n", "user", "presence_penalty", "frequency_penalty", "logit_bias"):
        result.pop(field, None)
    return result


def _openai_tool_calls(tool_calls: object, *, include_index: bool) -> list[dict]:
    """Translate Ollama native tool calls to the OpenAI Chat Completions shape."""
    if not isinstance(tool_calls, list):
        return []

    translated: list[dict] = []
    for position, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue

        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            try:
                arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            except (TypeError, ValueError):
                arguments = json.dumps(str(arguments), ensure_ascii=False)

        translated_call = {
            "id": tool_call.get("id") or f"call_{position}",
            "type": "function",
            "function": {"name": function["name"], "arguments": arguments},
        }
        if include_index:
            index = function.get("index", position)
            translated_call["index"] = index if isinstance(index, int) else position
        translated.append(translated_call)
    return translated


def wrap_chat_response(body: dict, model: str | None = None) -> dict:
    """Wrap one Ollama non-streaming chat response as OpenAI ChatCompletion."""
    message = body.get("message") or {}
    tool_calls = _openai_tool_calls(message.get("tool_calls"), include_index=False)
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
            **({"reasoning": message["thinking"]} if message.get("thinking") else {}),
            **({"tool_calls": tool_calls} if tool_calls else {}),
        }, "finish_reason": "tool_calls" if tool_calls else (
            body.get("done_reason", "stop") if body.get("done", True) else None
        )}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


def wrap_chat_chunk(
    body: dict, model: str | None = None, *, tool_calls_seen: bool = False
) -> dict:
    """Convert an Ollama streaming object to an OpenAI chunk object."""
    message = body.get("message") or {}
    tool_calls = _openai_tool_calls(message.get("tool_calls"), include_index=True)
    done = body.get("done", False)
    delta = {"role": message["role"]} if message.get("role") else {}
    if message.get("content"):
        delta["content"] = message["content"]
    if message.get("thinking"):
        delta["reasoning"] = message["thinking"]
    if tool_calls:
        delta["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-ollama", "object": "chat.completion.chunk", "created": 0,
        "model": model or body.get("model", ""),
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": (
                "tool_calls" if done and (tool_calls or tool_calls_seen)
                else (body.get("done_reason") or "stop") if done else None
            ),
        }],
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
