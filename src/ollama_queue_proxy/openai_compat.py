"""OpenAI-compatible embedding endpoint translation layer.

Translates POST /v1/embeddings ↔ POST /api/embed so that clients using the
OpenAI Embeddings API (e.g. Graphiti with provider=openai) can route through
OQP without reconfiguration.

Request translation: /v1/embeddings body is identical to /api/embed — both
accept {"model": "...", "input": "..." | [...]}. Only the path is rewritten.

Response translation: Ollama /api/embed returns:
    {"embeddings": [[...], ...], "model": "bge-m3", ...}

OpenAI /v1/embeddings expects:
    {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [...], "index": 0}, ...],
        "model": "bge-m3",
        "usage": {"prompt_tokens": 0, "total_tokens": 0}
    }
"""

from __future__ import annotations

_OPENAI_COMPAT_PATHS: frozenset[str] = frozenset({"/v1/embeddings"})


def is_openai_compat_path(path: str) -> bool:
    """Return True if the path should be handled by the compat layer."""
    normalized = "/" + path.lstrip("/")
    return normalized in _OPENAI_COMPAT_PATHS


def rewrite_path(path: str) -> str:
    """Rewrite an OpenAI-compat path to the native Ollama equivalent."""
    normalized = "/" + path.lstrip("/")
    if normalized in _OPENAI_COMPAT_PATHS:
        return "api/embed"
    return path


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
