"""Tests for the OpenAI-compat embedding translation layer (openai_compat.py)."""

from ollama_queue_proxy.openai_compat import (
    is_openai_compat_path,
    rewrite_path,
    translate_chat_request,
    wrap_chat_chunk,
    wrap_chat_response,
    wrap_error,
    wrap_response,
)

# ── path detection ────────────────────────────────────────────────────────────


def test_v1_embeddings_is_compat_path():
    assert is_openai_compat_path("/v1/embeddings") is True


def test_v1_chat_completions_is_compat_path():
    assert is_openai_compat_path("/v1/chat/completions") is True


def test_rewrite_chat_to_api_chat():
    assert rewrite_path("/v1/chat/completions") == "/api/chat"


def test_translate_chat_preserves_model_messages_and_maps_options():
    result = translate_chat_request({
        "model": "alias-model", "messages": [{"role": "user", "content": "hi"}],
        "stream": True, "max_tokens": 42, "temperature": 0.2,
    })
    assert result["model"] == "alias-model"
    assert result["messages"][0]["content"] == "hi"
    assert result["options"] == {"num_predict": 42, "temperature": 0.2}
    assert result["stream"] is True


def test_translate_chat_maps_reasoning_effort_to_think():
    assert translate_chat_request({"reasoning_effort": "off"})["think"] is False
    assert translate_chat_request({"reasoning_effort": "high"})["think"] is True
    assert translate_chat_request({"reasoning_effort": "off", "think": True})["think"] is True


def test_translate_chat_reserves_answer_budget_for_reasoning():
    result = translate_chat_request({
        "think": True, "max_tokens": 64,
    })
    assert result["options"]["num_predict"] == 256


def test_translate_chat_preserves_large_reasoning_budget():
    result = translate_chat_request({
        "think": True, "max_tokens": 1024,
    })
    assert result["options"]["num_predict"] == 1024


def test_translate_chat_does_not_raise_non_reasoning_budget():
    result = translate_chat_request({
        "think": False, "max_tokens": 64,
    })
    assert result["options"]["num_predict"] == 64


def test_translate_chat_pi_content_parts_for_all_message_roles():
    result = translate_chat_request({
        "model": "qwen3",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "System rules"}]},
            {"role": "developer", "content": [{"type": "text", "text": "Be concise"}]},
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,abc123",
                }},
            ]},
        ],
    })

    assert result["messages"] == [
        {"role": "system", "content": "System rules"},
        {"role": "developer", "content": "Be concise"},
        {
            "role": "user",
            "content": "Describe this image\n[image_url: data:image/png;base64,abc123]",
        },
    ]


def test_translate_chat_preserves_string_content_and_unknown_parts():
    string_content = "already an Ollama-compatible string"
    result = translate_chat_request({
        "messages": [
            {"role": "user", "content": string_content},
            {"role": "user", "content": [
                {"type": "audio", "audio": {"format": "wav", "data": "abc"}},
                {"type": "text", "text": "kept"},
            ]},
        ],
    })

    assert result["messages"][0]["content"] == string_content
    assert result["messages"][1]["content"] == (
        '[unsupported_content_part: {"audio":{"data":"abc","format":"wav"},"type":"audio"}]\n'
        "kept"
    )


def test_translate_chat_converts_openai_tool_history_to_ollama():
    result = translate_chat_request({
        "messages": [
            {"role": "user", "content": "Read the file."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_read",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"/tmp/example.txt"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_read",
                "content": '{"contents":"example"}',
            },
        ],
    })

    assert result["messages"] == [
        {"role": "user", "content": "Read the file."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "/tmp/example.txt"},
                },
            }],
        },
        {"role": "tool", "content": '{"contents":"example"}'},
    ]


def test_wrap_chat_response_openai_schema_and_usage():
    result = wrap_chat_response({
        "model": "llama3", "message": {"role": "assistant", "content": "hello"},
        "done": True, "prompt_eval_count": 3, "eval_count": 5,
    })
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["content"] == "hello"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"] == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}


def test_wrap_chat_thinking_as_openai_reasoning():
    response = wrap_chat_response({
        "model": "qwen3.8:27b",
        "message": {"role": "assistant", "content": "answer", "thinking": "reason"},
    })
    assert response["choices"][0]["message"]["reasoning"] == "reason"

    chunk = wrap_chat_chunk({"model": "qwen3.8:27b", "message": {"thinking": "reason"}})
    assert chunk["choices"][0]["delta"]["reasoning"] == "reason"


def test_wrap_chat_response_translates_native_tool_calls():
    response = wrap_chat_response({
        "model": "qwen3:8b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_read",
                "function": {
                    "index": 0,
                    "name": "read_file",
                    "arguments": {"path": "/tmp/example.txt"},
                },
            }],
        },
    })

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"] == [{
        "id": "call_read",
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": '{"path":"/tmp/example.txt"}',
        },
    }]


def test_wrap_chat_chunk_translates_tool_calls_and_preserves_terminal_reason():
    tool_chunk = wrap_chat_chunk({
        "model": "qwen3:8b",
        "message": {
            "tool_calls": [{
                "id": "call_read",
                "function": {
                    "index": 3,
                    "name": "read_file",
                    "arguments": {"path": "/tmp/example.txt"},
                },
            }],
        },
        "done": False,
    })
    assert tool_chunk["choices"][0]["delta"]["tool_calls"] == [{
        "id": "call_read",
        "index": 3,
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": '{"path":"/tmp/example.txt"}',
        },
    }]
    assert tool_chunk["choices"][0]["finish_reason"] is None

    terminal_chunk = wrap_chat_chunk(
        {"message": {}, "done": True, "done_reason": "stop"}, tool_calls_seen=True
    )
    assert terminal_chunk["choices"][0]["finish_reason"] == "tool_calls"


def test_wrap_chat_chunk_and_error():
    chunk = wrap_chat_chunk({"model": "llama3", "message": {"content": "hi"}})
    assert chunk["object"] == "chat.completion.chunk"
    assert chunk["choices"][0]["delta"]["content"] == "hi"
    assert wrap_error({"error": "model not found"})["error"]["message"] == "model not found"


def test_v1_embeddings_without_leading_slash():
    assert is_openai_compat_path("v1/embeddings") is True


def test_api_embed_is_not_compat_path():
    assert is_openai_compat_path("/api/embed") is False


def test_api_generate_is_not_compat_path():
    assert is_openai_compat_path("/api/generate") is False


# ── path rewrite ──────────────────────────────────────────────────────────────


def test_rewrite_v1_embeddings_to_api_embed():
    assert rewrite_path("/v1/embeddings") == "/api/embed"


def test_rewrite_non_compat_path_unchanged():
    assert rewrite_path("/api/generate") == "/api/generate"


def test_rewrite_without_leading_slash():
    assert rewrite_path("v1/embeddings") == "/api/embed"


# ── response wrapping ─────────────────────────────────────────────────────────


def test_wrap_response_single_embedding():
    ollama = {
        "embeddings": [[0.1, 0.2, 0.3]],
        "model": "bge-m3",
    }
    result = wrap_response(ollama)

    assert result["object"] == "list"
    assert result["model"] == "bge-m3"
    assert len(result["data"]) == 1
    assert result["data"][0]["object"] == "embedding"
    assert result["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert result["data"][0]["index"] == 0
    assert result["usage"]["prompt_tokens"] == 0
    assert result["usage"]["total_tokens"] == 0


def test_wrap_response_batch_embeddings():
    vectors = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    ollama = {"embeddings": vectors, "model": "bge-m3"}
    result = wrap_response(ollama)

    assert len(result["data"]) == 3
    for i, item in enumerate(result["data"]):
        assert item["index"] == i
        assert item["embedding"] == vectors[i]


def test_wrap_response_model_override():
    ollama = {"embeddings": [[0.1]], "model": "bge-m3"}
    result = wrap_response(ollama, model="nomic-embed-text")
    assert result["model"] == "nomic-embed-text"


def test_wrap_response_empty_embeddings():
    ollama = {"embeddings": [], "model": "bge-m3"}
    result = wrap_response(ollama)
    assert result["data"] == []
    assert result["object"] == "list"


def test_wrap_response_missing_embeddings_key():
    ollama = {"model": "bge-m3"}
    result = wrap_response(ollama)
    assert result["data"] == []


def test_wrap_response_missing_model_falls_back_to_empty_string():
    ollama = {"embeddings": [[0.1]]}
    result = wrap_response(ollama)
    assert result["model"] == ""


# ── test_env_override_list_index ──────────────────────────────────────────────


def test_env_override_list_index(monkeypatch):
    """OQP_OLLAMA__HOSTS__0__URL overrides a configured list item."""
    monkeypatch.setenv("OQP_OLLAMA__HOSTS__0__URL", "http://tampered:11434")

    from ollama_queue_proxy.config import _apply_env_overrides

    data = {"ollama": {"hosts": [{"url": "http://original:11434", "name": "primary"}]}}
    result = _apply_env_overrides(data)

    assert result["ollama"]["hosts"][0]["url"] == "http://tampered:11434"
