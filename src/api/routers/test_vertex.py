import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi import Request
from starlette.datastructures import Headers, QueryParams

import api.routers.vertex as vertex

@pytest.fixture
def dummy_request():
    class DummyRequest:
        def __init__(self, headers=None, body=None, method="POST", query_params=None):
            self.headers = Headers(headers or {})
            self._body = body or b'{}'
            self.method = method
            self.query_params = QueryParams(query_params or {})

        async def body(self):
            return self._body

    return DummyRequest

def test_to_vertex_anthropic():
    openai_messages = {
        "messages": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
    }
    result = vertex.to_vertex_anthropic(openai_messages)
    assert result["anthropic_version"] == "vertex-2023-10-16"
    assert result["max_tokens"] == 256
    assert isinstance(result["messages"], list)
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"][0]["text"] == "Hello!"

def test_from_vertex_anthropic_to_openai():
    msg = json.dumps({
        "id": "abc123",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "stop",
        "usage": {"prompt_tokens": 5, "completion_tokens": 2}
    })
    result = json.loads(vertex.from_vertex_anthropic_to_openai(msg))
    assert result["id"] == "abc123"
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["content"] == "Hello!"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["prompt_tokens"] == 5

def test_to_openai_response():
    resp = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Hello!"}]},
                "finishReason": "STOP"
            }
        ]
    }
    result = vertex.to_openai_response(resp)
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["content"] == "Hello!"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["choices"][0]["index"] == 0
    assert result["id"].startswith("chatcmpl-")

def test_get_gcp_target_env(monkeypatch):
    monkeypatch.setenv("PROXY_TARGET", "https://custom-proxy")
    result = vertex.get_gcp_target("any-model", "/v1/chat/completions")
    assert result == "https://custom-proxy"

def test_get_gcp_target_known_chat(monkeypatch):
    monkeypatch.delenv("PROXY_TARGET", raising=False)
    model = vertex.known_chat_models[0]
    path = "/v1/chat/completions"
    result = vertex.get_gcp_target(model, path)
    assert "endpoints/openapi/chat/completions" in result

def test_get_gcp_target_raw_predict(monkeypatch):
    monkeypatch.delenv("PROXY_TARGET", raising=False)
    model = "unknown-model"
    path = "/v1/other"
    result = vertex.get_gcp_target(model, path)
    assert ":rawPredict" in result

@patch("api.routers.vertex.get_access_token", return_value="dummy-token")
def test_get_header_removes_hop_headers(mock_token, dummy_request):
    req = dummy_request(headers={
        "Host": "example.com",
        "Content-Length": "123",
        "Accept-Encoding": "gzip",
        "Connection": "keep-alive",
        "Authorization": "Bearer old",
        "X-Custom": "foo"
    })
    model = "test-model"
    path = "/v1/chat/completions"
    with patch("api.routers.vertex.get_gcp_target", return_value="http://target"):
        target_url, headers = vertex.get_header(model, req, path)
    assert target_url == "http://target"
    assert "Host" not in headers
    assert "Content-Length" not in headers
    assert "Accept-Encoding" not in headers
    assert "Connection" not in headers
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer dummy-token"
    assert headers["X-Custom"] == "foo"

@pytest.mark.asyncio
@patch("api.routers.vertex.httpx.AsyncClient")
@patch("api.routers.vertex.get_header")
@patch("api.routers.vertex.get_model", return_value="test-model")
async def test_handle_proxy_basic(mock_get_model, mock_get_header, mock_async_client, dummy_request):
    req = dummy_request(body=json.dumps({"model": "foo"}).encode())
    mock_get_header.return_value = ("http://target", {"Authorization": "Bearer token"})
    mock_response = MagicMock()
    mock_response.content = b'{"candidates":[{"content":{"parts":[{"text":"hi"}]}, "finishReason":"STOP"}]}'
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_async_client.return_value.__aenter__.return_value.request.return_value = mock_response

    vertex.USE_MODEL_MAPPING = True
    vertex.known_chat_models.append("test-model")
    result = await vertex.handle_proxy(req, "/v1/chat/completions")
    assert result.status_code == 200
    assert b"hi" in result.body
    assert result.headers["content-type"] == "application/json"
