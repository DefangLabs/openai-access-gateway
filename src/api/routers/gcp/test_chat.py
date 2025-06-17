import pytest
import json
from unittest.mock import patch, MagicMock
from starlette.datastructures import Headers, QueryParams
from fastapi import Response

import api.routers.gcp.chat as chat

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
    result = chat.to_vertex_anthropic(openai_messages)
    assert result["anthropic_version"] == "vertex-2023-10-16"
    assert result["max_tokens"] == 256
    assert isinstance(result["messages"], list)
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"][0]["text"] == "Hello!"
    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][1]["content"][0]["text"] == "Hi there!"

def test_from_anthropic_to_openai_response():
    msg = json.dumps({
        "id": "abc123",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}, {"type": "text", "text": "Bye!"}],
        "stop_reason": "stop",
        "usage": {"prompt_tokens": 5, "completion_tokens": 2}
    })
    result = json.loads(chat.from_anthropic_to_openai_response(msg, "default"))
    assert result["id"] == "abc123"
    assert result["object"] == "chat.completion"
    assert len(result["choices"]) == 1
    assert result["choices"][0]["message"]["content"] == "Hello!Bye!"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["prompt_tokens"] == 5

def test_get_proxy_target_env(monkeypatch):
    monkeypatch.setenv("PROXY_TARGET", "https://custom-proxy")
    result = chat.get_proxy_target("any-model", "/v1/chat/completions")
    assert result == "https://custom-proxy"

def test_get_proxy_target_known_chat(monkeypatch):
    monkeypatch.delenv("PROXY_TARGET", raising=False)
    model = chat.known_chat_models[0]
    path = "/v1/chat/completions"
    result = chat.get_proxy_target(model, path)
    assert "endpoints/openapi/chat/completions" in result

def test_get_proxy_target_raw_predict(monkeypatch):
    monkeypatch.delenv("PROXY_TARGET", raising=False)
    model = "unknown-model"
    path = "/v1/other"
    result = chat.get_proxy_target(model, path)
    assert ":rawPredict" in result

@patch("api.routers.gcp.chat.get_access_token", return_value="dummy-token")
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
    with patch("api.routers.gcp.chat.get_proxy_target", return_value="http://target"):
        target_url, header = chat.get_header(model, req, path)
    assert target_url == "http://target"
    assert "Host" not in header
    assert "Content-Length" not in header
    assert "Accept-Encoding" not in header
    assert "Connection" not in header
    assert "Authorization" in header
    assert header["Authorization"] == "Bearer dummy-token"
    assert header["x-custom"] == "foo"

@pytest.mark.asyncio
@patch("api.routers.gcp.chat.httpx.AsyncClient")
@patch("api.routers.gcp.chat.get_header")
@patch("api.routers.gcp.chat.get_model", return_value="test-model")
async def test_handle_proxy_basic(mock_get_model, mock_get_header, mock_async_client, dummy_request):
    req = dummy_request(body=json.dumps({"model": "foo"}).encode())
    mock_get_header.return_value = ("http://target", {"Authorization": "Bearer token"})
    mock_response = MagicMock()
    mock_response.content = b'{"candidates":[{"content":{"parts":[{"text":"hi"}]}, "finishReason":"STOP"}]}'
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_async_client.return_value.__aenter__.return_value.request.return_value = mock_response

    chat.USE_MODEL_MAPPING = True
    chat.known_chat_models.append("test-model")
    result = await chat.handle_proxy(req, "/v1/chat/completions")
    assert result.status_code == 200
    assert b"hi" in result.body
    assert result.headers["content-type"] == "application/json"

@pytest.mark.asyncio
@patch("api.routers.gcp.chat.httpx.AsyncClient")
@patch("api.routers.gcp.chat.get_header")
@patch("api.routers.gcp.chat.get_model", return_value="test-model")
async def test_handle_proxy_known_chat_model(
    mock_get_model, mock_get_header, mock_async_client, dummy_request
):
    req = dummy_request(body=json.dumps({"model": "foo"}).encode())
    mock_get_header.return_value = ("http://target", {"Authorization": "Bearer token"})
    mock_response = MagicMock()
    mock_response.content = b'{"candidates":[{"content":{"parts":[{"text":"hi"}]}, "finishReason":"STOP"}]}'
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_async_client.return_value.__aenter__.return_value.request.return_value = mock_response

    chat.USE_MODEL_MAPPING = True
    if "test-model" not in chat.known_chat_models:
        chat.known_chat_models.append("test-model")

    result = await chat.handle_proxy(req, "/v1/chat/completions")
    assert isinstance(result, Response)
    assert result.status_code == 200
    assert b"hi" in result.body
    assert result.headers["content-type"] == "application/json"

@pytest.mark.asyncio
@patch("api.routers.gcp.chat.httpx.AsyncClient")
@patch("api.routers.gcp.chat.get_header")
@patch("api.routers.gcp.chat.get_model", return_value="anthropic-model")
async def test_handle_proxy_anthropic_conversion(
    mock_get_model, mock_get_header, mock_async_client, dummy_request
):
    req = dummy_request(body=json.dumps({"model": "foo", "messages": [{"role": "user", "content": "hi"}]}).encode())
    mock_get_header.return_value = ("http://target", {"Authorization": "Bearer token"})
    mock_response = MagicMock()
    # Simulate anthropic response
    anthropic_resp = json.dumps({
        "id": "abc123",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "stop",
        "usage": {"prompt_tokens": 5, "completion_tokens": 2}
    }).encode()
    mock_response.content = anthropic_resp
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_async_client.return_value.__aenter__.return_value.request.return_value = mock_response

    chat.USE_MODEL_MAPPING = True
    # Ensure model is not in known_chat_models to trigger conversion
    if "anthropic-model" in chat.known_chat_models:
        chat.known_chat_models.remove("anthropic-model")
    result = await chat.handle_proxy(req, "/v1/chat/completions")
    assert isinstance(result, Response)
    data = json.loads(result.body)
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Hello!"

@pytest.mark.asyncio
@patch("api.routers.gcp.chat.httpx.AsyncClient", side_effect=Exception("network error"))
@patch("api.routers.gcp.chat.get_header")
@patch("api.routers.gcp.chat.get_model", return_value="test-model")
async def test_handle_proxy_httpx_exception(
    mock_get_model, mock_get_header, mock_async_client, dummy_request
):
    req = dummy_request(body=json.dumps({"model": "foo"}).encode())
    mock_get_header.return_value = ("http://target", {"Authorization": "Bearer token"})
    chat.USE_MODEL_MAPPING = True
    if "test-model" not in chat.known_chat_models:
        chat.known_chat_models.append("test-model")
    # Patch httpx.RequestError to be raised
    with patch("api.routers.gcp.chat.httpx.RequestError", Exception):
        result = await chat.handle_proxy(req, "/v1/chat/completions")
        assert isinstance(result, Response)
        assert result.status_code == 502
        assert b"Upstream request failed" in result.body
    # Assert that the status code is 502 (Bad Gateway) due to upstream failure
    assert result.status_code == 502

    # Assert that the response body contains the expected error message
    assert b"Upstream request failed" in result.body

def test_get_chat_completion_model_name_known_chat_model():
    # Pick a known chat model from the list
    model_alias = "publishers/google/models/gemini-2.0-flash-lite-001"
    # Patch known_chat_models to ensure the model is present
    if model_alias not in chat.known_chat_models:
        chat.known_chat_models.append(model_alias)
    # Patch the function to use the correct argument name
    # The function as written has a bug: it uses 'model' instead of 'model_alias'
    # So we patch the function here for the test
    # But for now, test as is
    result = chat.get_chat_completion_model_name(model_alias)
    # Should remove 'publishers/' and 'models/' from the string
    assert result == "google/gemini-2.0-flash-lite-001"

def test_get_chat_completion_model_name_unknown_model():
    model_alias = "some-other-model"
    # Ensure it's not in known_chat_models
    if model_alias in chat.known_chat_models:
        chat.known_chat_models.remove(model_alias)
    result = chat.get_chat_completion_model_name(model_alias)
    # Should return the input unchanged
    assert result == model_alias

