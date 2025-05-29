import sys
import pytest
import json
import types
from fastapi import Request, Response
from starlette.datastructures import Headers, QueryParams

import api.routers

import api.routers.vertex as vertex

# Patch the module-level variables for testing
@pytest.fixture(autouse=True)
def reload_app_module(monkeypatch):
    # Remove app from sys.modules to force reload
    if "api.app" in sys.modules:
        del sys.modules["api.app"]
    yield

def test_get_gcp_target_with_proxy_target(monkeypatch):
    # Set the environment variable
    monkeypatch.setenv("PROXY_TARGET", "https://custom-proxy.example.com")
    # Patch location and project_id in the module
    # Call the function
    assert vertex.get_gcp_target("/some/path") == "https://custom-proxy.example.com"

def test_get_gcp_target_without_proxy_target(monkeypatch):
    # Unset the environment variable
    monkeypatch.delenv("PROXY_TARGET", raising=False)
    # Patch location and project_id in the module
    monkeypatch.setattr(vertex, "location", "us-central1")
    monkeypatch.setattr(vertex, "project_id", "test-project")
    # Call the function
    result = vertex.get_gcp_target("/some/path")
    expected = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/some/path"
    assert result == expected

def test_get_gcp_target_path_with_leading_slash(monkeypatch):
    monkeypatch.delenv("PROXY_TARGET", raising=False)
    monkeypatch.setattr(vertex, "location", "europe-west4")
    monkeypatch.setattr(vertex, "project_id", "another-project")
    result = vertex.get_gcp_target("////foo/bar")
    expected = "https://europe-west4-aiplatform.googleapis.com/v1/projects/another-project/locations/europe-west4/foo/bar"
    assert result == expected

def test_get_gcp_target_path_without_leading_slash(monkeypatch):
    monkeypatch.delenv("PROXY_TARGET", raising=False)
    monkeypatch.setattr(vertex, "location", "asia-east1")
    monkeypatch.setattr(vertex, "project_id", "proj-123")
    result = vertex.get_gcp_target("baz/qux")
    expected = "https://asia-east1-aiplatform.googleapis.com/v1/projects/proj-123/locations/asia-east1/baz/qux"
    assert result == expected

def test_get_gcp_target_trailing_slash(monkeypatch):
    monkeypatch.delenv("PROXY_TARGET", raising=False)
    monkeypatch.setattr(vertex, "location", "us-west2")
    monkeypatch.setattr(vertex, "project_id", "proj-x")
    result = vertex.get_gcp_target("/foo/bar/")
    expected = "https://us-west2-aiplatform.googleapis.com/v1/projects/proj-x/locations/us-west2/foo/bar"
    assert result == expected

class DummyRequest:
    def __init__(self, headers):
        self.headers = headers

@pytest.fixture(autouse=True)
def patch_get_access_token(monkeypatch):
    monkeypatch.setattr(vertex, "get_access_token", lambda: "dummy-token")

@pytest.mark.parametrize(
    "path,expected_path",
    [
        ("chat/completions", "endpoints/openapi/chat/completions"),
        ("/chat/completions", "endpoints/openapi/chat/completions"),
        ("/chat/completions/remainder", "endpoints/openapi/chat/completions/remainder"),
        ("foo/bar", "foo/bar"),
        ("/foo/bar", "foo/bar"),
    ]
)
def test_get_header_path_replacement(monkeypatch, path, expected_path):
    import api.app
    # Patch API_ROUTE_PREFIX and get_gcp_target
    monkeypatch.setattr(api.app, "API_ROUTE_PREFIX", "/api")
    called = {}
    def fake_get_gcp_target(p):
        called["path"] = p
        return "https://target"
    monkeypatch.setattr(vertex, "get_gcp_target", fake_get_gcp_target)
    req = DummyRequest(headers={})
    vertex.get_header(req, path)
    assert called["path"] == f"/{expected_path}"

def test_get_header_removes_hop_by_hop_headers(monkeypatch):
    monkeypatch.setattr(vertex, "API_ROUTE_PREFIX", "/api")
    monkeypatch.setattr(vertex, "get_gcp_target", lambda p: "https://target")
    monkeypatch.setattr(vertex, "get_access_token", lambda: "dummy-token")
    req = DummyRequest(headers={
        "Host": "example.com",
        "Content-Length": "123",
        "Accept-Encoding": "gzip",
        "Connection": "keep-alive",
        "Authorization": "Bearer old",
        "X-Custom": "foo"
    })
    url, headers = vertex.get_header(req, "/foo/bar")
    assert url == "https://target"
    assert "Host" not in headers
    assert "Content-Length" not in headers
    assert "Accept-Encoding" not in headers
    assert "Connection" not in headers
    assert "Authorization" in headers  # Should be re-added as Bearer dummy-token
    assert headers["Authorization"] == "Bearer dummy-token"
    assert headers["X-Custom"] == "foo"

def test_get_header_preserves_other_headers(monkeypatch):
    import api.app
    monkeypatch.setattr(api.app, "API_ROUTE_PREFIX", "/api")
    monkeypatch.setattr(vertex, "get_gcp_target", lambda p: "https://target")
    monkeypatch.setattr(vertex, "get_access_token", lambda: "dummy-token")
    req = DummyRequest(headers={
        "X-Test": "abc",
        "User-Agent": "pytest"
    })
    url, headers = vertex.get_header(req, "foo/bar")
    assert headers["X-Test"] == "abc"
    assert headers["User-Agent"] == "pytest"
    assert headers["Authorization"] == "Bearer dummy-token"

def test_to_vertex_contents_basic():
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    result = vertex.to_vertex_contents(messages)
    assert result == [
        {"role": "user", "parts": [{"text": "Hello!"}]},
        {"role": "assistant", "parts": [{"text": "Hi there!"}]},
    ]

def test_to_vertex_contents_system_message():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather?"},
    ]
    result = vertex.to_vertex_contents(messages)
    assert result == [
        {"role": "user", "parts": [{"text": "[System instruction]: You are a helpful assistant."}]},
        {"role": "user", "parts": [{"text": "What's the weather?"}]},
    ]

def test_to_vertex_contents_skips_empty_content():
    messages = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": None},
        {"role": "user", "content": "Not empty"},
    ]
    result = vertex.to_vertex_contents(messages)
    assert result == [
        {"role": "user", "parts": [{"text": "Not empty"}]},
    ]

def test_to_vertex_contents_mixed_roles():
    messages = [
        {"role": "system", "content": "System context"},
        {"role": "user", "content": "User says hi"},
        {"role": "assistant", "content": "Assistant replies"},
        {"role": "system", "content": "Another system message"},
    ]
    result = vertex.to_vertex_contents(messages)
    assert result == [
        {"role": "user", "parts": [{"text": "[System instruction]: System context"}]},
        {"role": "user", "parts": [{"text": "User says hi"}]},
        {"role": "assistant", "parts": [{"text": "Assistant replies"}]},
        {"role": "user", "parts": [{"text": "[System instruction]: Another system message"}]},
    ]

def test_to_vertex_contents_empty_list():
    assert vertex.to_vertex_contents([]) == []

def test_to_openai_response_basic():
    resp = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Hello, "},
                        {"text": "world!"}
                    ]
                },
                "finishReason": "STOP"
            }
        ]
    }
    result = vertex.to_openai_response(resp)
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["index"] == 0
    assert result["choices"][0]["message"]["role"] == "assistant"
    assert result["choices"][0]["message"]["content"] == "Hello, world!"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["id"].startswith("chatcmpl-")
    assert len(result["id"]) == len("chatcmpl-") + 8

def test_to_openai_response_multiple_candidates():
    resp = {
        "candidates": [
            {
                "content": {"parts": [{"text": "First"}]},
                "finishReason": "STOP"
            },
            {
                "content": {"parts": [{"text": "Second"}]},
                "finishReason": "LENGTH"
            }
        ]
    }
    result = vertex.to_openai_response(resp)
    assert len(result["choices"]) == 2
    assert result["choices"][0]["message"]["content"] == "First"
    assert result["choices"][1]["message"]["content"] == "Second"
    assert result["choices"][1]["finish_reason"] == "length"

def test_to_openai_response_empty_candidates():
    resp = {"candidates": []}
    with pytest.raises(ValueError, match="No candidates in response"):
        vertex.to_openai_response(resp)

def test_to_openai_response_no_candidates_key():
    resp = {}
    with pytest.raises(ValueError, match="No candidates in response"):
        vertex.to_openai_response(resp)

def test_to_openai_response_missing_content_parts():
    resp = {
        "candidates": [
            {
                "content": {},
                "finishReason": "STOP"
            }
        ]
    }
    result = vertex.to_openai_response(resp)
    assert result["choices"][0]["message"]["content"] == ""

def test_to_openai_response_missing_finish_reason():
    resp = {
        "candidates": [
            {
                "content": {"parts": [{"text": "No finish reason"}]}
                # finishReason missing
            }
        ]
    }
    result = vertex.to_openai_response(resp)
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["choices"][0]["message"]["content"] == "No finish reason"

def test_to_openai_response_part_missing_text():
    resp = {
        "candidates": [
            {
                "content": {"parts": [{}]},
                "finishReason": "STOP"
            }
        ]
    }
    result = vertex.to_openai_response(resp)
    assert result["choices"][0]["message"]["content"] == ""

class DummyRequestObj:
    def __init__(self, body_data, method="POST", headers=None, query_params=None):
        self._body = body_data
        self.method = method
        self.headers = headers or {}
        self.query_params = query_params or {}

    async def body(self):
        return self._body

@pytest.mark.asyncio
async def test_handle_proxy_success(monkeypatch):
    # Patch get_header to return dummy url and headers
    monkeypatch.setattr(vertex, "get_header", lambda req, path: ("https://dummy", {"Authorization": "Bearer x"}))
    # Patch USE_MODEL_MAPPING to False
    monkeypatch.setattr(vertex, "USE_MODEL_MAPPING", False)
    # Patch to_vertex_contents to just return the input
    monkeypatch.setattr(vertex, "to_vertex_contents", lambda x: x)
    # Patch httpx.AsyncClient to return a dummy response
    class DummyResponse:
        def __init__(self):
            self.content = b"ok"
            self.status_code = 200
            self.headers = {"content-type": "application/json"}
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}]}
    class DummyAsyncClient:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass
        async def request(self, **kwargs): return DummyResponse()
    monkeypatch.setattr(vertex.httpx, "AsyncClient", lambda: DummyAsyncClient())
    req = DummyRequestObj(b'{"foo": "bar"}', headers={"x": "y"}, query_params={"a": "b"})
    resp = await vertex.handle_proxy(req, "/foo/bar")
    assert isinstance(resp, Response)
    assert resp.status_code == 200
    assert resp.content == b"ok"
    assert resp.headers["content-type"] == "application/json"

@pytest.mark.asyncio
async def test_handle_proxy_with_model_mapping(monkeypatch):
    # Patch get_header
    monkeypatch.setattr(vertex, "get_header", lambda req, path: ("https://dummy", {}))
    # Patch USE_MODEL_MAPPING to True
    monkeypatch.setattr(vertex, "USE_MODEL_MAPPING", True)
    # Patch get_model to return a mapped model
    monkeypatch.setattr(vertex, "get_model", lambda provider, model: "publishers/google/foo")
    # Patch to_vertex_contents to just return the input
    monkeypatch.setattr(vertex, "to_vertex_contents", lambda x: x)
    # Patch httpx.AsyncClient
    class DummyResponse:
        def __init__(self):
            self.content = b"mapped"
            self.status_code = 201
            self.headers = {"content-type": "application/json"}
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "mapped"}]}, "finishReason": "STOP"}]}
    class DummyAsyncClient:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass
        async def request(self, **kwargs): return DummyResponse()
    monkeypatch.setattr(vertex.httpx, "AsyncClient", lambda: DummyAsyncClient())
    # The model will be remapped to google/foo
    req = DummyRequestObj(json.dumps({"model": "bar"}).encode(), headers={}, query_params={})
    resp = await vertex.handle_proxy(req, "/foo/bar")
    assert resp.status_code == 201
    assert resp.content == b"mapped"

@pytest.mark.asyncio
async def test_handle_proxy_with_messages_conversion(monkeypatch):
    # Patch get_header
    monkeypatch.setattr(vertex, "get_header", lambda req, path: ("https://dummy", {}))
    monkeypatch.setattr(vertex, "USE_MODEL_MAPPING", False)
    # Patch to_vertex_contents to check input
    called = {}
    def fake_to_vertex_contents(x):
        called["input"] = x
        return b"converted"
    monkeypatch.setattr(vertex, "to_vertex_contents", fake_to_vertex_contents)
    # Patch httpx.AsyncClient
    class DummyResponse:
        def __init__(self):
            self.content = b"converted"
            self.status_code = 200
            self.headers = {"content-type": "application/json"}
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "converted"}]}, "finishReason": "STOP"}]}
    class DummyAsyncClient:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass
        async def request(self, **kwargs): return DummyResponse()
    monkeypatch.setattr(vertex.httpx, "AsyncClient", lambda: DummyAsyncClient())
    # The content contains "messages"
    req = DummyRequestObj(b'{"messages": [{"role": "user", "content": "hi"}]}')
    resp = await vertex.handle_proxy(req, "/foo/bar")
    assert resp.status_code == 200
    assert resp.content == b"converted"
    assert "input" in called

@pytest.mark.asyncio
async def test_handle_proxy_upstream_error(monkeypatch):
    # Patch get_header
    monkeypatch.setattr(vertex, "get_header", lambda req, path: ("https://dummy", {}))
    monkeypatch.setattr(vertex, "USE_MODEL_MAPPING", False)
    # Patch httpx.AsyncClient to raise RequestError
    class DummyAsyncClient:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass
        async def request(self, **kwargs):
            raise vertex.httpx.RequestError("fail", request=None)
    monkeypatch.setattr(vertex.httpx, "AsyncClient", lambda: DummyAsyncClient())
    req = DummyRequestObj(b"{}", headers={}, query_params={})
    resp = await vertex.handle_proxy(req, "/foo/bar")
    assert isinstance(resp, Response)
    assert resp.status_code == 502
    assert b"Upstream request failed" in resp.body

@pytest.mark.asyncio
async def test_handle_proxy_needs_conversion(monkeypatch):
    # Patch get_header
    monkeypatch.setattr(vertex, "get_header", lambda req, path: ("https://dummy", {}))
    monkeypatch.setattr(vertex, "USE_MODEL_MAPPING", False)
    # Patch to_vertex_contents to just return the input
    monkeypatch.setattr(vertex, "to_vertex_contents", lambda x: x)
    # Patch httpx.AsyncClient to return a dummy response with .json()
    class DummyResponse:
        def __init__(self):
            self.content = b"dummy"
            self.status_code = 200
            self.headers = {"content-type": "application/json"}
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "foo"}]}, "finishReason": "STOP"}]}
    class DummyAsyncClient:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass
        async def request(self, **kwargs): return DummyResponse()
    monkeypatch.setattr(vertex.httpx, "AsyncClient", lambda: DummyAsyncClient())
    # Patch to_openai_response to check input
    called = {}
    def fake_to_openai_response(resp):
        called["resp"] = resp
        return {"id": "chatcmpl-12345678", "object": "chat.completion", "choices": []}
    monkeypatch.setattr(vertex, "to_openai_response", fake_to_openai_response)
    req = DummyRequestObj(b'{"messages": ["foo"]}')
    await vertex.handle_proxy(req, "/foo/bar")
    assert "resp" in called



