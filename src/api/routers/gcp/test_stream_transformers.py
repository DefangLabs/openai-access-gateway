import asyncio
import json
import re

import pytest
from stream_transformers import (
    generate_openai_id,
    handle_data_line,
    sse_chunk,
    sse_done,
)


def test_sse_chunk():
    payload = '{"foo": "bar"}'
    assert sse_chunk(payload) == f"data: {payload}\n\n"


def test_generate_openai_id():
    id1 = generate_openai_id()
    assert id1.startswith("chatcmpl-ts")
    assert re.match(r"chatcmpl-ts\d{13}", id1)


@pytest.mark.asyncio
async def test_handle_data_line_content_block_delta():
    data = {
        "type": "content_block_delta",
        "delta": {"text": "Hello!"},
    }
    gen = handle_data_line(json.dumps(data), "test-model")
    chunk = await anext(gen)
    obj = json.loads(chunk[len("data: ") : -2])
    assert obj["object"] == "chat.completion.chunk"
    assert obj["choices"][0]["delta"]["content"] == "Hello!"
    with pytest.raises(StopAsyncIteration):
        await anext(gen)


@pytest.mark.asyncio
async def test_handle_data_line_message_delta_with_stop_reason():
    data = {
        "type": "message_delta",
        "delta": {"stop_reason": "stop"},
    }
    gen = handle_data_line(json.dumps(data), "test-model")

    # chunk 1
    chunk = await anext(gen)
    obj = json.loads(chunk[len("data: ") : -2])
    assert obj["choices"][0]["finish_reason"] == "stop"

    # chunk 2
    chunk = await anext(gen)
    assert chunk == sse_done()
    with pytest.raises(StopAsyncIteration):
        await anext(gen)


@pytest.mark.asyncio
async def test_handle_data_line_message_stop():
    data = {"type": "message_stop"}
    gen = handle_data_line(json.dumps(data), "test-model")
    chunk = await anext(gen)
    assert chunk == sse_done()
    with pytest.raises(StopAsyncIteration):
        await anext(gen)


@pytest.mark.asyncio
async def test_handle_data_line_claude_content_block_delta():
    data = {"type": "content_block_delta", "delta": {"text": "Hi!"}, "model": "claude-2"}
    raw_json = json.dumps(data)
    gen = handle_data_line(raw_json, "override-model")
    chunk = await anext(gen)
    obj = json.loads(chunk[len("data: ") : -2])
    assert obj["choices"][0]["delta"]["content"] == "Hi!"
    assert "model" not in obj  # model should not be in delta blocks
    with pytest.raises(StopAsyncIteration):
        await anext(gen)


@pytest.mark.asyncio
async def test_handle_data_line_gemini_or_openai():
    data = {
        "id": "abc",
        "object": "chat.completion",
        "model": "gemini-2.0",
    }
    raw_json = json.dumps(data)
    gen = handle_data_line(raw_json, "override-model")
    chunk = await anext(gen)
    obj = json.loads(chunk[len("data: ") : -2])
    assert obj["id"] == "abc"
    assert obj["model"] == "override-model"
    with pytest.raises(StopAsyncIteration):
        await anext(gen)


@pytest.mark.asyncio
async def test_handle_data_line_invalid_json():
    raw_json = "not a json"
    gen = handle_data_line(raw_json, "any-model")
    chunk = await anext(gen)
    assert chunk == sse_chunk(raw_json)
    with pytest.raises(StopAsyncIteration):
        await anext(gen)
