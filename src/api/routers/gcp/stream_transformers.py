import json
import logging
import time
from typing import AsyncGenerator


def sse_chunk(payload: str) -> str:
    return f"data: {payload}\n\n"


def sse_done() -> str:
    return "data: [DONE]\n\n"


def generate_openai_id() -> str:
    return f"chatcmpl-ts{int(time.time() * 1000)}"


def transform_claude(data: dict):
    if data["type"] == "content_block_delta":
        return json.dumps(
            {
                "id": generate_openai_id(),
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": data["delta"]["text"]}, "index": 0, "finish_reason": None}],
            }
        )

    if data["type"] == "message_delta":
        return json.dumps({"choices": [{"delta": {}, "index": 0, "finish_reason": data["delta"]["stop_reason"]}]})

    if data["type"] == "message":
        return json.dumps(
            {
                "id": generate_openai_id(),
                "object": "chat.completion",
                "model": data["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": data["content"][0]["text"]},
                        "finish_reason": data.get("stop_reason", "stop"),
                    }
                ],
                "usage": {},
            }
        )

    logging.warning(f"Unknown data type: {data['type']}")


async def handle_data_line(raw_json: str, model: str) -> AsyncGenerator[str, None]:
    try:
        data = json.loads(raw_json)
        # override model
        if "model" in data and data["model"] is not None:
            data["model"] = model
    except json.JSONDecodeError:
        yield sse_chunk(raw_json)
        return

    if "type" in data:  # Claude
        if data.get("type") == "message_stop":
            yield sse_done()
        else:
            yield sse_chunk(transform_claude(data))
            if data.get("type") == "message_delta" and data["delta"].get("stop_reason"):
                yield sse_done()
    else:  # Gemini or OpenAI
        yield sse_chunk(json.dumps(data))
