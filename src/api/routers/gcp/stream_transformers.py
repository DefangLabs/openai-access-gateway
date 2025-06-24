import json
import time
from typing import AsyncGenerator

def openai_chunk(payload: str) -> str:
    return f"data: {payload}\n\n"

def openai_done() -> str:
    return "data: [DONE]\n\n"

def generate_openai_id() -> str:
    return f"chatcmpl-ts{int(time.time() * 1000)}"

async def transform_claude(data: dict) -> AsyncGenerator[str, None]:
    if data["type"] == "content_block_delta":
        yield openai_chunk(json.dumps({
            "id": generate_openai_id(),
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "delta": { "content": data["delta"]["text"] },
                    "index": 0,
                    "finish_reason": None
                }
            ]
        }))

    elif data["type"] == "message_delta" and "stop_reason" in data["delta"]:
        yield openai_chunk(json.dumps({
            "choices": [
                {
                    "delta": {},
                    "index": 0,
                    "finish_reason": data["delta"]["stop_reason"]
                }
            ]
        }))
        yield openai_done()

    elif data["type"] == "message_stop":
        yield openai_done()

async def handle_data_line(raw_json: str, model: str) -> AsyncGenerator[str, None]:
    try:
        data = json.loads(raw_json)
        # override model
        if data["model"] != None:
            data["model"] = model
    except json.JSONDecodeError:
        yield openai_chunk(raw_json)
        return

    if "type" in data:  # Claude
        async for chunk in transform_claude(data):
            yield chunk
    else:  # Gemini or OpenAI
        yield openai_chunk(json.dumps(data))
