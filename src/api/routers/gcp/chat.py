import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse
from google.auth import default
from google.auth.transport.requests import Request as AuthRequest

from api.auth import api_key_auth
from api.gcp.credentials.metadata import get_access_token, location, project_id
from api.modelmapper import get_model
from api.routers.gcp.common import get_headers_and_target, sse_done, to_openai_usage
from api.routers.gcp.stream_transformers import handle_data_line
from api.schema import ChatResponse, ChatStreamResponse, Error
from api.setting import API_ROUTE_PREFIX, USE_MODEL_MAPPING

known_chat_models = [
    "publishers/mistral-ai/models/mistral-7b-instruct-v0.3",
    "publishers/mistral-ai/models/mistral-nemo-instruct-2407",
    "publishers/mistral-ai/models/mistral-nemo@2407",
    "publishers/mistral-ai/models/mistral-7b-instruct@v0.3",
    "publishers/google/models/gemma-2-27b-it",
    "publishers/google/models/gemma-2-9b-it",
    "publishers/google/models/gemma-2b",
    "publishers/google/models/gemini-2.0-flash-001",
    "publishers/google/models/gemini-2.0-flash-lite-001",
    "publishers/google/models/gemini-2.5-pro-preview-05-06",
    "publishers/google/models/gemini-2.5-flash-preview-05-20",
    "publishers/meta/models/llama3-8b",
    "publishers/meta/models/llama-3-1-8b-instruct",
    "publishers/meta/models/llama2-7b",
]

router = APIRouter(
    prefix="/chat",
    dependencies=[Depends(api_key_auth)],
    responses={404: {"description": "Not found"}},
)


def _parse_system_prompts(openai_messages) -> str:
    system_prompts = ""
    for message in openai_messages:
        if message["role"] != "system":
            # ignore system messages here
            continue
        assert isinstance(message["content"], str)
        system_prompts += message["content"] + "\n"

        return system_prompts


def to_vertex_anthropic(openai_messages):
    message = []
    for m in openai_messages["messages"]:
        if m["role"] == "system":
            continue
        if isinstance(m["content"], str):
            message.append({"role": m["role"], "content": {"type": "text", "text": m["content"]}})
        else:
            message.append({"role": m["role"], "content": m["content"]})

    system_prompts = _parse_system_prompts(openai_messages["messages"])

    return {"anthropic_version": "vertex-2023-10-16", "max_tokens": 256, "system": system_prompts, "messages": message}


def from_anthropic_to_openai_response(msg, model):
    msg_json = json.loads(msg)
    response = {
        "id": msg_json["id"],
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": msg_json["role"],
                    "content": "".join(part["text"] for part in msg_json["content"] if part["type"] == "text"),
                },
                "finish_reason": msg_json.get("stop_reason", "stop"),
            }
        ],
    }
    if msg_json.get("usage"):
        response["usage"] = to_openai_usage(msg_json["usage"])
    return json.dumps(response)


def get_chat_completion_model_name(model_alias):
    if model_alias.startswith("publishers/google/"):
        return f"google/{model_alias.split('/')[-1]}"

    return model_alias.split("/")[-1]


def transform_vertex_chunk_to_openai(chunk_line: str, index: int = 0) -> str:
    if not chunk_line.startswith("data: "):
        return ""  # skip irrelevant lines like empty keepalives

    try:
        payload = json.loads(chunk_line[len("data: ") :])
        parts = payload.get("candidates", [])[0].get("content", {}).get("parts", [])
        if not parts:
            return ""
        text = parts[0].get("text", "")
    except Exception:
        return ""

    transformed = {"choices": [{"delta": {"content": text}, "index": index, "finish_reason": None}]}

    return f"data: {json.dumps(transformed)}\n"


async def stream_generator(
    target_url: str, request_headers: dict, content_json: dict, model_alias: str
) -> AsyncGenerator[str, None]:
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            target_url,
            headers=request_headers,
            json=content_json,
        ) as response:
            logging.debug(f"Received response with status code: {response.status_code}")
            logging.debug(f"Response headers: {response.headers}")

            async for line in response.aiter_lines():
                logging.debug(f"Received line: {line}")
                if not line.strip():
                    continue

                if line.strip() == "data: [DONE]":
                    yield sse_done()
                    break

                if line.startswith("data: "):
                    raw_json = line[len("data: ") :].strip()
                else:
                    raw_json = line.strip()
                async for chunk in handle_data_line(raw_json, model_alias):
                    logging.debug(f"Yielding chunk: '{chunk}'")
                    yield chunk
    yield sse_done()


@router.post(
    "/completions", response_model=ChatResponse | ChatStreamResponse | Error, response_model_exclude_unset=True
)
async def handle_proxy(request: Request):
    try:
        content = await request.body()
        content_json = json.loads(content)
        is_streaming = content_json.get("stream", False)
        model_alias = content_json.get("model", "chat-default")
        model = get_model("gcp", model_alias, "chat-default")

        if USE_MODEL_MAPPING:
            content_json["model"] = get_chat_completion_model_name(model)

        conversion_target = None
        if model not in known_chat_models:
            # openai messages to vertex contents
            if "anthropic" in model:
                content_json = to_vertex_anthropic(content_json)
                conversion_target = "anthropic"

        # Build safe target URL
        target_url, request_headers = get_headers_and_target(
            model, request, "chat/completions", is_streaming, known_chat_models
        )

        logging.debug(f"Proxying request to: {target_url}")
        logging.debug(f"Request headers: {request_headers}")
        logging.debug(f"Request content: {content_json}")
        if is_streaming:
            return StreamingResponse(
                stream_generator(target_url, request_headers, content_json, model_alias), media_type="text/event-stream"
            )

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=request_headers,
                content=json.dumps(content_json),
                params=request.query_params,
                timeout=5.0,
            )

        content = response.content
        if conversion_target == "anthropic":
            # convert vertex response to openai format
            content = from_anthropic_to_openai_response(response.content, model_alias)

    except httpx.RequestError as e:
        logging.error(f"Proxy request failed: {e}")
        return Response(status_code=502, content=f"Upstream request failed: {e}")

    # remove hop-by-hop headers
    response_headers = {
        k: v
        for k, v in response.headers.items()
        if k.lower() not in {"content-encoding", "transfer-encoding", "connection"}
    }

    return Response(
        content=content,
        status_code=response.status_code,
        headers=response_headers,
        media_type=response.headers.get("content-type", "application/octet-stream"),
    )
