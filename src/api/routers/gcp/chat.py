import httpx
import json
import logging
import os

from typing import AsyncGenerator
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from api.setting import API_ROUTE_PREFIX, GCP_PROJECT_ID, GCP_REGION, USE_MODEL_MAPPING
from google.auth import default
from google.auth.transport.requests import Request as AuthRequest

from api.auth import api_key_auth
from api.modelmapper import get_model
from api.gcp.credentials.metadata import get_access_token, project_id, location
from api.schema import ChatResponse, ChatStreamResponse, Error
from api.routers.gcp.stream_transformers import handle_data_line, openai_done, openai_chunk

known_chat_models = [
    "publishers/mistral-ai/models/mistral-7b-instruct-v0.3",
    "publishers/mistral-ai/models/mistral-nemo-instruct-2407",
    "publishers/mistral-ai/models/mistral-nemo@2407",
    "publishers/mistral-ai/models/mistral-7b-instruct@v0.3",
    "publishers/google/models/gemma-2-27b-it",
    "publishers/google/models/gemma-2-9b-it",
    "publishers/google/models/gemma-2b"
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

def get_proxy_target(model, path, stream):
    """
    Check if the environment variable is set to use GCP.
    """
    if os.getenv("PROXY_TARGET"):
        return os.getenv("PROXY_TARGET")
    elif model in known_chat_models and path.endswith("/chat/completions"):
        return f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi/chat/completions"
    else:
        endPointSuffix = "streamRawPredict" if stream else "rawPredict"
        return f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/{model}:{endPointSuffix}"

def get_headers(model, request, path, stream):
    target_url = None
    path_no_prefix = f"/{path.lstrip('/')}".removeprefix(API_ROUTE_PREFIX)
    target_url = get_proxy_target(model, path_no_prefix, stream)

    # remove hop-by-hop headers
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length", "accept-encoding", "connection", "authorization"}
    }

    # Fetch service account token
    access_token = get_access_token()
    headers["Authorization"] = f"Bearer {access_token}"
    return target_url, headers

def to_vertex_anthropic(openai_messages):
    message = [
        {
            "role": m["role"],
            "content": [{"type": "text", "text": m["content"]}]
        }
        for m in openai_messages["messages"]
    ]

    return {
        "anthropic_version": "vertex-2023-10-16",
        "max_tokens": 256,
        "messages": message
    }

def from_anthropic_to_openai_response(msg, model):
    msg_json = json.loads(msg)
    return json.dumps({
        "id": msg_json["id"],
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": msg_json["role"],
                    "content": "".join(
                        part["text"] for part in msg_json["content"]
                        if part["type"] == "text"
                    )
                },
                "finish_reason": msg_json.get("stop_reason", "stop")
            }
        ],
        "usage": msg_json.get("usage", {})
    })

def get_chat_completion_model_name(model_alias):
    if model_alias.startswith("publishers/google/"):
        return f"google/{model_alias.split('/')[-1]}"

    return model_alias.split('/')[-1]

def transform_vertex_chunk_to_openai(chunk_line: str, index: int = 0) -> str:
    if not chunk_line.startswith("data: "):
        return ""  # skip irrelevant lines like empty keepalives

    try:
        payload = json.loads(chunk_line[len("data: "):])
        parts = payload.get("candidates", [])[0].get("content", {}).get("parts", [])
        if not parts:
            return ""
        text = parts[0].get("text", "")
    except Exception:
        return ""

    transformed = {
        "choices": [
            {
                "delta": {"content": text},
                "index": index,
                "finish_reason": None
            }
        ]
    }

    return f"data: {json.dumps(transformed)}\n"

async def stream_generator(target_url: str, request_headers: dict, content_json: dict, model_alias: str) -> AsyncGenerator[str, None]:
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            target_url,
            headers=request_headers,
            json=content_json,
        ) as response:

            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                if line.strip() == "data: [DONE]":
                    yield openai_done()
                    break

                if line.startswith("data: "):
                    raw_json = line[6:].strip()
                    async for chunk in handle_data_line(raw_json, model_alias):
                        yield chunk
                else:
                    yield openai_chunk(line.strip())
    yield openai_done()

@router.post(
    "/completions", response_model=ChatResponse | ChatStreamResponse | Error, response_model_exclude_unset=True
)
async def handle_proxy(request: Request):
    try:
        content = await request.body()
        content_json = json.loads(content)
        is_streaming = content_json.get("stream", False)
        model_alias = content_json.get("model", "default")
        model = get_model("gcp", model_alias)

        if USE_MODEL_MAPPING:
            if "model" in content_json:
                content_json["model"] = get_chat_completion_model_name(model)

        conversion_target = None
        if not model in known_chat_models:
            # openai messages to vertex contents 
            if "anthropic" in model:
                content_json = to_vertex_anthropic(content_json)
                conversion_target = "anthropic"

        # Build safe target URL
        target_url, request_headers = get_headers(model, request, "chat/completions", is_streaming)

        if is_streaming:           
            return StreamingResponse(stream_generator(target_url, request_headers, content_json, model_alias), media_type="text/event-stream")
        else:
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
        k: v for k, v in response.headers.items()
        if k.lower() not in {"content-encoding", "transfer-encoding", "connection"}
    }

    return Response(
        content=content,
        status_code=response.status_code,
        headers=response_headers,
        media_type=response.headers.get("content-type", "application/octet-stream"),
    )
