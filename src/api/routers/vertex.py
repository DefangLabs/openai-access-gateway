import httpx
import json
import logging
import os
import requests
import uuid

from fastapi import Request, Response
from contextlib import asynccontextmanager
from api.setting import API_ROUTE_PREFIX, GCP_PROJECT_ID, GCP_REGION, USE_MODEL_MAPPING
from google.auth import default
from google.auth.transport.requests import Request as AuthRequest

from api.modelmapper import get_model

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


# GCP credentials and project details
credentials = None
project_id = None
location = None

def get_gcp_project_details():
    from google.auth import default

    # Try metadata server for region
    credentials = None
    project_id = GCP_PROJECT_ID
    location = GCP_REGION

    try:
        credentials, project = default()
        if not project_id:
            project_id = project

        if not location:
            zone = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/zone",
                headers={"Metadata-Flavor": "Google"},
                timeout=1
            ).text
            location = zone.split("/")[-1].rsplit("-", 1)[0]

    except Exception:
        logging.warning(f"Error: Failed to get project and location from metadata server. Using local settings.")

    return credentials, project_id, location

credentials, project_id, location = get_gcp_project_details()

# Utility: get service account access token
def get_access_token():
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    auth_request = AuthRequest()
    credentials.refresh(auth_request)
    return credentials.token

def get_proxy_target(model, path):
    """
    Check if the environment variable is set to use GCP.
    """
    if os.getenv("PROXY_TARGET"):
        return os.getenv("PROXY_TARGET")
    elif model in known_chat_models and path.endswith("/chat/completions"):
        return f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi/chat/completions"
    else:
        return f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/{model}:rawPredict"

def get_header(model, request, path):
    path_no_prefix = f"/{path.lstrip('/')}".removeprefix(API_ROUTE_PREFIX)
    target_url = get_proxy_target(model, path_no_prefix)

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
    # publishers/google/models/gemini-2.0-flash-lite-001 -> "gemini-2.0-flash-lite-001"
    return model_alias.split("/")[-1]

async def handle_proxy(request: Request, path: str):
    try:
        content = await request.body()
        content_json = json.loads(content)
        model_alias = content_json.get("model", "default")
        model = get_model("gcp", model_alias)

        if USE_MODEL_MAPPING:
            if "model" in content_json:
                content_json["model"]= get_chat_completion_model_name(model)

        conversion_target = None
        if not model in known_chat_models:
            # openai messages to vertex contents 
            if "anthropic" in model:
                content_json = to_vertex_anthropic(content_json)
                conversion_target = "anthropic"

        # Build safe target URL
        target_url, headers = get_header(model, request, path)
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
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
