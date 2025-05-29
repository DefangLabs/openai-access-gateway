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

def get_gcp_target(path):
    """
    Check if the environment variable is set to use GCP.
    """
    if os.getenv("PROXY_TARGET"):
        return os.getenv("PROXY_TARGET")
    else:
        return f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/{path.lstrip('/')}".rstrip("/")

def get_header(request, path):
    if "chat/completions" in path:
        path = path.replace("chat/completions", "endpoints/openapi/chat/completions")

    path_no_prefix = f"/{path.lstrip('/')}".removeprefix(API_ROUTE_PREFIX)
    target_url = get_gcp_target(path_no_prefix)

    # remove hop-by-hop headers
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length", "accept-encoding", "connection", "authorization"}
    }

    # Fetch service account token
    access_token = get_access_token()
    headers["Authorization"] = f"Bearer {access_token}"
    return target_url,headers

def to_vertex_contents(openai_messages):
    """
    Convert OpenAI-style chat messages to Anthropic `contents` format for Vertex AI.
    """
    anthropic_contents = []

    for msg in openai_messages:
        # Skip empty messages
        if not msg.get("content"):
            continue

        role = msg["role"]
        if role == "system":
            # System messages can be treated as a user message prefixing the context
            anthropic_contents.append({
                "role": "user",
                "parts": [{"text": f"[System instruction]: {msg['content']}"}]
            })
        else:
            anthropic_contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

    return anthropic_contents


def to_openai_response(resp):
    """
    Convert an Vertex AI response to an OpenAI-style chat completion format.
    """
    if "candidates" not in resp or not resp["candidates"]:
        raise ValueError("No candidates in response")

    choices = []
    for i, candidate in enumerate(resp["candidates"]):
        content_parts = candidate.get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in content_parts)

        choices.append({
            "index": i,
            "message": {
                "role": "assistant",
                "content": text
            },
            "finish_reason": candidate.get("finishReason", "stop").lower()
        })

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "choices": choices
    }

async def handle_proxy(request: Request, path: str):
    # Build safe target URL
    target_url, headers = get_header(request, path)

    try:
        content = await request.body()

        if USE_MODEL_MAPPING:
            data = json.loads(content)
            if "model" in data:
                request_model = data.get("model", None)
                model = get_model("gcp", request_model)

                if model != None and model != request_model and "publishers/google/" in model:
                    model = f"google/{model.split('/')[-1]}"

                data["model"]= model
            content = json.dumps(data)

        needs_conversion = False
        if "messages" in content:
            needs_conversion = True
            # openai messages to vertex contents 
            content = to_vertex_contents(content)

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=content,
                params=request.query_params,
                timeout=30.0,
            )
        if needs_conversion:
            # convert vertex response to openai format
            response = to_openai_response(response.json())

    except httpx.RequestError as e:
        logging.error(f"Proxy request failed: {e}")
        return Response(status_code=502, content=f"Upstream request failed: {e}")

    # remove hop-by-hop headers
    response_headers = {
        k: v for k, v in response.headers.items()
        if k.lower() not in {"content-encoding", "transfer-encoding", "connection"}
    }

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=response_headers,
        media_type=response.headers.get("content-type", "application/octet-stream"),
    )
