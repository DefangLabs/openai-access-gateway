import httpx
import json
import logging
import os

from fastapi import APIRouter, Body, Depends, Request, Response
from contextlib import asynccontextmanager
from api.auth import api_key_auth
from api.schema import EmbeddingsResponse
from api.setting import API_ROUTE_PREFIX
from google.auth import default
from google.auth.transport.requests import Request as AuthRequest

from api.modelmapper import get_model
from api.gcp.credentials.metadata import get_access_token, project_id, location

router = APIRouter(
    prefix="/embeddings",
    dependencies=[Depends(api_key_auth)],
)
def get_proxy_target(model, path):
    """
    Check if the environment variable is set to use GCP.
    """
    if os.getenv("PROXY_TARGET"):
        return os.getenv("PROXY_TARGET")
    elif path.endswith("/embeddings"):
        return f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/{model}:predict"

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

@router.post("", response_model=EmbeddingsResponse)
async def handle_proxy(request: Request, path: str):
    try:
        content = await request.body()
        content_json = json.loads(content)
        model_alias = content_json.get("model", "default")
        model = get_model("gcp", model_alias)

        # Build safe target URL
        target_url, request_headers = get_header(model, request, path)
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
