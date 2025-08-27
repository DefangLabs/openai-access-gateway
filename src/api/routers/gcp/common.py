"""Common utility functions for GCP routers."""

import os
import time

from api.gcp.credentials.metadata import get_access_token, location, project_id
from api.setting import API_ROUTE_PREFIX


def to_openai_usage(usage: dict) -> dict:
    """Convert GCP usage format to OpenAI usage format."""
    return {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
    }


def get_proxy_target(model: str, path: str, stream: bool = False, known_chat_models: list = None):
    """
    Get the appropriate GCP proxy target URL.

    Args:
        model: The model identifier
        path: The API path
        stream: Whether this is for streaming (chat only)
        known_chat_models: List of known chat models (chat endpoints only)
    """
    if os.getenv("PROXY_TARGET"):
        return os.getenv("PROXY_TARGET")

    # Chat completions endpoint
    if known_chat_models and model in known_chat_models and path.endswith("/chat/completions"):
        return f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi/chat/completions"

    # Embeddings endpoint
    elif path.endswith("/embeddings"):
        return f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/{model}:predict"

    # Other endpoints (chat with streaming)
    else:
        endpoint_suffix = "streamRawPredict" if stream else "rawPredict"
        return f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/{model}:{endpoint_suffix}"


def get_headers_and_target(model: str, request, path: str, stream: bool = False, known_chat_models: list = None):
    """
    Get headers and target URL for GCP requests.

    Args:
        model: The model identifier
        request: FastAPI request object
        path: The API path
        stream: Whether this is for streaming
        known_chat_models: List of known chat models (for chat endpoints)

    Returns:
        tuple: (target_url, headers)
    """
    path_no_prefix = f"/{path.lstrip('/')}".removeprefix(API_ROUTE_PREFIX)
    target_url = get_proxy_target(model, path_no_prefix, stream, known_chat_models)

    # Remove hop-by-hop headers
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length", "accept-encoding", "connection", "authorization"}
    }

    # Fetch service account token
    access_token = get_access_token()
    headers["Authorization"] = f"Bearer {access_token}"
    return target_url, headers


# SSE (Server-Sent Events) utilities
def sse_chunk(payload: str) -> str:
    """Format a payload as an SSE chunk."""
    return f"data: {payload}\n\n"


def sse_done() -> str:
    """Return the SSE done marker."""
    return "data: [DONE]\n\n"


def generate_openai_id() -> str:
    """Generate an OpenAI-style ID with timestamp."""
    return f"chatcmpl-ts{int(time.time() * 1000)}"
