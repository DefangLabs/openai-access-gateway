import logging
import requests
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from mangum import Mangum
import httpx
import json
import os
from contextlib import asynccontextmanager

from api.setting import API_ROUTE_PREFIX, DESCRIPTION, GCP_PROJECT_ID, GCP_REGION, SUMMARY, PROVIDER, TITLE, USE_MODEL_MAPPING, VERSION

from google.auth import default
from google.auth.transport.requests import Request as AuthRequest

from api.modelmapper import get_model, load_model_map

# GCP credentials and project details
credentials = None
project_id = None
location = None

def is_aws():
    env = os.getenv("AWS_EXECUTION_ENV")
    if env == "AWS_ECS_FARGATE":
        return True
    elif env == "AWS_ECS_EC2":
        return True
    elif os.getenv("ECS_CONTAINER_METADATA_URI_V4"):
        return True
    return False

provider = PROVIDER.lower() if PROVIDER else None
if provider == None:
    if is_aws():
        provider = "aws"
    else:
        provider = "gcp"

if USE_MODEL_MAPPING:
    load_model_map()


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

if not is_aws():
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

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=content,
                params=request.query_params,
                timeout=30.0,
            )
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

config = {
    "title": TITLE,
    "description": DESCRIPTION,
    "summary": SUMMARY,
    "version": VERSION,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = FastAPI(**config)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if provider != "aws":
    logging.info(f"Proxy target set to: GCP")
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
    async def proxy(request: Request, path: str):
        return await handle_proxy(request, path)
else:
    from api.routers import chat, embeddings, model
    logging.info("No proxy target set. Using internal routers.")
    app.include_router(model.router, prefix=API_ROUTE_PREFIX)
    app.include_router(chat.router, prefix=API_ROUTE_PREFIX)
    app.include_router(embeddings.router, prefix=API_ROUTE_PREFIX)

@app.get("/health")
async def health():
    """For health check if needed"""
    return {"status": "OK"}

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)

handler = Mangum(app)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
