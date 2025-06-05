import logging
import os
import uvicorn

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from mangum import Mangum

from api.setting import API_ROUTE_PREFIX, DESCRIPTION, SUMMARY, PROVIDER, TITLE, USE_MODEL_MAPPING, VERSION
from api.modelmapper import load_model_map
from api.routers.vertex import handle_proxy

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
