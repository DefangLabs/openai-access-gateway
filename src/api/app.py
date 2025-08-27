import logging
import os

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from mangum import Mangum

from api.modelmapper import load_model_map
from api.setting import API_ROUTE_PREFIX, DESCRIPTION, PROVIDER, SUMMARY, TITLE, USE_MODEL_MAPPING, VERSION


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
if provider is None:
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

level = logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
logging.basicConfig(
    level=level,
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
    from api.routers.gcp import chat, embeddings

    logging.info("Proxy target set to: GCP")
    app.include_router(chat.router, prefix=API_ROUTE_PREFIX)
    app.include_router(embeddings.router, prefix=API_ROUTE_PREFIX)
else:
    from api.routers import chat, embeddings, model

    logging.info("No proxy target set. Using AWS.")
    app.include_router(model.router, prefix=API_ROUTE_PREFIX)
    app.include_router(chat.router, prefix=API_ROUTE_PREFIX)
    app.include_router(embeddings.router, prefix=API_ROUTE_PREFIX)


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint for the API"""
    return {"status": "OK"}


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
