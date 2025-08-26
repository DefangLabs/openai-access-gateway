import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends
from fastapi.responses import StreamingResponse

from api.auth import api_key_auth
from api.modelmapper import get_model
from api.models.bedrock import BedrockModel
from api.schema import ChatRequest, ChatResponse, ChatStreamResponse, Error, ErrorMessage
from api.setting import DEFAULT_MODEL, USE_MODEL_MAPPING

router = APIRouter(
    prefix="/chat",
    dependencies=[Depends(api_key_auth)],
    responses={404: {"description": "Not found"}},
)


@router.post(
    "/completions", response_model=ChatResponse | ChatStreamResponse | Error, response_model_exclude_unset=True
)
async def chat_completions(
    chat_request: Annotated[
        ChatRequest,
        Body(
            examples=[
                {
                    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                }
            ],
        ),
    ],
):
    try:
        logging.info(f"Chat request received: {chat_request}")
        if chat_request.model is not None and chat_request.model.lower().startswith("gpt-"):
            chat_request.model = DEFAULT_MODEL

        # replace with mapped model name
        if USE_MODEL_MAPPING:
            req_model = chat_request.model
            req_model = get_model("aws", req_model, "chat-default")
            chat_request.model = req_model

        model = BedrockModel()
        # Exception will be raised if model not supported.
        model.validate(chat_request)

        if chat_request.stream:
            return StreamingResponse(content=model.chat_stream(chat_request), media_type="text/event-stream")
        return await model.chat(chat_request)
    except Exception as e:
        logging.error(f"Chat request failed: {e}")
        return Error(error=ErrorMessage(message="Chat request failed"))
