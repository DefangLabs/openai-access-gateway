import requests
import logging
from fastapi import APIRouter, Depends
from google.cloud import aiplatform_v1beta1, aiplatform
import google.auth.transport.requests

from api.auth import api_key_auth
from api.modelmapper import get_model
from api.routers.gcp.chat_types import ChatRequest, ChatCompletionResponse

from api.setting import GCP_PROJECT_ID, GCP_REGION

from vertexai.preview.generative_models import GenerativeModel
import vertexai

client = None

router = APIRouter(
    prefix="/chat",
    dependencies=[Depends(api_key_auth)],
    responses={404: {"description": "Not found"}},
)

def is_gcp():
    try:
        # contact gcp metadata server to check if running on GCP
        res = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/project/project-id",
            headers={"Metadata-Flavor": "Google"},
            timeout=0.5
        )
        return res.ok
    except:
        return False

def get_project_and_location_and_auth():
    from google.auth import default

    # Try metadata server for region
    credentials = None
    project_id = GCP_PROJECT_ID
    location = GCP_REGION
    auth_req = None

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

        auth_req = google.auth.transport.requests.Request()
    except Exception:
        logging.warning(f"Error: Failed to get project and location from metadata server. Using local settings.")

    return credentials, project_id, location, auth_req

def to_vertex_content(messages):
    return [
        {
            "role": msg.role,
            "parts": [{"text": msg.content}]
        }
        for msg in messages
    ]

def aggregate_parts(response):
    generated_texts = []
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, "text"):
                generated_texts.append(part.text)
    return "\n".join(generated_texts)

credentials, project_id, location, auth_req = get_project_and_location_and_auth()

if project_id and location:
    vertexai.init(
        project=project_id,
        location=location,
    )
    client = aiplatform_v1beta1.PredictionServiceClient()


def make_response(content):
    return {
        "id": "chat-response",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ]
    }

def handle_gemini(request: ChatRequest):
    content = ""
    try:
        modelId = get_model("gcp", request.model) 
        model_name = modelId.split("/")[-1]
        model = GenerativeModel(model_name)

        content = to_vertex_content(request.messages)
        response = model.generate_content(content)

        content = aggregate_parts(response)
    except Exception as e:
        content = f"ProjectID({project_id} - Location({location}))Error: {str(e)}"

    return make_response(content)

def to_prompt_instances(messages):
    return [
            {
                "prompt": msg.content,
                "temperature": 0.7, # balance between randomness and determinism
                "max_tokens": 100, # short summary sizes
            }
            for msg in messages
        ]

def handle_vertex(request: ChatRequest):
    modelId = get_model("gcp", request.model)
    endpoint = f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/{modelId}:rawPredict"
    payload = {"instances": to_prompt_instances(request.messages)}

    # Send request
    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json"
    }

    response = requests.post(endpoint, headers=headers, json=payload)
    return make_response(response.json())

# def handle_predict(request: ChatRequest):
#     modelId = get_model("gcp", request.model)
#     model = aiplatform.Model(modelId)
#     payload = to_prompt_instances(request.messages)
#     prediction = model.predict(instances=payload)
#     return make_response(prediction.predictions[0])

@router.post("/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatRequest):
    credentials.refresh(auth_req)
    model = get_model("gcp", request.model)
    if "gemini" in model.lower():
        return handle_gemini(request)
    else:
        return handle_vertex(request)

