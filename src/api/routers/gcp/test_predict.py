from api.routers.gcp.predict import to_vertex_content
import types
from api.routers.gcp.predict import aggregate_parts
import pytest
from unittest import mock
from api.routers.gcp import predict
from api.routers.gcp.predict import handle_gemini, make_response

class Msg:
    def __init__(self, role, content):
        self.role = role
        self.content = content

def test_to_vertex_content_single_message():
    messages = [Msg(role="user", content="Test Content")]
    result = to_vertex_content(messages)
    assert result == [
        {
            "role": "user",
            "parts": [{"text": "Test Content"}]
        }
    ]

def test_to_vertex_content_multiple_messages():
    messages = [
        Msg(role="user", content="Test Content 1"),
        Msg(role="assistant", content="Test Content 2")
    ]
    result = to_vertex_content(messages)
    assert result == [
        {
            "role": "user",
            "parts": [{"text": "Test Content 1"}]
        },
        {
            "role": "assistant",
            "parts": [{"text": "Test Content 2"}]
        }
    ]

def test_to_vertex_content_empty_list():
    messages = []
    result = to_vertex_content(messages)
    assert result == []

class DummyPart:
    def __init__(self, text=None):
        self.text = text

class DummyContent:
    def __init__(self, parts):
        self.parts = parts

class DummyCandidate:
    def __init__(self, parts):
        self.content = DummyContent(parts)

class DummyResponse:
    def __init__(self, candidates):
        self.candidates = candidates

def test_aggregate_parts_single_candidate_single_part():
    response = DummyResponse([
        DummyCandidate([DummyPart("Dummy Message")])
    ])
    result = aggregate_parts(response)
    assert result == "Dummy Message"

def test_aggregate_parts_single_candidate_multiple_parts():
    response = DummyResponse([
        DummyCandidate([DummyPart("Dummy Message 1"), DummyPart("Dummy Message 2")])
    ])
    result = aggregate_parts(response)
    assert result == "Dummy Message 1\nDummy Message 2"

def test_aggregate_parts_multiple_candidates():
    response = DummyResponse([
        DummyCandidate([DummyPart("First")]),
        DummyCandidate([DummyPart("Second")])
    ])
    result = aggregate_parts(response)
    assert result == "First\nSecond"

def test_aggregate_parts_no_candidates():
    response = DummyResponse([])
    result = aggregate_parts(response)
    assert result == ""

def test_aggregate_parts_part_without_text():
    response = DummyResponse([
        DummyCandidate([])
    ])
    result = aggregate_parts(response)
    assert result == ""

def make_default_mock(credentials="creds", project="proj"):
    def _default():
        return credentials, project
    return _default

@mock.patch("api.routers.gcp.predict.GCP_PROJECT_ID", None)
@mock.patch("api.routers.gcp.predict.GCP_REGION", None)
@mock.patch("api.routers.gcp.predict.requests.get")
@mock.patch("api.routers.gcp.predict.google.auth.transport.requests.Request")
@mock.patch("api.routers.gcp.predict.google.auth.default")
def test_get_project_details_success(
    mock_default, mock_request, mock_requests_get
):
    # Setup
    mock_default.return_value = ("fake_creds", "my_project")
    mock_requests_get.return_value.text = "projects/my_project/zones/fake_region"
    mock_requests_get.return_value.ok = True
    mock_request.return_value = "auth_req"

    creds, project_id, location, auth_req = predict.get_project_details()

    assert creds == "fake_creds"
    assert project_id == "my_project"
    assert location == "fake_region"
    assert auth_req == "auth_req"

@mock.patch("api.routers.gcp.predict.GCP_PROJECT_ID", "explicit_project")
@mock.patch("api.routers.gcp.predict.GCP_REGION", "explicit_region")
@mock.patch("api.routers.gcp.predict.google.auth.transport.requests.Request")
@mock.patch("api.routers.gcp.predict.google.auth.default")
def test_get_project_details_env_vars(
    mock_default, mock_request
):
    # Setup
    mock_default.return_value = ("fake_creds", "my_project")
    mock_request.return_value = "auth_req"

    creds, project_id, location, auth_req = predict.get_project_details()

    assert creds == "fake_creds"
    assert project_id == "explicit_project"
    assert location == "explicit_region"
    assert auth_req == "auth_req"

@mock.patch("api.routers.gcp.predict.GCP_PROJECT_ID", None)
@mock.patch("api.routers.gcp.predict.GCP_REGION", None)
@mock.patch("api.routers.gcp.predict.requests.get")
@mock.patch("api.routers.gcp.predict.google.auth.transport.requests.Request")
@mock.patch("api.routers.gcp.predict.google.auth.default")
def test_get_project_details_zone_parse(
    mock_default, mock_request, mock_requests_get
):
    # Setup
    mock_default.return_value = ("fake_creds", "my_project")
    mock_requests_get.return_value.text = "projects/my_project/zones/europe-west2-c"
    mock_requests_get.return_value.ok = True
    mock_request.return_value = "auth_req"

    creds, project_id, location, auth_req = predict.get_project_details()

    assert creds == "fake_creds"
    assert project_id == "my_project"
    assert location == "europe-west2"
    assert auth_req == "auth_req"

@mock.patch("api.routers.gcp.predict.GCP_PROJECT_ID", None)
@mock.patch("api.routers.gcp.predict.GCP_REGION", None)
@mock.patch("api.routers.gcp.predict.requests.get", side_effect=Exception("fail"))
@mock.patch("api.routers.gcp.predict.google.auth.transport.requests.Request")
@mock.patch("api.routers.gcp.predict.google.auth.default", side_effect=Exception("fail"))
def test_get_project_details_exception(mock_default, mock_request, mock_requests_get):
    creds, project_id, location, auth_req = predict.get_project_details()
    assert creds is None
    assert project_id is None
    assert location is None
    assert auth_req is None

class ChatRequest:
    def __init__(self, model, messages):
        self.model = model
        self.messages = messages

@mock.patch("api.routers.gcp.predict.get_model")
@mock.patch("api.routers.gcp.predict.to_vertex_content")
@mock.patch("api.routers.gcp.predict.aggregate_parts")
@mock.patch("api.routers.gcp.predict.GenerativeModel")
def test_handle_gemini_success(
    mock_generative_model, mock_aggregate_parts, mock_to_vertex_content, mock_get_model
):
    # setup
    test_model_id = "google/models/fake-gemini-2.0-flash-lite-001"
    test_model_name = "fake-gemini-2.0-flash-lite-001"
    test_messages = [mock.Mock(role="user", content="hello")]
    test_vertex_content = [{"role": "user", "parts": [{"text": "hello"}]}]
    test_response = mock.Mock()
    test_aggregated = "response text"

    mock_get_model.return_value = test_model_id
    mock_to_vertex_content.return_value = test_vertex_content
    mock_aggregate_parts.return_value = test_aggregated

    test_generate_content = mock.Mock(return_value=test_response)
    generative_model_instance = mock.Mock(generate_content=test_generate_content)
    mock_generative_model.return_value = generative_model_instance

    req = ChatRequest(model=test_model_name, messages=test_messages)

    # test
    response = handle_gemini(req)

    mock_generative_model.assert_called_once_with(test_model_name)
    test_generate_content.assert_called_once_with(test_vertex_content)
    assert response == make_response(test_aggregated)

@mock.patch("api.routers.gcp.predict.get_model")
@mock.patch("api.routers.gcp.predict.to_vertex_content")
@mock.patch("api.routers.gcp.predict.GenerativeModel")
def test_handle_gemini_exception(
    mock_generative_model, mock_to_vertex_content, mock_get_model
):
    # setup
    dummy_model_id = "google/models/fake-gemini-2.0-flash-lite-001"
    dummy_messages = [mock.Mock(role="user", content="hello")]

    mock_get_model.return_value = dummy_model_id
    mock_to_vertex_content.return_value = [{"role": "user", "parts": [{"text": "hello"}]}]

    # make GenerativeModel raise an exception
    mock_generative_model.side_effect = RuntimeError("fail")

    req = ChatRequest(model="ai/fake-model", messages=dummy_messages)

    # test
    result = handle_gemini(req)

    assert "Error: fail" in result["choices"][0]["message"]["content"]
    assert result["choices"][0]["message"]["role"] == "assistant"
