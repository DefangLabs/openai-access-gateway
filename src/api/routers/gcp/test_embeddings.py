import pytest
from api.routers.gcp.embeddings import to_vertex_embeddings
import json
from unittest.mock import patch, AsyncMock, MagicMock
import httpx
from fastapi import Response, Request
from starlette.datastructures import Headers, QueryParams
from api.routers.gcp.embeddings import to_openai_response

def test_to_vertex_embeddings_with_string_input():
    request = {"input": "hello world"}
    expected = {
        "instances": [
            {"content": "hello world"}
        ]
    }
    assert to_vertex_embeddings(request) == expected

def test_to_vertex_embeddings_with_list_of_strings():
    request = {"input": ["foo", "bar"]}
    expected = {
        "instances": [
            {"content": "foo"},
            {"content": "bar"}
        ]
    }
    assert to_vertex_embeddings(request) == expected

def test_to_vertex_embeddings_with_list_of_numbers():
    request = {"input": [1, 2, 3]}
    expected = {
        "instances": [
            {"content": "1"},
            {"content": "2"},
            {"content": "3"}
        ]
    }
    assert to_vertex_embeddings(request) == expected

def test_to_vertex_embeddings_with_empty_list():
    request = {"input": []}
    expected = {
        "instances": []
    }
    assert to_vertex_embeddings(request) == expected

def test_to_vertex_embeddings_with_missing_input_key():
    request = {}
    expected = {
        "instances": []
    }
    assert to_vertex_embeddings(request) == expected

def test_to_openai_response_with_multiple_predictions():
    embedding_content = {
        "predictions": [
            {"embeddings": {"values": [0.1, 0.2, 0.3], "statistics": {"token_count": 10}}},
            {"embeddings": {"values": [0.4, 0.5, 0.6], "statistics": {"token_count": 20}}}
        ]
    }
    model = "test-model"
    expected = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"},
            {"embedding": [0.4, 0.5, 0.6], "index": 1, "object": "embedding"}
        ],
        "model": "test-model",
        "object": "list",
        "usage": {
            "total_tokens": 30
        }
    }
    assert to_openai_response(embedding_content, model) == expected

def test_to_openai_response_with_empty_predictions():
    embedding_content = {
        "predictions": []
    }
    model = "empty-model"
    expected = {
        "data": [],
        "model": "empty-model",
        "object": "list",
        "usage": {
            "total_tokens": 0
        }
    }
    assert to_openai_response(embedding_content, model) == expected

def test_to_openai_response_with_missing_predictions_key():
    embedding_content = {}
    model = "default-model"
    expected = {
        "data": [],
        "model": "default-model",
        "object": "list",
        "usage": {
            "total_tokens": 0
        }
    }
    assert to_openai_response(embedding_content, model) == expected

def test_to_openai_response_with_single_prediction():
    embedding_content = {
        "predictions": [
            {
                "embeddings": {
                    "values": [1.0, 2.0, 3.0], "statistics": {"token_count": 5}
                }
            }
        ]
    }
    model = "single-model"
    expected = {
        "data": [
            {"embedding": [1.0, 2.0, 3.0], "index": 0, "object": "embedding"}
        ],
        "model": "single-model",
        "object": "list",
        "usage": {
            "total_tokens": 5
        }
    }
    assert to_openai_response(embedding_content, model) == expected

def test_to_openai_response_with_nonstandard_embedding_key():
    # This should raise a KeyError if "embeddings" or "values" is missing
    embedding_content = {
        "predictions": [
            {"not_embeddings": {"values": [1, 2, 3]}}
        ]
    }
    model = "bad-model"
    try:
        to_openai_response(embedding_content, model)
        assert False, "Expected KeyError"
    except KeyError:
        pass
