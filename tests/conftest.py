"""Pytest configuration and fixtures."""

from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
async def async_client() -> AsyncIterator[AsyncClient]:
    """Create an async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_hf_search_response():
    """Mock HuggingFace search API response."""
    return [
        {
            "id": "meta-llama/Llama-3.1-8B-Instruct",
            "modelId": "meta-llama/Llama-3.1-8B-Instruct",
            "cardData": {"license": "llama3.1", "tags": ["text-generation"]},
            "private": False,
            "gated": "manual",  # HF API returns string for gated models
            "tags": ["text-generation", "llama"],
        },
        {
            "id": "bert-base-uncased",
            "modelId": "bert-base-uncased",
            "cardData": {"license": "apache-2.0"},
            "private": False,
            "gated": False,
            "tags": ["text-classification"],
        },
    ]


@pytest.fixture
def mock_hf_repo_response():
    """Mock HuggingFace repo details API response."""
    return {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "modelId": "meta-llama/Llama-3.1-8B-Instruct",
        "private": False,
        "gated": "manual",  # HF API returns string for gated models
        "tags": ["text-generation", "llama", "pytorch"],
        "lastModified": "2024-07-23T14:48:00.000Z",
        "cached": False,
        "siblings": [
            {
                "rfilename": "config.json",
                "size": 1234,
                "blobId": "abc123",
                "lfs": None,
            },
            {
                "rfilename": "pytorch_model.bin",
                "size": 16000000000,
                "blobId": "def456",
                "lfs": {"size": 16000000000, "sha256": "xyz789"},
            },
        ],
    }
