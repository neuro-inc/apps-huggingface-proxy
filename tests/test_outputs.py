"""Tests for outputs endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.services import HuggingFaceService


def create_mock_service(search_response=None, repo_response=None, should_fail=False):
    """Helper to create a properly configured mock service."""
    mock_service = AsyncMock(spec=HuggingFaceService)

    if should_fail:
        mock_service.search_models.side_effect = Exception("API Error")
        mock_service.get_repo_details.side_effect = Exception("API Error")
    else:
        if search_response:
            mock_service.search_models.return_value = search_response
        if repo_response:
            mock_service.get_repo_details.return_value = repo_response

    mock_service.__aenter__.return_value = mock_service
    mock_service.__aexit__.return_value = None

    return mock_service


@pytest.mark.asyncio
async def test_list_outputs(client: TestClient, mock_hf_search_response):
    """Test listing all outputs."""
    mock_service = create_mock_service(search_response=mock_hf_search_response)

    with patch("src.dependencies.HuggingFaceService", return_value=mock_service):
        response = client.get("/outputs?hf_app_name=test-app")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 2
        assert data["data"][0]["id"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert "value" in data["data"][0]
        assert data["data"][0]["value"]["repo_id"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert data["data"][0]["value"]["gated"] is True


@pytest.mark.asyncio
async def test_list_outputs_with_filter(client: TestClient, mock_hf_search_response):
    """Test listing outputs with filter."""
    mock_service = create_mock_service(search_response=mock_hf_search_response)

    with patch("src.dependencies.HuggingFaceService", return_value=mock_service):
        response = client.get("/outputs?hf_app_name=test-app&filter=llama")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


@pytest.mark.asyncio
async def test_list_outputs_missing_param(client: TestClient):
    """Test list outputs without hf_app_name param."""
    response = client.get("/outputs")

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_outputs_error_handling(client: TestClient):
    """Test list outputs error handling."""
    mock_service = create_mock_service(should_fail=True)

    with patch("src.dependencies.HuggingFaceService", return_value=mock_service):
        response = client.get("/outputs?hf_app_name=test-app")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert data["data"] is None


@pytest.mark.asyncio
async def test_get_output_detail(client: TestClient, mock_hf_repo_response):
    """Test getting output details for a specific repo."""
    mock_service = create_mock_service(repo_response=mock_hf_repo_response)

    with patch("src.dependencies.HuggingFaceService", return_value=mock_service):
        response = client.get("/outputs/meta-llama/Llama-3.1-8B-Instruct?hf_app_name=test-app")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert data["data"]["repo_id"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert data["data"]["gated"] is True
        assert data["data"]["cached"] is False
        assert "tags" in data["data"]
        assert len(data["data"]["tags"]) > 0


@pytest.mark.asyncio
async def test_get_output_detail_cached_always_false(client: TestClient, mock_hf_repo_response):
    """Test that cached is always False in output details."""
    mock_service = create_mock_service(repo_response=mock_hf_repo_response)

    with patch("src.dependencies.HuggingFaceService", return_value=mock_service):
        response = client.get("/outputs/some-model?hf_app_name=test-app")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"]["cached"] is False


@pytest.mark.asyncio
async def test_get_output_detail_with_slash(client: TestClient, mock_hf_repo_response):
    """Test repo ID with slash is handled correctly."""
    mock_service = create_mock_service(repo_response=mock_hf_repo_response)

    with patch("src.dependencies.HuggingFaceService", return_value=mock_service):
        response = client.get("/outputs/organization/model-name?hf_app_name=test-app")

        assert response.status_code == 200
        mock_service.get_repo_details.assert_called_once_with("organization/model-name")


@pytest.mark.asyncio
async def test_get_output_detail_error(client: TestClient):
    """Test output details error handling."""
    mock_service = create_mock_service(should_fail=True)

    with patch("src.dependencies.HuggingFaceService", return_value=mock_service):
        response = client.get("/outputs/invalid/repo?hf_app_name=test-app")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert data["data"] is None
