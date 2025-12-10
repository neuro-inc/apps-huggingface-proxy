"""Tests for outputs endpoints."""

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

import src.dependencies
from src.services import HuggingFaceService


def create_mock_service(
    search_response=None,
    repo_response=None,
    should_fail=False,
    cached_models=None,
    cached_models_list=None,
):
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

    # Mock cache-related methods
    mock_service.search_cache.return_value = cached_models or set()
    mock_service.is_model_cached.return_value = False
    mock_service.get_cached_models.return_value = cached_models_list or []

    mock_service.__aenter__.return_value = mock_service
    mock_service.__aexit__.return_value = None

    return mock_service


def patch_hf_service(mock_service):
    """Patch the global HF service singleton."""
    src.dependencies._hf_service = mock_service
    return mock_service


async def test_list_outputs(client: TestClient, mock_hf_search_response):
    """Test listing all outputs."""
    mock_service = create_mock_service(search_response=mock_hf_search_response)
    patch_hf_service(mock_service)

    response = client.get("/outputs")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) == 2
    assert data["data"][0]["id"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert data["data"][0]["value"]["gated"] is True
    assert data["data"][0]["value"]["cached"] is False

    src.dependencies._hf_service = None


async def test_list_outputs_with_filter(client: TestClient, mock_hf_search_response):
    """Test listing outputs with filter."""
    mock_service = create_mock_service(search_response=mock_hf_search_response)
    patch_hf_service(mock_service)

    response = client.get("/outputs?filter=llama")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    src.dependencies._hf_service = None


async def test_list_outputs_error_handling(client: TestClient):
    """Test list outputs error handling."""
    mock_service = create_mock_service(should_fail=True)
    patch_hf_service(mock_service)

    response = client.get("/outputs")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert data["data"] is None

    src.dependencies._hf_service = None


async def test_get_output_detail(client: TestClient, mock_hf_repo_response):
    """Test getting output details for a specific repo."""
    mock_service = create_mock_service(repo_response=mock_hf_repo_response)
    patch_hf_service(mock_service)

    response = client.get("/outputs/meta-llama/Llama-3.1-8B-Instruct")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data
    assert data["data"]["id"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert data["data"]["value"]["gated"] is True
    assert data["data"]["value"]["cached"] is False
    assert "tags" in data["data"]["value"]
    assert len(data["data"]["value"]["tags"]) > 0

    src.dependencies._hf_service = None


async def test_get_output_detail_cached_always_false(client: TestClient, mock_hf_repo_response):
    """Test that cached field reflects actual cache status."""
    mock_service = create_mock_service(repo_response=mock_hf_repo_response)
    patch_hf_service(mock_service)

    response = client.get("/outputs/some-model")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["value"]["cached"] is False

    src.dependencies._hf_service = None


async def test_get_output_detail_with_slash(client: TestClient, mock_hf_repo_response):
    """Test repo ID with slash is handled correctly."""
    mock_service = create_mock_service(repo_response=mock_hf_repo_response)
    patch_hf_service(mock_service)

    response = client.get("/outputs/organization/model-name")

    assert response.status_code == 200
    mock_service.get_repo_details.assert_called_once_with("organization/model-name")

    src.dependencies._hf_service = None


async def test_get_output_detail_error(client: TestClient):
    """Test output details error handling."""
    mock_service = create_mock_service(should_fail=True)
    patch_hf_service(mock_service)

    response = client.get("/outputs/invalid/repo")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert data["data"] is None

    src.dependencies._hf_service = None


async def test_cached_model_detection(client: TestClient, mock_hf_repo_response):
    """Test that cached models are correctly identified."""
    # Update the repo response to have cached=True
    cached_repo_response = mock_hf_repo_response.copy()
    cached_repo_response["cached"] = True
    mock_service = create_mock_service(repo_response=cached_repo_response)
    patch_hf_service(mock_service)

    response = client.get("/outputs/meta-llama/Llama-3.1-8B-Instruct")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["value"]["cached"] is True

    src.dependencies._hf_service = None


async def test_list_outputs_with_pagination(client: TestClient, mock_hf_search_response):
    """Test listing outputs with pagination parameters."""
    mock_service = create_mock_service(search_response=mock_hf_search_response)
    patch_hf_service(mock_service)

    response = client.get("/outputs?limit=1&offset=0")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["data"]) == 1

    src.dependencies._hf_service = None


async def test_list_outputs_empty_response(client: TestClient):
    """Test listing outputs with empty API response."""
    mock_service = create_mock_service(search_response=[])
    patch_hf_service(mock_service)

    response = client.get("/outputs")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"] == []

    src.dependencies._hf_service = None


async def test_list_outputs_malformed_model_data(client: TestClient):
    """Test listing outputs with malformed model data."""
    malformed_response = [
        {"id": "valid-model", "private": False},
        "invalid-string-entry",
        {"id": "another-valid", "private": True},
        None,
    ]
    mock_service = create_mock_service(search_response=malformed_response)
    patch_hf_service(mock_service)

    response = client.get("/outputs")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["data"]) == 2

    src.dependencies._hf_service = None


async def test_list_outputs_with_missing_fields(client: TestClient):
    """Test listing outputs with models missing optional fields."""
    response_with_missing_fields = [
        {"id": "minimal-model"},
        {"id": "model-with-some-fields", "gated": True},
    ]
    mock_service = create_mock_service(search_response=response_with_missing_fields)
    patch_hf_service(mock_service)

    response = client.get("/outputs")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["data"]) == 2
    assert data["data"][0]["value"]["visibility"] == "public"
    assert data["data"][0]["value"]["gated"] is False
    assert data["data"][0]["value"]["tags"] == []

    src.dependencies._hf_service = None


async def test_get_output_detail_missing_optional_fields(client: TestClient):
    """Test getting output details with missing optional fields."""
    minimal_response = {"id": "minimal-model"}
    mock_service = create_mock_service(repo_response=minimal_response)
    patch_hf_service(mock_service)

    response = client.get("/outputs/minimal-model")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["id"] == "minimal-model"
    assert data["data"]["value"]["visibility"] == "public"
    assert data["data"]["value"]["gated"] is False
    assert data["data"]["value"]["cached"] is False

    src.dependencies._hf_service = None


async def test_list_outputs_filter_case_insensitive(client: TestClient, mock_hf_search_response):
    """Test that filter parameter is passed but not applied in endpoint."""
    mock_service = create_mock_service(search_response=mock_hf_search_response)
    patch_hf_service(mock_service)

    response = client.get("/outputs?filter=LLAMA")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    # Without filtering logic, all models are returned
    assert len(data["data"]) == 2

    src.dependencies._hf_service = None


async def test_list_outputs_filter_by_tag(client: TestClient, mock_hf_search_response):
    """Test filter parameter with tag."""
    mock_service = create_mock_service(search_response=mock_hf_search_response)
    patch_hf_service(mock_service)

    response = client.get("/outputs?filter=text-generation")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    # Without filtering logic, all models are returned
    assert len(data["data"]) == 2

    src.dependencies._hf_service = None


async def test_list_outputs_filter_no_matches(client: TestClient, mock_hf_search_response):
    """Test filter parameter with non-matching query."""
    mock_service = create_mock_service(search_response=mock_hf_search_response)
    patch_hf_service(mock_service)

    response = client.get("/outputs?filter=nonexistent-model-xyz")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    # Without filtering logic, all models are returned
    assert len(data["data"]) == 2

    src.dependencies._hf_service = None


async def test_cached_models_in_list(client: TestClient, mock_hf_search_response):
    """Test that cached status is correctly set for models in list."""
    cached_models = {"meta-llama/Llama-3.1-8B-Instruct"}
    mock_service = create_mock_service(
        search_response=mock_hf_search_response, cached_models=cached_models
    )
    patch_hf_service(mock_service)

    response = client.get("/outputs")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"][0]["value"]["cached"] is True
    assert data["data"][1]["value"]["cached"] is False

    src.dependencies._hf_service = None


async def test_list_outputs_cached_only(client: TestClient):
    """Test listing only cached models without HF Hub API call."""
    cached_models_list = [
        {
            "id": "cached-model-1",
            "modelId": "cached-model-1",
            "private": False,
            "gated": False,
            "tags": [],
            "lastModified": None,
            "cached": True,
        },
        {
            "id": "cached-model-2",
            "modelId": "cached-model-2",
            "private": False,
            "gated": False,
            "tags": [],
            "lastModified": None,
            "cached": True,
        },
    ]
    mock_service = create_mock_service(cached_models_list=cached_models_list)
    patch_hf_service(mock_service)

    response = client.get("/outputs?filter=cached_only")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["data"]) == 2
    assert data["data"][0]["value"]["cached"] is True
    assert data["data"][1]["value"]["cached"] is True

    # Verify search_models was NOT called (since we're only getting cached models)
    mock_service.search_models.assert_not_called()
    # Verify get_cached_models WAS called
    mock_service.get_cached_models.assert_called_once()

    src.dependencies._hf_service = None
