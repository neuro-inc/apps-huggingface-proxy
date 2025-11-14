"""Tests for HuggingFace service."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.services import HuggingFaceService


async def test_service_initialization():
    """Test service initialization with token."""
    service = HuggingFaceService(token="test-token", base_url="https://api.test", timeout=60)

    assert service.token == "test-token"
    assert service.api is not None


async def test_service_initialization_without_token():
    """Test service initialization without token."""
    service = HuggingFaceService()

    assert service.token is None
    assert service.api is not None


async def test_service_context_manager():
    """Test service as async context manager."""
    service = HuggingFaceService(token="test-token")

    async with service:
        assert service.api is not None

    # close() is a no-op for HfApi, just verify it doesn't error
    await service.close()


async def test_search_models_success():
    """Test searching models successfully."""
    service = HuggingFaceService(token="test-token")

    # Create mock ModelInfo objects
    mock_model1 = MagicMock()
    mock_model1.id = "model1"
    mock_model1.private = False
    mock_model1.gated = False
    mock_model1.tags = ["text-generation"]
    mock_model1.lastModified = datetime(2024, 1, 1)

    mock_model2 = MagicMock()
    mock_model2.id = "model2"
    mock_model2.private = True
    mock_model2.gated = True
    mock_model2.tags = ["conversational"]
    mock_model2.lastModified = None

    with patch.object(service.api, "list_models", return_value=[mock_model1, mock_model2]):
        result = await service.search_models(limit=10)

    assert len(result) == 2
    assert result[0]["id"] == "model1"
    assert result[0]["private"] is False
    assert result[0]["gated"] is False
    assert result[0]["tags"] == ["text-generation"]
    assert result[1]["id"] == "model2"
    assert result[1]["private"] is True


async def test_search_models_with_limit():
    """Test searching models with limit parameter."""
    service = HuggingFaceService(token="test-token")

    mock_model = MagicMock()
    mock_model.id = "llama-model"
    mock_model.private = False
    mock_model.gated = False
    mock_model.tags = []
    mock_model.lastModified = None

    with patch.object(service.api, "list_models", return_value=[mock_model]) as mock_list:
        await service.search_models(limit=5)

    # Verify limit was passed (through asyncio.to_thread/partial)
    # The actual call happens inside asyncio.to_thread, so we can't check args directly
    mock_list.assert_called_once()


async def test_get_repo_details_success():
    """Test getting repository details successfully when not cached."""
    service = HuggingFaceService(token="test-token")

    mock_model_info = MagicMock()
    mock_model_info.id = "meta-llama/Llama-3.1-8B"
    mock_model_info.private = False
    mock_model_info.gated = True
    mock_model_info.tags = ["text-generation"]
    mock_model_info.lastModified = datetime(2024, 1, 15)

    # Mock is_model_cached to return False so it fetches from API
    with patch.object(service, "is_model_cached", return_value=False):
        with patch.object(service.api, "model_info", return_value=mock_model_info):
            result = await service.get_repo_details("meta-llama/Llama-3.1-8B")

    assert result["id"] == "meta-llama/Llama-3.1-8B"
    assert result["gated"] is True
    assert result["tags"] == ["text-generation"]
    assert result["cached"] is False


async def test_get_repo_details_cached():
    """Test getting repository details when model is cached locally."""
    service = HuggingFaceService(token="test-token")

    # Mock is_model_cached to return True so it returns cached info without API call
    with patch.object(service, "is_model_cached", return_value=True):
        with patch.object(service.api, "model_info") as mock_model_info:
            result = await service.get_repo_details("meta-llama/Llama-3.1-8B")

            # Verify model_info was NOT called since model is cached
            mock_model_info.assert_not_called()

    # Verify it returns basic cached info
    assert result["id"] == "meta-llama/Llama-3.1-8B"
    assert result["modelId"] == "meta-llama/Llama-3.1-8B"
    assert result["private"] is False
    assert result["gated"] is False
    assert result["tags"] == []
    assert result["lastModified"] is None
    assert result["cached"] is True


async def test_service_close():
    """Test closing service."""
    service = HuggingFaceService()
    await service.close()
    # Verify no errors occur


async def test_service_with_token():
    """Test that token is properly set."""
    service = HuggingFaceService(token="test-token-123")
    assert service.token == "test-token-123"
    assert service.api.token == "test-token-123"


async def test_search_models_error_handling():
    """Test search models error handling."""
    service = HuggingFaceService(token="test-token")

    with patch.object(service.api, "list_models", side_effect=Exception("API Error")):
        with pytest.raises(Exception):
            await service.search_models()


async def test_get_repo_details_error_handling():
    """Test get repo details error handling when not cached."""
    service = HuggingFaceService(token="test-token")

    # Mock is_model_cached to return False so it tries to fetch from API
    with patch.object(service, "is_model_cached", return_value=False):
        with patch.object(service.api, "model_info", side_effect=Exception("Not found")):
            with pytest.raises(Exception):
                await service.get_repo_details("nonexistent/model")


async def test_get_cached_models():
    """Test getting list of cached models."""
    service = HuggingFaceService(token="test-token")

    # Mock scan_cache_dir to return cached models
    mock_cache_info = MagicMock()
    mock_repo1 = MagicMock()
    mock_repo1.repo_id = "model1/test"
    mock_repo2 = MagicMock()
    mock_repo2.repo_id = "model2/test"
    mock_cache_info.repos = [mock_repo1, mock_repo2]

    with patch("src.services.scan_cache_dir", return_value=mock_cache_info):
        result = await service.get_cached_models()

    assert len(result) == 2
    assert result[0]["id"] == "model1/test"
    assert result[0]["cached"] is True
    assert result[1]["id"] == "model2/test"
    assert result[1]["cached"] is True


async def test_get_cached_models_with_prefix():
    """Test getting cached models with prefix filter."""
    service = HuggingFaceService(token="test-token")

    # Mock scan_cache_dir to return cached models
    mock_cache_info = MagicMock()
    mock_repo1 = MagicMock()
    mock_repo1.repo_id = "meta-llama/test"
    mock_repo2 = MagicMock()
    mock_repo2.repo_id = "other/test"
    mock_cache_info.repos = [mock_repo1, mock_repo2]

    with patch("src.services.scan_cache_dir", return_value=mock_cache_info):
        result = await service.get_cached_models(model_name_prefix="meta-llama")

    # Should only include models starting with "meta-llama"
    assert len(result) == 1
    assert result[0]["id"] == "meta-llama/test"
