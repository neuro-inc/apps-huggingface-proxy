"""Tests for HuggingFace service."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.services import HuggingFaceService


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx client."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


async def test_service_initialization():
    """Test service initialization with token."""
    service = HuggingFaceService(token="test-token", base_url="https://api.test", timeout=60)

    assert service.token == "test-token"
    assert service.base_url == "https://api.test"
    assert service.timeout == 60
    assert service._client is None


async def test_service_initialization_without_token():
    """Test service initialization without token."""
    service = HuggingFaceService()

    assert service.token is None
    assert service.base_url == "https://huggingface.co/api"
    assert service.timeout == 30


async def test_service_context_manager():
    """Test service as async context manager."""
    service = HuggingFaceService(token="test-token")

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        async with service:
            # Client is lazy-loaded, so get it to trigger initialization
            await service.get_client()
            assert service._client is not None

        mock_client.aclose.assert_called_once()


async def test_search_models_success():
    """Test searching models successfully."""
    service = HuggingFaceService(token="test-token")

    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"id": "model1", "private": False},
        {"id": "model2", "private": True}
    ]

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        async with service:
            result = await service.search_models(limit=10)

        assert len(result) == 2
        assert result[0]["id"] == "model1"
        mock_client.get.assert_called_once()


async def test_search_models_with_limit():
    """Test searching models with limit parameter."""
    service = HuggingFaceService(token="test-token")

    mock_response = MagicMock()
    mock_response.json.return_value = [{"id": "llama-model"}]

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        async with service:
            await service.search_models(limit=5)

        call_args = mock_client.get.call_args
        assert call_args[1].get("params", {}).get("limit") == 5


async def test_get_repo_details_success():
    """Test getting repository details successfully."""
    service = HuggingFaceService(token="test-token")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "meta-llama/Llama-3.1-8B",
        "private": False,
        "gated": True,
        "tags": ["text-generation"]
    }

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        async with service:
            result = await service.get_repo_details("meta-llama/Llama-3.1-8B")

        assert result["id"] == "meta-llama/Llama-3.1-8B"
        assert result["gated"] is True


async def test_service_close_without_client():
    """Test closing service when client was never initialized."""
    service = HuggingFaceService()
    await service.close()

    assert service._client is None


async def test_service_with_authentication_headers():
    """Test that authentication headers are properly set."""
    service = HuggingFaceService(token="test-token-123")

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        async with service:
            # Trigger client creation
            await service.get_client()

            call_kwargs = mock_client_class.call_args[1]
            assert "headers" in call_kwargs
            assert call_kwargs["headers"].get("Authorization") == "Bearer test-token-123"


async def test_search_models_error_handling():
    """Test search models error handling."""
    service = HuggingFaceService(token="test-token")

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.HTTPError("Network error")
        mock_client_class.return_value = mock_client

        async with service:
            with pytest.raises(httpx.HTTPError):
                await service.search_models()


async def test_get_repo_details_error_handling():
    """Test get repo details error handling."""
    service = HuggingFaceService(token="test-token")

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.HTTPError("Not found")
        mock_client_class.return_value = mock_client

        async with service:
            with pytest.raises(httpx.HTTPError):
                await service.get_repo_details("nonexistent/model")
