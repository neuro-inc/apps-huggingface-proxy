"""Tests for cache utilities."""

from unittest.mock import MagicMock, patch

from src.cache import is_model_cached


async def test_is_model_cached_with_valid_cache(tmp_path):
    """Test cache detection with a valid cached model."""
    cache_dir = str(tmp_path / "cache")

    # Mock scan_cache_dir to return a cache with the model
    mock_cache_info = MagicMock()
    mock_repo = MagicMock()
    mock_repo.repo_id = "meta-llama/Llama-3.1-8B-Instruct"
    mock_cache_info.repos = [mock_repo]

    with patch("src.cache.scan_cache_dir", return_value=mock_cache_info):
        assert await is_model_cached("meta-llama/Llama-3.1-8B-Instruct", cache_dir) is True


async def test_is_model_cached_model_not_found(tmp_path):
    """Test cache detection when model is not cached."""
    cache_dir = str(tmp_path / "cache")

    # Mock scan_cache_dir to return empty cache
    mock_cache_info = MagicMock()
    mock_cache_info.repos = []

    with patch("src.cache.scan_cache_dir", return_value=mock_cache_info):
        assert await is_model_cached("meta-llama/Llama-3.1-8B-Instruct", cache_dir) is False


async def test_is_model_cached_cache_dir_not_exists():
    """Test cache detection when cache directory doesn't exist."""
    # scan_cache_dir will raise an exception for non-existent directory
    with patch("src.cache.scan_cache_dir", side_effect=Exception("Cache dir not found")):
        result = await is_model_cached("meta-llama/Llama-3.1-8B-Instruct", "/nonexistent/path")
        assert result is False


async def test_is_model_cached_no_hub_directory(tmp_path):
    """Test cache detection when hub directory doesn't exist."""
    cache_dir = str(tmp_path / "cache")

    # scan_cache_dir will raise an exception for invalid cache structure
    with patch("src.cache.scan_cache_dir", side_effect=Exception("No hub directory")):
        assert await is_model_cached("meta-llama/Llama-3.1-8B-Instruct", cache_dir) is False


async def test_is_model_cached_empty_cache_dir():
    """Test cache detection with empty cache_dir parameter."""
    assert await is_model_cached("meta-llama/Llama-3.1-8B-Instruct", "") is False


async def test_is_model_cached_none_cache_dir():
    """Test cache detection with None cache_dir parameter."""
    assert await is_model_cached("meta-llama/Llama-3.1-8B-Instruct", None) is False


async def test_is_model_cached_with_file_instead_of_directory(tmp_path):
    """Test cache detection when model path is a file instead of directory."""
    cache_dir = str(tmp_path / "cache")

    # scan_cache_dir will handle invalid structures, return empty
    mock_cache_info = MagicMock()
    mock_cache_info.repos = []

    with patch("src.cache.scan_cache_dir", return_value=mock_cache_info):
        assert await is_model_cached("meta-llama/Llama-3.1-8B-Instruct", cache_dir) is False


async def test_is_model_cached_special_characters(tmp_path):
    """Test cache detection with model names containing special characters."""
    cache_dir = str(tmp_path / "cache")

    # Mock scan_cache_dir to return a cache with the model
    mock_cache_info = MagicMock()
    mock_repo = MagicMock()
    mock_repo.repo_id = "organization/model.name-v2"
    mock_cache_info.repos = [mock_repo]

    with patch("src.cache.scan_cache_dir", return_value=mock_cache_info):
        assert await is_model_cached("organization/model.name-v2", cache_dir) is True


async def test_is_model_cached_nested_organization(tmp_path):
    """Test cache detection with nested organization names."""
    cache_dir = str(tmp_path / "cache")

    # Mock scan_cache_dir to return a cache with the model
    mock_cache_info = MagicMock()
    mock_repo = MagicMock()
    mock_repo.repo_id = "org/suborg--model"
    mock_cache_info.repos = [mock_repo]

    with patch("src.cache.scan_cache_dir", return_value=mock_cache_info):
        assert await is_model_cached("org/suborg--model", cache_dir) is True
