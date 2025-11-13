"""Tests for cache utilities."""

import os
import tempfile
from pathlib import Path

import pytest

from src.cache import is_model_cached


def test_is_model_cached_with_valid_cache(tmp_path):
    """Test cache detection with a valid cached model."""
    cache_dir = tmp_path / "cache"
    hub_dir = cache_dir / "hub"
    hub_dir.mkdir(parents=True)

    model_dir = hub_dir / "models--meta-llama--Llama-3.1-8B-Instruct"
    model_dir.mkdir()

    assert is_model_cached("meta-llama/Llama-3.1-8B-Instruct", str(cache_dir)) is True


def test_is_model_cached_model_not_found(tmp_path):
    """Test cache detection when model is not cached."""
    cache_dir = tmp_path / "cache"
    hub_dir = cache_dir / "hub"
    hub_dir.mkdir(parents=True)

    assert is_model_cached("meta-llama/Llama-3.1-8B-Instruct", str(cache_dir)) is False


def test_is_model_cached_cache_dir_not_exists():
    """Test cache detection when cache directory doesn't exist."""
    assert is_model_cached("meta-llama/Llama-3.1-8B-Instruct", "/nonexistent/path") is False


def test_is_model_cached_no_hub_directory(tmp_path):
    """Test cache detection when hub directory doesn't exist."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    assert is_model_cached("meta-llama/Llama-3.1-8B-Instruct", str(cache_dir)) is False


def test_is_model_cached_empty_cache_dir():
    """Test cache detection with empty cache_dir parameter."""
    assert is_model_cached("meta-llama/Llama-3.1-8B-Instruct", "") is False


def test_is_model_cached_none_cache_dir():
    """Test cache detection with None cache_dir parameter."""
    assert is_model_cached("meta-llama/Llama-3.1-8B-Instruct", None) is False


def test_is_model_cached_with_file_instead_of_directory(tmp_path):
    """Test cache detection when model path is a file instead of directory."""
    cache_dir = tmp_path / "cache"
    hub_dir = cache_dir / "hub"
    hub_dir.mkdir(parents=True)

    model_file = hub_dir / "models--meta-llama--Llama-3.1-8B-Instruct"
    model_file.touch()

    assert is_model_cached("meta-llama/Llama-3.1-8B-Instruct", str(cache_dir)) is False


def test_is_model_cached_special_characters(tmp_path):
    """Test cache detection with model names containing special characters."""
    cache_dir = tmp_path / "cache"
    hub_dir = cache_dir / "hub"
    hub_dir.mkdir(parents=True)

    model_dir = hub_dir / "models--organization--model.name-v2"
    model_dir.mkdir()

    assert is_model_cached("organization/model.name-v2", str(cache_dir)) is True


def test_is_model_cached_nested_organization(tmp_path):
    """Test cache detection with nested organization names."""
    cache_dir = tmp_path / "cache"
    hub_dir = cache_dir / "hub"
    hub_dir.mkdir(parents=True)

    model_dir = hub_dir / "models--org--suborg--model"
    model_dir.mkdir()

    # HuggingFace doesn't support nested orgs, but test the replacement logic
    assert is_model_cached("org/suborg--model", str(cache_dir)) is True
