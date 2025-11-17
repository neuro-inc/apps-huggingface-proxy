"""Tests for HuggingFace Proxy output generation."""

import json

import pytest
from apolo_apps_hf_proxy.outputs_processor import HfProxyOutputProcessor

from tests.unit.constants import APP_ID


@pytest.mark.asyncio
async def test_hf_proxy_outputs(setup_clients, mock_kubernetes_client):
    """Test output generation with basic helm values."""
    # Arrange
    storage_uri = "storage:.apps/hugging-face-cache"
    storage_config = [
        {
            "storage_uri": storage_uri,
            "mount_path": "/root/.cache/huggingface",
            "mount_mode": "rw",
        }
    ]

    helm_values = {
        "podAnnotations": {
            "platform.apolo.us/inject-storage": json.dumps(storage_config),
        },
        "hf_token_secret": {
            "name": "hf-token",
            "key": "HF_TOKEN",
        },
    }

    # Act
    res = await HfProxyOutputProcessor().generate_outputs(
        helm_values=helm_values,
        app_instance_id=APP_ID,
    )

    # Assert
    assert res["cache_config"]["files_path"]["path"] == storage_uri
    assert res["token"]["token_name"] == "hf-token"
    assert res["token"]["token"]["key"] == "HF_TOKEN"
    assert res["internal_url"] == "http://app.default-namespace:80/"


@pytest.mark.asyncio
async def test_hf_proxy_outputs_custom_storage(setup_clients, mock_kubernetes_client):
    """Test output generation with custom storage URI."""
    # Arrange
    custom_storage_uri = "storage://default/org/project/custom-hf-cache"
    storage_config = [
        {
            "storage_uri": custom_storage_uri,
            "mount_path": "/root/.cache/huggingface",
            "mount_mode": "rw",
        }
    ]

    helm_values = {
        "podAnnotations": {
            "platform.apolo.us/inject-storage": json.dumps(storage_config),
        },
        "hf_token_secret": {
            "name": "custom-token",
            "key": "CUSTOM_HF_TOKEN",
        },
    }

    # Act
    res = await HfProxyOutputProcessor().generate_outputs(
        helm_values=helm_values,
        app_instance_id=APP_ID,
    )

    # Assert
    assert res["cache_config"]["files_path"]["path"] == custom_storage_uri
    assert res["token"]["token_name"] == "custom-token"
    assert res["token"]["token"]["key"] == "CUSTOM_HF_TOKEN"


@pytest.mark.asyncio
async def test_hf_proxy_outputs_with_defaults(setup_clients, mock_kubernetes_client):
    """Test output generation with minimal helm values (uses defaults)."""
    # Arrange - Minimal helm values
    helm_values = {}

    # Act
    res = await HfProxyOutputProcessor().generate_outputs(
        helm_values=helm_values,
        app_instance_id=APP_ID,
    )

    # Assert - Should use default values
    assert res["cache_config"]["files_path"]["path"] == "storage:.apps/hugging-face-cache"
    assert res["token"]["token_name"] == "hf-token"
    assert res["token"]["token"]["key"] == "HF_TOKEN"
    assert res["internal_url"] == "http://app.default-namespace:80/"
