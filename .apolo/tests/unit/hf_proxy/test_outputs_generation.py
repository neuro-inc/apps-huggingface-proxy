"""Tests for HuggingFace Proxy output generation."""

import json
from unittest.mock import patch

import pytest
from apolo_app_types.protocols.common.hugging_face import HuggingFaceCache, HuggingFaceToken
from apolo_app_types.protocols.common.secrets_ import ApoloSecret
from apolo_app_types.protocols.common.storage import ApoloFilesPath
from apolo_apps_hf_proxy.outputs_processor import HfProxyOutputProcessor
from apolo_apps_hf_proxy.types import HfProxyInputs


@pytest.fixture
def test_inputs():
    """Test inputs for HF Proxy."""
    return HfProxyInputs(
        cache_config=HuggingFaceCache(
            files_path=ApoloFilesPath(path="storage:.apps/hugging-face-cache")
        ),
        token=HuggingFaceToken(token_name="hf-token", token=ApoloSecret(key="HF_TOKEN")),
    )


@pytest.fixture
def helm_values_basic(test_inputs):
    """Basic helm values fixture."""
    storage_config = [
        {
            "storage_uri": test_inputs.cache_config.files_path.path,
            "mount_path": "/root/.cache/huggingface",
            "mount_mode": "rw",
        }
    ]

    return {
        "image": {
            "repository": "hf-proxy",
            "tag": "latest",
        },
        "podAnnotations": {
            "platform.apolo.us/inject-storage": json.dumps(storage_config),
        },
        "hf_token_secret": {
            "name": test_inputs.token.token_name,
            "key": test_inputs.token.token.key,
        },
    }


@pytest.mark.asyncio
async def test_hf_proxy_internal_url_generation(app_instance_id, helm_values_basic, test_inputs):
    """Test internal URL generation."""
    service_host = f"hf-proxy-{app_instance_id}.default.svc.cluster.local"
    service_port = "8080"

    processor = HfProxyOutputProcessor()

    with patch("apolo_apps_hf_proxy.outputs_processor.get_service_host_port") as mock_get_service:
        mock_get_service.return_value = (service_host, service_port)

        outputs = await processor._generate_outputs(helm_values_basic, app_instance_id)

        expected_url = f"http://{service_host}:{service_port}/"
        assert outputs.internal_url == expected_url
        assert outputs.cache_config == test_inputs.cache_config
        assert outputs.token == test_inputs.token

        # Verify get_service_host_port was called with correct labels
        mock_get_service.assert_called_once()
        call_args = mock_get_service.call_args
        labels = call_args.kwargs["match_labels"]
        assert labels["application"] == "hf-proxy"
        assert labels["app.kubernetes.io/instance"] == app_instance_id


@pytest.mark.asyncio
async def test_hf_proxy_no_service_found(app_instance_id, helm_values_basic, test_inputs):
    """Test when no service is found (returns empty string)."""
    # Arrange
    processor = HfProxyOutputProcessor()

    with patch("apolo_apps_hf_proxy.outputs_processor.get_service_host_port") as mock_get_service:
        mock_get_service.return_value = (None, None)

        outputs = await processor._generate_outputs(helm_values_basic, app_instance_id)

        # Assert - Should return empty string when no service found
        assert outputs.internal_url == ""
        assert outputs.cache_config == test_inputs.cache_config
        assert outputs.token == test_inputs.token


@pytest.mark.asyncio
async def test_hf_proxy_output_structure(app_instance_id, helm_values_basic, test_inputs):
    """Test that output structure matches the expected schema."""
    # Arrange
    service_host = f"hf-proxy-{app_instance_id}.default.svc.cluster.local"
    service_port = "8080"

    processor = HfProxyOutputProcessor()

    with patch("apolo_apps_hf_proxy.outputs_processor.get_service_host_port") as mock_get_service:
        mock_get_service.return_value = (service_host, service_port)

        outputs = await processor._generate_outputs(helm_values_basic, app_instance_id)

        assert hasattr(outputs, "internal_url")
        assert hasattr(outputs, "cache_config")
        assert hasattr(outputs, "token")
        assert isinstance(outputs.internal_url, str)
        assert outputs.cache_config == test_inputs.cache_config
        assert outputs.token == test_inputs.token

        assert outputs.internal_url.startswith("http://")
        assert service_host in outputs.internal_url


@pytest.mark.asyncio
async def test_hf_proxy_custom_storage_uri(app_instance_id, test_inputs):
    """Test custom storage URI in helm values."""
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
            "name": test_inputs.token.token_name,
            "key": test_inputs.token.token.key,
        },
    }

    processor = HfProxyOutputProcessor()

    with patch("apolo_apps_hf_proxy.outputs_processor.get_service_host_port") as mock_get_service:
        mock_get_service.return_value = ("hf-proxy.default.svc.cluster.local", "8080")

        # Act
        outputs = await processor._generate_outputs(helm_values, app_instance_id)

        # Assert - Custom storage URI should be in output
        assert outputs.cache_config.files_path.path == custom_storage_uri


@pytest.mark.asyncio
async def test_hf_proxy_default_values_when_missing(app_instance_id):
    """Test that defaults are used when values are missing from helm_values."""
    # Arrange - Minimal helm_values
    helm_values = {}

    processor = HfProxyOutputProcessor()

    with patch("apolo_apps_hf_proxy.outputs_processor.get_service_host_port") as mock_get_service:
        mock_get_service.return_value = ("hf-proxy.default.svc.cluster.local", "8080")

        outputs = await processor._generate_outputs(helm_values, app_instance_id)

        assert outputs.cache_config.files_path.path == "storage:.apps/hugging-face-cache"
        assert outputs.token.token_name == "hf-token"
        assert outputs.token.token.key == "HF_TOKEN"
        assert outputs.internal_url.startswith("http://")
