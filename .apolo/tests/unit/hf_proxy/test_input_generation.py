"""Tests for HuggingFace Proxy input generation."""

import json

import pytest
from apolo_app_types.protocols.common.hugging_face import HuggingFaceCache, HuggingFaceToken
from apolo_app_types.protocols.common.secrets_ import ApoloSecret
from apolo_app_types.protocols.common.storage import ApoloFilesPath
from apolo_apps_hf_proxy.inputs_processor import HfProxyChartValueProcessor
from apolo_apps_hf_proxy.types import HfProxyInputs


@pytest.mark.asyncio
async def test_hf_proxy_basic_values_generation(setup_clients, app_instance_id, cluster_domain):
    """Test basic Helm values generation from user inputs."""
    # Arrange
    inputs = HfProxyInputs(
        cache_config=HuggingFaceCache(
            files_path=ApoloFilesPath(path="storage:.apps/hugging-face-cache")
        ),
        token=HuggingFaceToken(token_name="hf-token", token=ApoloSecret(key="HF_TOKEN")),
    )

    processor = HfProxyChartValueProcessor(
        inputs=inputs,
        app_instance_id=app_instance_id,
        cluster_domain=cluster_domain,
    )

    # Act
    values = processor.gen_extra_values()

    # Assert - Image configuration
    assert values["image"]["repository"] == "hf-proxy"
    assert values["image"]["tag"] == "latest"
    assert values["image"]["pullPolicy"] == "Always"

    # Assert - Resource limits (minimal hardcoded)
    assert values["resources"]["limits"]["cpu"] == "0.5"
    assert values["resources"]["limits"]["memory"] == "1Gi"
    assert values["resources"]["requests"]["cpu"] == "0.25"
    assert values["resources"]["requests"]["memory"] == "512Mi"

    # Assert - Service configuration
    assert values["service"]["type"] == "ClusterIP"
    assert values["service"]["port"] == 8080

    # Assert - HF token secret
    assert values["hf_token_secret"]["name"] == "hf-token"
    assert values["hf_token_secret"]["key"] == "HF_TOKEN"

    # Assert - Environment variables
    assert values["env"]["HF_TIMEOUT"] == "30"
    assert values["env"]["HF_CACHE_DIR"] == "/root/.cache/huggingface"
    assert values["env"]["PORT"] == "8080"

    # Assert - Apolo app ID
    assert values["apolo_app_id"] == app_instance_id


@pytest.mark.asyncio
async def test_hf_proxy_storage_injection_annotations(
    setup_clients, app_instance_id, cluster_domain
):
    """Test that storage injection annotations are correctly set."""
    # Arrange
    inputs = HfProxyInputs(
        cache_config=HuggingFaceCache(
            files_path=ApoloFilesPath(path="storage:.apps/hugging-face-cache")
        ),
        token=HuggingFaceToken(token_name="hf-token", token=ApoloSecret(key="HF_TOKEN")),
    )

    processor = HfProxyChartValueProcessor(
        inputs=inputs,
        app_instance_id=app_instance_id,
        cluster_domain=cluster_domain,
    )

    # Act
    values = processor.gen_extra_values()

    # Assert - Pod labels include storage injection flag
    assert "platform.apolo.us/inject-storage" in values["podLabels"]
    assert values["podLabels"]["platform.apolo.us/inject-storage"] == "true"
    assert values["podLabels"]["application"] == "hf-proxy"

    # Assert - Pod annotations include storage configuration
    assert "platform.apolo.us/inject-storage" in values["podAnnotations"]

    # Parse storage config from annotation
    storage_config_str = values["podAnnotations"]["platform.apolo.us/inject-storage"]
    storage_config = json.loads(storage_config_str)

    assert isinstance(storage_config, list)
    assert len(storage_config) == 1

    # Assert storage mount configuration
    mount = storage_config[0]
    assert mount["storage_uri"] == "storage:.apps/hugging-face-cache"
    assert mount["mount_path"] == "/root/.cache/huggingface"
    assert mount["mount_mode"] == "rw"


@pytest.mark.asyncio
async def test_hf_proxy_custom_storage_uri(setup_clients, app_instance_id, cluster_domain):
    """Test custom storage URI configuration."""
    # Arrange
    custom_storage_uri = "storage://default/org/project/custom-hf-cache"
    inputs = HfProxyInputs(
        cache_config=HuggingFaceCache(files_path=ApoloFilesPath(path=custom_storage_uri)),
        token=HuggingFaceToken(token_name="hf-token", token=ApoloSecret(key="HF_TOKEN")),
    )

    processor = HfProxyChartValueProcessor(
        inputs=inputs,
        app_instance_id=app_instance_id,
        cluster_domain=cluster_domain,
    )

    # Act
    values = processor.gen_extra_values()

    # Assert - Custom storage URI is used
    storage_config_str = values["podAnnotations"]["platform.apolo.us/inject-storage"]
    storage_config = json.loads(storage_config_str)

    assert storage_config[0]["storage_uri"] == custom_storage_uri
