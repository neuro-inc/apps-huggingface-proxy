"""Tests for HuggingFace Proxy input generation."""

import json

import pytest
from apolo_app_types.protocols.common.hugging_face import HuggingFaceCache, HuggingFaceToken
from apolo_app_types.protocols.common.secrets_ import ApoloSecret
from apolo_app_types.protocols.common.storage import ApoloFilesPath
from apolo_apps_hf_proxy.inputs_processor import HfProxyChartValueProcessor
from apolo_apps_hf_proxy.types import HfProxyInputs

from tests.unit.constants import APP_ID, DEFAULT_CLUSTER_NAME


@pytest.mark.asyncio
async def test_hf_proxy_basic_values_generation(setup_clients, mock_get_preset_cpu):
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
        app_instance_id=APP_ID,
        cluster_domain=f"{DEFAULT_CLUSTER_NAME}.local",
        client=setup_clients,
    )

    # Act
    values = await processor.gen_extra_values()

    # Assert - Image configuration
    assert values["image"]["repository"] == "hf-proxy"
    assert values["image"]["tag"] == "latest"
    assert values["image"]["pullPolicy"] == "Always"

    # Assert - Resource limits from preset (cpu-small: 1.0 CPU, 2GB)
    assert values["resources"]["limits"]["cpu"] == "1000.0m"  # 1.0 * 1000
    assert values["resources"]["limits"]["memory"] == "1907M"  # 2e9 / 1048576
    assert values["resources"]["requests"]["cpu"] == "1000.0m"
    assert values["resources"]["requests"]["memory"] == "1907M"

    # Assert - Tolerations are included
    assert "tolerations" in values
    assert isinstance(values["tolerations"], list)

    # Assert - Affinity is included
    assert "affinity" in values
    assert isinstance(values["affinity"], dict)

    # Assert - Preset name is included
    assert values["preset_name"] == "cpu-small"  # Cheapest viable preset

    # Assert - Pod labels include component and preset
    assert values["podLabels"]["platform.apolo.us/component"] == "app"
    assert values["podLabels"]["platform.apolo.us/preset"] == "cpu-small"
    assert values["podLabels"]["platform.apolo.us/inject-storage"] == "true"
    assert values["podLabels"]["application"] == "hf-proxy"

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
    assert values["apolo_app_id"] == APP_ID


@pytest.mark.asyncio
async def test_hf_proxy_storage_injection_annotations(setup_clients, mock_get_preset_cpu):
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
        app_instance_id=APP_ID,
        cluster_domain=f"{DEFAULT_CLUSTER_NAME}.local",
        client=setup_clients,
    )

    # Act
    values = await processor.gen_extra_values()

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
async def test_hf_proxy_custom_storage_uri(setup_clients, mock_get_preset_cpu):
    """Test custom storage URI configuration."""
    # Arrange
    custom_storage_uri = "storage://default/org/project/custom-hf-cache"
    inputs = HfProxyInputs(
        cache_config=HuggingFaceCache(files_path=ApoloFilesPath(path=custom_storage_uri)),
        token=HuggingFaceToken(token_name="hf-token", token=ApoloSecret(key="HF_TOKEN")),
    )

    processor = HfProxyChartValueProcessor(
        inputs=inputs,
        app_instance_id=APP_ID,
        cluster_domain=f"{DEFAULT_CLUSTER_NAME}.local",
        client=setup_clients,
    )

    # Act
    values = await processor.gen_extra_values()

    # Assert - Custom storage URI is used
    storage_config_str = values["podAnnotations"]["platform.apolo.us/inject-storage"]
    storage_config = json.loads(storage_config_str)

    assert storage_config[0]["storage_uri"] == custom_storage_uri


@pytest.mark.asyncio
async def test_hf_proxy_preset_selection_cheapest(setup_clients, mock_get_preset_cpu):
    """Test that preset auto-selection chooses the cheapest viable preset."""
    # Arrange
    inputs = HfProxyInputs(
        cache_config=HuggingFaceCache(
            files_path=ApoloFilesPath(path="storage:.apps/hugging-face-cache")
        ),
        token=HuggingFaceToken(token_name="hf-token", token=ApoloSecret(key="HF_TOKEN")),
    )

    processor = HfProxyChartValueProcessor(
        inputs=inputs,
        app_instance_id=APP_ID,
        cluster_domain=f"{DEFAULT_CLUSTER_NAME}.local",
        client=setup_clients,
    )

    # Act
    values = await processor.gen_extra_values()

    # Assert - Should select cpu-small (cheapest at 1.0 credits/hour)
    assert values["preset_name"] == "cpu-small"
    assert values["resources"]["limits"]["cpu"] == "1000.0m"  # 1.0 * 1000
    assert values["resources"]["limits"]["memory"] == "1907M"  # 2e9 / 1048576
