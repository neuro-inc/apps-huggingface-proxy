"""Tests for HuggingFace Proxy input generation."""

import json
from unittest.mock import AsyncMock

import pytest
from apolo_app_types.protocols.common.hugging_face import HuggingFaceCache, HuggingFaceToken
from apolo_app_types.protocols.common.secrets_ import ApoloSecret
from apolo_app_types.protocols.common.storage import ApoloFilesPath
from apolo_apps_hf_proxy.inputs_processor import HfProxyChartValueProcessor
from apolo_apps_hf_proxy.types import HfProxyInputs


async def test_hf_proxy_basic_values_generation(setup_clients, app_instance_id, mock_apolo_client):
    """Test basic Helm values generation from user inputs."""
    # Arrange
    inputs = HfProxyInputs(
        cache_config=HuggingFaceCache(
            files_path=ApoloFilesPath(path="storage:.apps/hugging-face-cache")
        ),
        token=HuggingFaceToken(token_name="hf-token", token=ApoloSecret(key="HF_TOKEN")),
    )

    processor = HfProxyChartValueProcessor(client=mock_apolo_client)

    # Act
    values = await processor.gen_extra_values(
        input_=inputs,
        app_name="test-app",
        namespace="test-namespace",
        app_id=app_instance_id,
        app_secrets_name="test-secrets",
    )

    # Assert - Image configuration
    assert values["image"]["repository"] == "ghcr.io/neuro-inc/apps-huggingface-proxy"
    assert values["image"]["tag"] == "latest"
    assert values["image"]["pullPolicy"] == "Always"

    # Assert - Resource limits from preset (cpu-small: 1.0 CPU, 2GB)
    # preset_to_resources formats as: cpu * 1000 + "m", memory / (1<<20) + "M"
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
    assert values["apolo_app_id"] == app_instance_id


@pytest.mark.asyncio
async def test_hf_proxy_storage_injection_annotations(
    setup_clients, app_instance_id, mock_apolo_client
):
    """Test that storage injection annotations are correctly set."""
    # Arrange
    inputs = HfProxyInputs(
        cache_config=HuggingFaceCache(
            files_path=ApoloFilesPath(path="storage:.apps/hugging-face-cache")
        ),
        token=HuggingFaceToken(token_name="hf-token", token=ApoloSecret(key="HF_TOKEN")),
    )

    processor = HfProxyChartValueProcessor(client=mock_apolo_client)

    # Act
    values = await processor.gen_extra_values(
        input_=inputs,
        app_name="test-app",
        namespace="test-namespace",
        app_id=app_instance_id,
        app_secrets_name="test-secrets",
    )

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
    # Helper function expands relative paths to full storage URIs
    assert mount["storage_uri"].endswith(".apps/hugging-face-cache")
    assert mount["storage_uri"].startswith("storage://")
    assert mount["mount_path"] == "/root/.cache/huggingface"
    assert mount["mount_mode"] == "rw"


@pytest.mark.asyncio
async def test_hf_proxy_custom_storage_uri(setup_clients, app_instance_id, mock_apolo_client):
    """Test custom storage URI configuration."""
    # Arrange
    custom_storage_uri = "storage://default/org/project/custom-hf-cache"
    inputs = HfProxyInputs(
        cache_config=HuggingFaceCache(files_path=ApoloFilesPath(path=custom_storage_uri)),
        token=HuggingFaceToken(token_name="hf-token", token=ApoloSecret(key="HF_TOKEN")),
    )

    processor = HfProxyChartValueProcessor(client=mock_apolo_client)

    # Act
    values = await processor.gen_extra_values(
        input_=inputs,
        app_name="test-app",
        namespace="test-namespace",
        app_id=app_instance_id,
        app_secrets_name="test-secrets",
    )

    # Assert - Custom storage URI is used
    storage_config_str = values["podAnnotations"]["platform.apolo.us/inject-storage"]
    storage_config = json.loads(storage_config_str)

    assert storage_config[0]["storage_uri"] == custom_storage_uri


@pytest.mark.asyncio
async def test_hf_proxy_preset_selection_cheapest(
    setup_clients, app_instance_id, mock_apolo_client
):
    """Test that preset auto-selection chooses the cheapest viable preset."""
    # Arrange
    inputs = HfProxyInputs(
        cache_config=HuggingFaceCache(
            files_path=ApoloFilesPath(path="storage:.apps/hugging-face-cache")
        ),
        token=HuggingFaceToken(token_name="hf-token", token=ApoloSecret(key="HF_TOKEN")),
    )

    processor = HfProxyChartValueProcessor(client=mock_apolo_client)

    # Act
    values = await processor.gen_extra_values(
        input_=inputs,
        app_name="test-app",
        namespace="test-namespace",
        app_id=app_instance_id,
        app_secrets_name="test-secrets",
    )

    # Assert - Should select cpu-small (cheapest at 1.0 credits/hour)
    assert values["preset_name"] == "cpu-small"
    assert values["resources"]["limits"]["cpu"] == "1000.0m"  # 1.0 * 1000
    assert values["resources"]["limits"]["memory"] == "1907M"  # 2e9 / 1048576


@pytest.mark.asyncio
async def test_hf_proxy_preset_filters_gpu(setup_clients, app_instance_id, mock_apolo_client):
    """Test that GPU presets are filtered out."""
    # Arrange - Remove all CPU presets except GPU
    mock_apolo_client.config.presets = {
        "gpu-1x-a100": mock_apolo_client.config.presets["gpu-1x-a100"],
    }

    inputs = HfProxyInputs(
        cache_config=HuggingFaceCache(
            files_path=ApoloFilesPath(path="storage:.apps/hugging-face-cache")
        ),
        token=HuggingFaceToken(token_name="hf-token", token=ApoloSecret(key="HF_TOKEN")),
    )

    processor = HfProxyChartValueProcessor(client=mock_apolo_client)

    # Act & Assert - Should raise error (no CPU presets available)
    with pytest.raises(RuntimeError, match="No suitable CPU preset found"):
        await processor.gen_extra_values(
            input_=inputs,
            app_name="test-app",
            namespace="test-namespace",
            app_id=app_instance_id,
            app_secrets_name="test-secrets",
        )


@pytest.mark.asyncio
async def test_hf_proxy_preset_filters_no_capacity(
    setup_clients, app_instance_id, mock_apolo_client
):
    """Test that presets without capacity are filtered out."""

    # Arrange - Set all capacities to 0
    async def mock_get_capacity_zero():
        return {preset: 0 for preset in mock_apolo_client.config.presets}

    mock_apolo_client.jobs.get_capacity = AsyncMock(side_effect=mock_get_capacity_zero)

    inputs = HfProxyInputs(
        cache_config=HuggingFaceCache(
            files_path=ApoloFilesPath(path="storage:.apps/hugging-face-cache")
        ),
        token=HuggingFaceToken(token_name="hf-token", token=ApoloSecret(key="HF_TOKEN")),
    )

    processor = HfProxyChartValueProcessor(client=mock_apolo_client)

    # Act & Assert - Should raise error (no capacity)
    with pytest.raises(RuntimeError, match="No suitable CPU preset found"):
        await processor.gen_extra_values(
            input_=inputs,
            app_name="test-app",
            namespace="test-namespace",
            app_id=app_instance_id,
            app_secrets_name="test-secrets",
        )


@pytest.mark.asyncio
async def test_hf_proxy_preset_filters_insufficient_resources(
    setup_clients, app_instance_id, mock_apolo_client
):
    """Test that presets below minimum requirements are filtered out."""
    # Arrange - Only keep cpu-tiny which is below minimum
    mock_apolo_client.config.presets = {
        "cpu-tiny": mock_apolo_client.config.presets["cpu-tiny"],
    }

    inputs = HfProxyInputs(
        cache_config=HuggingFaceCache(
            files_path=ApoloFilesPath(path="storage:.apps/hugging-face-cache")
        ),
        token=HuggingFaceToken(token_name="hf-token", token=ApoloSecret(key="HF_TOKEN")),
    )

    processor = HfProxyChartValueProcessor(client=mock_apolo_client)

    # Act & Assert - Should raise error (insufficient resources)
    with pytest.raises(RuntimeError, match="No suitable CPU preset found"):
        await processor.gen_extra_values(
            input_=inputs,
            app_name="test-app",
            namespace="test-namespace",
            app_id=app_instance_id,
            app_secrets_name="test-secrets",
        )


@pytest.mark.asyncio
async def test_hf_proxy_preset_prefers_higher_capacity_when_equal_cost(
    setup_clients, app_instance_id, mock_apolo_client
):
    """Test that when costs are equal, preset with more capacity is selected."""
    # Arrange - Make cpu-medium same cost as cpu-small
    from dataclasses import replace
    from decimal import Decimal

    cpu_medium_updated = replace(
        mock_apolo_client.config.presets["cpu-medium"], credits_per_hour=Decimal("1.0")
    )
    mock_apolo_client.config.presets["cpu-medium"] = cpu_medium_updated

    # cpu-small has capacity=5, cpu-medium has capacity=3
    # With same cost, should still prefer cpu-small (more capacity: 5 > 3)

    inputs = HfProxyInputs(
        cache_config=HuggingFaceCache(
            files_path=ApoloFilesPath(path="storage:.apps/hugging-face-cache")
        ),
        token=HuggingFaceToken(token_name="hf-token", token=ApoloSecret(key="HF_TOKEN")),
    )

    processor = HfProxyChartValueProcessor(client=mock_apolo_client)

    # Act
    values = await processor.gen_extra_values(
        input_=inputs,
        app_name="test-app",
        namespace="test-namespace",
        app_id=app_instance_id,
        app_secrets_name="test-secrets",
    )

    # Assert - Should still select cpu-small (higher capacity: 5 > 3)
    assert values["preset_name"] == "cpu-small"
