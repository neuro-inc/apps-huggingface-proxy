"""Tests for HuggingFace Proxy output generation."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from apolo_app_types.protocols.common.hugging_face import HuggingFaceCache, HuggingFaceToken
from apolo_app_types.protocols.common.secrets_ import ApoloSecret
from apolo_app_types.protocols.common.storage import ApoloFilesPath
from apolo_apps_hf_proxy.outputs_processor import HfProxyOutputProcessor
from apolo_apps_hf_proxy.types import HfProxyInputs


@pytest.fixture
def mock_kubernetes_client():
    """Mock Kubernetes client for testing."""
    client = AsyncMock()
    return client


@pytest.fixture
def test_inputs():
    """Test inputs for HF Proxy."""
    return HfProxyInputs(
        cache_config=HuggingFaceCache(
            files_path=ApoloFilesPath(path="storage:.apps/hugging-face-cache")
        ),
        token=HuggingFaceToken(token_name="hf-token", token=ApoloSecret(key="HF_TOKEN")),
    )


@pytest.mark.asyncio
async def test_hf_proxy_internal_url_generation(
    mock_kubernetes_client, app_instance_id, app_namespace, test_inputs
):
    """Test internal URL generation."""
    # Arrange
    service_name = f"hf-proxy-{app_instance_id}"

    # Mock service list response
    mock_service = MagicMock()
    mock_service.metadata.name = service_name

    mock_service_list = MagicMock()
    mock_service_list.items = [mock_service]

    mock_kubernetes_client.list_namespaced_service = AsyncMock(return_value=mock_service_list)

    processor = HfProxyOutputProcessor(
        k8s_client=mock_kubernetes_client,
        app_instance_id=app_instance_id,
        app_namespace=app_namespace,
        inputs=test_inputs,
    )

    # Act
    outputs = await processor.gen_outputs()

    # Assert
    expected_url = f"http://{service_name}.{app_namespace}.svc.cluster.local:8080"
    assert outputs.internal_url == expected_url
    assert outputs.cache_config == test_inputs.cache_config
    assert outputs.token == test_inputs.token

    # Verify Kubernetes client was called correctly
    mock_kubernetes_client.list_namespaced_service.assert_called_once()
    call_kwargs = mock_kubernetes_client.list_namespaced_service.call_args.kwargs
    assert call_kwargs["namespace"] == app_namespace
    assert f"app.kubernetes.io/instance={app_instance_id}" in call_kwargs["label_selector"]
    assert "output-server=true" in call_kwargs["label_selector"]


@pytest.mark.asyncio
async def test_hf_proxy_fallback_service_name(
    mock_kubernetes_client, app_instance_id, app_namespace, test_inputs
):
    """Test fallback to default service name when service not found."""
    # Arrange
    # Mock empty service list
    mock_service_list = MagicMock()
    mock_service_list.items = []

    mock_kubernetes_client.list_namespaced_service = AsyncMock(return_value=mock_service_list)

    processor = HfProxyOutputProcessor(
        k8s_client=mock_kubernetes_client,
        app_instance_id=app_instance_id,
        app_namespace=app_namespace,
        inputs=test_inputs,
    )

    # Act
    outputs = await processor.gen_outputs()

    # Assert - Should use fallback service name
    expected_service_name = f"hf-proxy-{app_instance_id}"
    expected_url = f"http://{expected_service_name}.{app_namespace}.svc.cluster.local:8080"
    assert outputs.internal_url == expected_url
    assert outputs.cache_config == test_inputs.cache_config
    assert outputs.token == test_inputs.token


@pytest.mark.asyncio
async def test_hf_proxy_output_structure(
    mock_kubernetes_client, app_instance_id, app_namespace, test_inputs
):
    """Test that output structure matches the expected schema."""
    # Arrange
    service_name = f"hf-proxy-{app_instance_id}"

    # Mock service
    mock_service = MagicMock()
    mock_service.metadata.name = service_name

    mock_service_list = MagicMock()
    mock_service_list.items = [mock_service]

    mock_kubernetes_client.list_namespaced_service = AsyncMock(return_value=mock_service_list)

    processor = HfProxyOutputProcessor(
        k8s_client=mock_kubernetes_client,
        app_instance_id=app_instance_id,
        app_namespace=app_namespace,
        inputs=test_inputs,
    )

    # Act
    outputs = await processor.gen_outputs()

    # Assert - Check structure
    assert hasattr(outputs, "internal_url")
    assert hasattr(outputs, "cache_config")
    assert hasattr(outputs, "token")
    assert isinstance(outputs.internal_url, str)
    assert outputs.cache_config == test_inputs.cache_config
    assert outputs.token == test_inputs.token

    # Verify internal URL format
    assert outputs.internal_url.startswith("http://")
    assert ".svc.cluster.local:8080" in outputs.internal_url
