"""Tests for HuggingFace Proxy output generation."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from apolo_app_types.outputs.utils.cleanup import cleanup_secrets
from apolo_app_types.outputs.utils.type_search import find_instances_recursive_simple
from apolo_app_types.protocols.common.hugging_face import HuggingFaceToken
from apolo_app_types.protocols.common.secrets_ import ApoloSecret
from apolo_apps_hf_proxy.inputs_processor import HfProxyChartValueProcessor
from apolo_apps_hf_proxy.outputs_processor import HfProxyOutputProcessor
from apolo_apps_hf_proxy.types import HfProxyInputs, HfProxyOutputs

from tests.unit.constants import APP_ID, APP_SECRETS_NAME


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
    assert res["files_path"]["path"] == storage_uri
    assert "token" not in res


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
    assert res["files_path"]["path"] == custom_storage_uri
    assert "token" not in res


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
    assert res["files_path"]["path"] == "storage:.apps/hugging-face-cache"
    assert "token" not in res


@pytest.mark.parametrize("secret_key", ["HF_TOKEN", "MY_HF_TOKEN"])
async def test_hf_proxy_outputs_publish_no_secrets(mock_apolo_client, app_instance_id, secret_key):
    inputs = HfProxyInputs(
        token=HuggingFaceToken(token_name="my-token", token=ApoloSecret(key=secret_key)),
    )
    helm_values = await HfProxyChartValueProcessor(client=mock_apolo_client).gen_extra_values(
        input_=inputs,
        app_name="test-app",
        namespace="test-namespace",
        app_id=app_instance_id,
        app_secrets_name=APP_SECRETS_NAME,
    )

    res = await HfProxyOutputProcessor().generate_outputs(
        helm_values=helm_values,
        app_instance_id=APP_ID,
    )
    outputs = HfProxyOutputs.model_validate(res)

    assert find_instances_recursive_simple(obj=outputs, target_type=ApoloSecret) == []


STORED_OUTPUTS_WITH_TOKEN = {
    "app_url": None,
    "files_path": {
        "__type__": "ApoloFilesPath",
        "path": "storage://default/org/project/hf",
    },
    "token": {
        "token": {"key": "HF_TOKEN", "__type__": "ApoloSecret"},
        "__type__": "HuggingFaceToken",
        "token_name": "hf-token",
    },
    "huggingface_models": [],
}


def test_hf_proxy_stored_outputs_with_token_still_validate():
    outputs = HfProxyOutputs.model_validate(STORED_OUTPUTS_WITH_TOKEN)

    assert outputs.files_path.path == "storage://default/org/project/hf"
    assert "token" not in outputs.model_dump()
    assert find_instances_recursive_simple(obj=outputs, target_type=ApoloSecret) == []


async def test_hf_proxy_cleanup_keeps_token_secret_of_existing_instance():
    client = MagicMock()
    client.apps.get_output = AsyncMock(return_value=STORED_OUTPUTS_WITH_TOKEN)
    client.secrets.rm = AsyncMock()

    await cleanup_secrets(app_id=APP_ID, output_class=HfProxyOutputs, client=client)

    client.apps.get_output.assert_awaited_once_with(app_id=APP_ID)
    client.secrets.rm.assert_not_awaited()
