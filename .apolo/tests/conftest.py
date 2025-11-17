"""Test configuration for HuggingFace Proxy App tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def setup_clients():
    """Setup Apolo clients for testing.

    Note: In a real deployment, this would use apolo_app_types_fixtures.apolo_clients
    For demonstration purposes, we return a mock object.
    """
    return {}


@pytest.fixture
def app_instance_id():
    """Test app instance ID."""
    return "test-hf-proxy-12345"


@pytest.fixture
def cluster_domain():
    """Test cluster domain."""
    return "apolo.dev"


@pytest.fixture
def app_namespace():
    """Test app namespace."""
    return "default"


@pytest.fixture
def mock_apolo_client():
    """Mock Apolo client with presets and capacity."""
    # Create mock presets
    cpu_small = MagicMock()
    cpu_small.cpu = 1.0
    cpu_small.memory = 2 * 1e9  # 2Gi
    cpu_small.nvidia_gpu = None
    cpu_small.amd_gpu = None
    cpu_small.credits_per_hour = 1.0

    cpu_medium = MagicMock()
    cpu_medium.cpu = 2.0
    cpu_medium.memory = 4 * 1e9  # 4Gi
    cpu_medium.nvidia_gpu = None
    cpu_medium.amd_gpu = None
    cpu_medium.credits_per_hour = 2.0

    cpu_large = MagicMock()
    cpu_large.cpu = 4.0
    cpu_large.memory = 8 * 1e9  # 8Gi
    cpu_large.nvidia_gpu = None
    cpu_large.amd_gpu = None
    cpu_large.credits_per_hour = 4.0

    # GPU preset (should be filtered out)
    gpu_preset = MagicMock()
    gpu_preset.cpu = 8.0
    gpu_preset.memory = 16 * 1e9
    gpu_preset.nvidia_gpu = MagicMock()
    gpu_preset.nvidia_gpu.count = 1
    gpu_preset.amd_gpu = None
    gpu_preset.credits_per_hour = 10.0

    # Too small preset (should be filtered out)
    cpu_tiny = MagicMock()
    cpu_tiny.cpu = 0.25
    cpu_tiny.memory = 512 * 1e6  # 512Mi
    cpu_tiny.nvidia_gpu = None
    cpu_tiny.amd_gpu = None
    cpu_tiny.credits_per_hour = 0.5

    # Mock client
    client = MagicMock()
    client.config.presets = {
        "cpu-small": cpu_small,
        "cpu-medium": cpu_medium,
        "cpu-large": cpu_large,
        "gpu-1x-a100": gpu_preset,
        "cpu-tiny": cpu_tiny,
    }

    # Mock capacity - all presets have capacity except cpu-large (for testing)
    async def mock_get_capacity():
        return {
            "cpu-small": 5,
            "cpu-medium": 3,
            "cpu-large": 0,  # No capacity
            "gpu-1x-a100": 2,
            "cpu-tiny": 10,
        }

    client.jobs.get_capacity = AsyncMock(side_effect=mock_get_capacity)

    return client
