from unittest.mock import AsyncMock, MagicMock

import pytest

pytest_plugins = [
    "apolo_app_types_fixtures.apolo_clients",
    "apolo_app_types_fixtures.constants",
]


@pytest.fixture
def cluster_domain():
    """Cluster domain for testing."""
    return "example.cluster.local"


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
    cpu_small.available_resource_pool_names = ["default"]

    cpu_medium = MagicMock()
    cpu_medium.cpu = 2.0
    cpu_medium.memory = 4 * 1e9  # 4Gi
    cpu_medium.nvidia_gpu = None
    cpu_medium.amd_gpu = None
    cpu_medium.credits_per_hour = 2.0
    cpu_medium.available_resource_pool_names = ["default"]

    cpu_large = MagicMock()
    cpu_large.cpu = 4.0
    cpu_large.memory = 8 * 1e9  # 8Gi
    cpu_large.nvidia_gpu = None
    cpu_large.amd_gpu = None
    cpu_large.credits_per_hour = 4.0
    cpu_large.available_resource_pool_names = ["default"]

    # GPU preset (should be filtered out)
    gpu_preset = MagicMock()
    gpu_preset.cpu = 8.0
    gpu_preset.memory = 16 * 1e9
    gpu_preset.nvidia_gpu = MagicMock()
    gpu_preset.nvidia_gpu.count = 1
    gpu_preset.amd_gpu = None
    gpu_preset.credits_per_hour = 10.0
    gpu_preset.available_resource_pool_names = ["gpu-pool"]

    # Too small preset (should be filtered out)
    cpu_tiny = MagicMock()
    cpu_tiny.cpu = 0.25
    cpu_tiny.memory = 512 * 1e6  # 512Mi
    cpu_tiny.nvidia_gpu = None
    cpu_tiny.amd_gpu = None
    cpu_tiny.credits_per_hour = 0.5
    cpu_tiny.available_resource_pool_names = ["default"]

    # Mock client
    client = MagicMock()
    client.config.presets = {
        "cpu-small": cpu_small,
        "cpu-medium": cpu_medium,
        "cpu-large": cpu_large,
        "gpu-1x-a100": gpu_preset,
        "cpu-tiny": cpu_tiny,
    }

    # Mock capacity
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
