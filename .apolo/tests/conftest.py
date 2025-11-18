from unittest.mock import AsyncMock

import pytest

from tests.unit.constants import DEFAULT_CLUSTER_NAME, TEST_PRESETS

pytest_plugins = [
    "apolo_app_types_fixtures.apolo_clients",
    "apolo_app_types_fixtures.constants",
]


@pytest.fixture
def presets_available(request):
    """Override to use our test presets."""
    return getattr(request, "param", TEST_PRESETS)


@pytest.fixture
def cluster_domain():
    """Cluster domain for testing."""
    return f"{DEFAULT_CLUSTER_NAME}.local"


@pytest.fixture
def mock_apolo_client(setup_clients):
    """Mock Apolo client with our test presets."""
    # Use the setup_clients mock and override presets
    setup_clients.config.presets = TEST_PRESETS

    # Mock capacity
    async def mock_get_capacity():
        return {
            "cpu-small": 5,
            "cpu-medium": 3,
            "cpu-large": 1,
            "gpu-1x-a100": 2,
            "cpu-tiny": 10,
        }

    setup_clients.jobs.get_capacity = AsyncMock(side_effect=mock_get_capacity)
    return setup_clients
