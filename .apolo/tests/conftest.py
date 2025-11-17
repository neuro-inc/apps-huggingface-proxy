import pytest

from tests.unit.constants import TEST_PRESETS

pytest_plugins = [
    "apolo_app_types_fixtures.apolo_clients",
    "apolo_app_types_fixtures.constants",
]


@pytest.fixture
def presets_available(request):
    """Override to use our test presets."""
    return getattr(request, "param", TEST_PRESETS)
