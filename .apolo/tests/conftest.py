"""Test configuration for HuggingFace Proxy App tests."""

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
