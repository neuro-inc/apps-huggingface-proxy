"""Tests for health check endpoint."""

from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """Test the /healthz endpoint."""
    response = client.get("/healthz")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_root_endpoint(client: TestClient):
    """Test the root endpoint."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
