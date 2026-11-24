"""Shared fixtures: the API tests run against a small model on synthetic flights."""

import pytest

from src.api.app import app
from src.models.smoke import build_smoke_model
from src.monitoring.traffic import TrafficLog


@pytest.fixture(scope="session")
def smoke_model():
    return build_smoke_model()


@pytest.fixture
def traffic_dir(tmp_path):
    return tmp_path / "traffic"


@pytest.fixture(autouse=True)
def loaded_api_model(smoke_model, traffic_dir):
    """Clients created without a context manager skip the startup that loads it."""
    app.state.model = smoke_model
    app.state.model_name = "smoke"
    app.state.model_version = "test"
    app.state.fallback_class = "on_time"
    app.state.traffic = TrafficLog(traffic_dir)
