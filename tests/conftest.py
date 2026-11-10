"""Shared fixtures: the API tests run against a small model on synthetic flights."""

import pytest

from src.api.app import app
from src.models.smoke import build_smoke_model


@pytest.fixture(scope="session")
def smoke_model():
    return build_smoke_model()


@pytest.fixture(autouse=True)
def loaded_api_model(smoke_model):
    """Clients created without a context manager skip the startup that loads it."""
    app.state.model = smoke_model
    app.state.model_name = "smoke"
    app.state.model_version = "test"
