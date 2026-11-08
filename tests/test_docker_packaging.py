
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCKER_DIR = PROJECT_ROOT / "docker"


def load_compose():
    compose_path = DOCKER_DIR / "docker-compose.yml"

    with compose_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def test_serving_dockerfile_exists():
    dockerfile = DOCKER_DIR / "Dockerfile"

    assert dockerfile.is_file()

    content = dockerfile.read_text(encoding="utf-8")

    assert "requirements-serving.txt" in content
    assert "USER appuser" in content
    assert "HEALTHCHECK" in content
    assert "src.api.app:app" in content


def test_training_dockerfile_exists():
    dockerfile = DOCKER_DIR / "Dockerfile.train"

    assert dockerfile.is_file()

    content = dockerfile.read_text(encoding="utf-8")

    assert "requirements-training.txt" in content
    assert "USER trainer" in content
    assert "src.models.train" in content


def test_requirements_are_separated():
    serving = (
        DOCKER_DIR / "requirements-serving.txt"
    ).read_text(encoding="utf-8")

    training = (
        DOCKER_DIR / "requirements-training.txt"
    ).read_text(encoding="utf-8")

    assert "fastapi" in serving.lower()
    assert "uvicorn" in serving.lower()

    assert "mlflow" in training.lower()
    assert "scikit-learn" in training.lower()

    assert "jupyterlab" not in serving.lower()
    assert "pytest" not in serving.lower()


def test_compose_services():
    compose = load_compose()

    services = compose["services"]

    assert "api" in services
    assert "mlflow" in services
    assert "training" in services

    assert "training" in services["training"]["profiles"]


def test_mlflow_storage_is_persistent():
    compose = load_compose()

    mlflow = compose["services"]["mlflow"]

    assert "mlflow-data:/mlflow" in mlflow["volumes"]
    assert "mlflow-data" in compose["volumes"]

    command = " ".join(mlflow["command"])

    assert "--serve-artifacts" in command
    assert "--artifacts-destination" in command
    assert "sqlite:////mlflow/mlflow.db" in command


def test_training_uses_mlflow_network():
    compose = load_compose()

    training = compose["services"]["training"]

    assert (
        training["environment"]["MLFLOW_TRACKING_URI"]
        == "http://mlflow:5000"
    )

    assert (
        training["depends_on"]["mlflow"]["condition"]
        == "service_started"
    )


def test_training_data_is_read_only():
    compose = load_compose()

    training = compose["services"]["training"]

    assert "../data:/app/data:ro" in training["volumes"]


def test_third_party_images_are_versioned():
    compose = load_compose()

    for service in compose["services"].values():
        image = service.get("image")

        if not image:
            continue

        assert not image.endswith(":latest")
