import json
import re
from pathlib import Path

import yaml
from prometheus_client import REGISTRY

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
    serving = (DOCKER_DIR / "requirements-serving.txt").read_text(encoding="utf-8")

    training = (DOCKER_DIR / "requirements-training.txt").read_text(encoding="utf-8")

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

    assert training["environment"]["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"

    assert training["depends_on"]["mlflow"]["condition"] == "service_started"


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


MONITORING_DIR = DOCKER_DIR / "monitoring"


def load_yaml(path):
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def monitored_composes():
    return [load_compose(), load_yaml(DOCKER_DIR / "compose.vm2.yml")]


def test_monitoring_services_are_pinned_and_persistent():
    for compose in monitored_composes():
        services = compose["services"]
        prometheus, grafana = services["prometheus"], services["grafana"]

        assert prometheus["image"] == "prom/prometheus:v3.1.0"
        assert grafana["image"] == "grafana/grafana:11.4.0"
        assert "prometheus-data:/prometheus" in prometheus["volumes"]
        assert "grafana-data:/var/lib/grafana" in grafana["volumes"]
        assert {"prometheus-data", "grafana-data"} <= set(compose["volumes"])
        assert "--storage.tsdb.path=/prometheus" in prometheus["command"]


def test_monitoring_mounts_exist():
    for compose in monitored_composes():
        for name in ("prometheus", "grafana"):
            for volume in compose["services"][name]["volumes"]:
                source = volume.split(":")[0]
                if source.startswith("./"):
                    assert (DOCKER_DIR / source).exists(), source


def test_vm2_grafana_requires_a_password():
    grafana = load_yaml(DOCKER_DIR / "compose.vm2.yml")["services"]["grafana"]
    password = grafana["environment"]["GF_SECURITY_ADMIN_PASSWORD"]
    assert password.startswith("${GRAFANA_ADMIN_PASSWORD:?")


def test_vm2_publishes_the_api_where_the_deploy_script_checks_it():
    # Port 8000 is blocked on the DEI VMs; the leaderboard uses 8080.
    api = load_yaml(DOCKER_DIR / "compose.vm2.yml")["services"]["api"]
    assert api["ports"] == ["8080:8000"]
    script = (PROJECT_ROOT / "scripts" / "deploy_model.sh").read_text(encoding="utf-8")
    assert "HEALTH_URL=${HEALTH_URL:-http://localhost:8080/health}" in script


def test_prometheus_scrapes_the_api_metrics():
    config = load_yaml(MONITORING_DIR / "prometheus.yml")
    api = next(
        job for job in config["scrape_configs"] if job["job_name"] == "flight-delay-api"
    )
    assert api["metrics_path"] == "/metrics"
    assert api["static_configs"][0]["targets"] == ["api:8000"]


def test_grafana_dashboard_uses_the_provisioned_data_source():
    datasource = load_yaml(
        MONITORING_DIR / "grafana/provisioning/datasources/prometheus.yml"
    )["datasources"][0]
    dashboard = json.loads(
        (MONITORING_DIR / "grafana/dashboards/flight-delay-api.json").read_text()
    )
    panels = [panel for panel in dashboard["panels"] if panel["type"] != "row"]
    uids = {panel["datasource"]["uid"] for panel in panels}
    assert uids == {datasource["uid"]}
    assert len({panel["id"] for panel in dashboard["panels"]}) == len(
        dashboard["panels"]
    )


def test_dashboard_queries_only_metrics_the_api_exposes():
    from fastapi.testclient import TestClient

    from src.api.app import app

    # The HTTP metrics are registered when the middleware handles a request.
    TestClient(app).get("/health")

    suffixes = ("", "_total", "_bucket", "_count", "_sum", "_info")
    exposed = {
        family.name + suffix for family in REGISTRY.collect() for suffix in suffixes
    }
    dashboard = (
        MONITORING_DIR / "grafana/dashboards/flight-delay-api.json"
    ).read_text()
    queried = set(re.findall(r"\b((?:flight_delay|http)_[a-z_]+)", dashboard))
    assert queried <= exposed, queried - exposed


def test_prometheus_keeps_history_for_the_whole_evaluation():
    for compose in monitored_composes():
        command = compose["services"]["prometheus"]["command"]
        assert "--storage.tsdb.retention.time=90d" in command
        assert "--storage.tsdb.retention.size=2GB" in command
