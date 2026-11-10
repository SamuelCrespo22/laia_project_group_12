"""MLflow tracking tests (Issue #6). Synthetic data and a local SQLite store."""

import json
import subprocess

import mlflow
import pytest
import yaml
from mlflow.tracking import MlflowClient

from src.models import artifacts, tracking
from src.models.tracking import (
    TrackingConfig,
    decide_promotion,
    experiment_name,
    load_tracking_config,
    register_and_promote,
)
from src.models.train import main
from tests.test_training import environment as training_environment
from tests.test_training import prepared

environment = training_environment
KEPT = {"dummy_prior", "logistic_balanced", "lightgbm_balanced"}


@pytest.fixture
def tracked(environment, tmp_path, monkeypatch):
    """Project config logging three candidates to a temporary MLflow store."""
    path, config, _, _ = environment
    config = json.loads(json.dumps(config))
    config["training"]["tracking"] = {"enabled": True}
    config["training"]["experiments"] = [
        e for e in config["training"]["experiments"] if e["name"] in KEPT
    ]
    config["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path}/mlflow.db"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    # A SQLite store writes artifacts below the working directory.
    monkeypatch.chdir(tmp_path)
    yield path, config, tmp_path
    mlflow.set_tracking_uri(None)


def train(path, tmp_path, name):
    return main(["--config", str(path), "--output-dir", str(tmp_path / name)])


def client_for(config):
    return MlflowClient(config["mlflow"]["tracking_uri"])


def aliases_of(config):
    model = client_for(config).get_registered_model("flight-delay-severity")
    return {alias: str(version) for alias, version in model.aliases.items()}


def settings(config):
    return TrackingConfig(
        config["mlflow"]["tracking_uri"], "flight-delay", "flight-delay-severity"
    )


def test_training_logs_parent_and_candidate_runs(tracked):
    path, config, tmp_path = tracked

    report = train(path, tmp_path, "run")

    client = client_for(config)
    experiment = client.get_experiment_by_name("flight-delay/baseline")
    assert report["mlflow"]["experiment"] == "flight-delay/baseline"
    runs = client.search_runs([experiment.experiment_id])
    parent = client.get_run(report["mlflow"]["run_id"])
    children = [r for r in runs if r.info.run_id != parent.info.run_id]
    assert len(children) == 3
    assert parent.data.tags["selected_candidate"] == report["selected"]
    assert {r.data.tags["mlflow.parentRunId"] for r in children} == {parent.info.run_id}

    child = next(r for r in children if r.info.run_name == "lightgbm_balanced")
    metrics = child.data.metrics
    for name in (
        "cv_macro_f1",
        "cv_log_loss",
        "val_macro_f1",
        "val_log_loss",
        "val_f1_severe_delay",
        "fit_seconds",
        "model_bytes",
    ):
        assert name in metrics
    assert child.data.params["model"] == "lightgbm"
    assert child.data.params["params.class_weight"] == "balanced"
    assert len(client.get_metric_history(child.info.run_id, "cv_fold_macro_f1")) == 2
    for name in ("data.train_sha256", "data.validation_sha256", "git_commit"):
        assert child.data.tags[name]

    names = {a.path for a in client.list_artifacts(child.info.run_id)}
    assert {
        "pipeline.joblib",
        "metrics.json",
        "cv.json",
        "classification_report.txt",
        "confusion_matrix.csv",
        "model",
    } <= names


def test_subgroup_metrics_ignore_small_groups():
    group = {"macro_f1": 0.5, "support": 100}
    tiny = {"macro_f1": 0.0, "support": 99}
    per_class = {"on_time": {"f1": 1, "precision": 1, "recall": 1}}
    result = {
        "cv_macro_f1": 0.4,
        "cv_log_loss": 1.0,
        "validation": {
            "fit_seconds": 1.0,
            "inference_seconds": 1.0,
            "model_bytes": 10,
            "macro_f1": 0.4,
            "log_loss": 1.0,
            "accuracy": 0.5,
            "balanced_accuracy": 0.5,
            "per_class": per_class,
            "subgroups": {
                "OP_UNIQUE_CARRIER": {"AA": group, "ZZ": tiny},
                "ORIGIN": {"JFK": tiny},
            },
        },
    }

    metrics = tracking._metrics(result)

    assert metrics["val_worst_macro_f1_carrier"] == 0.5
    assert metrics["val_median_macro_f1_carrier"] == 0.5
    assert "val_worst_macro_f1_origin" not in metrics


def test_first_run_becomes_champion_and_loads_by_alias(tracked, environment):
    path, config, tmp_path = tracked

    report = train(path, tmp_path, "run")

    assert report["mlflow"]["version"] == "1"
    assert report["mlflow"]["promoted_to_champion"] is True
    assert aliases_of(config) == {"staging": "1", "champion": "1"}
    client = client_for(config)
    version = client.get_model_version_by_alias("flight-delay-severity", "champion")
    assert version.tags["promotion"] == "First champion"
    assert version.run_id != report["mlflow"]["run_id"]

    X, _, _, _ = prepared(environment, "validation")
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    loaded = mlflow.pyfunc.load_model("models:/flight-delay-severity@champion")
    inputs = {c.name: c.required for c in loaded.metadata.signature.inputs.inputs}
    assert inputs["ORIGIN"] is True
    assert inputs["DISTANCE"] is False
    plain = X.head(5).copy()
    for column in plain.select_dtypes("string"):
        plain[column] = plain[column].astype(object)
    for column in plain.select_dtypes(["Int64", "Float64"]):
        plain[column] = plain[column].astype("float64")
    assert len(loaded.predict(plain)) == 5


def test_equal_score_moves_staging_but_keeps_champion(tracked):
    path, config, tmp_path = tracked
    train(path, tmp_path, "first")

    report = train(path, tmp_path, "second")

    assert report["mlflow"]["version"] == "2"
    assert report["mlflow"]["promoted_to_champion"] is False
    assert aliases_of(config) == {"staging": "2", "champion": "1"}


@pytest.mark.parametrize(
    "delta, champion_version", [(0.1, "2"), (-0.1, "1")], ids=["better", "worse"]
)
def test_promotion_follows_the_validation_score(tracked, delta, champion_version):
    path, config, tmp_path = tracked
    report = train(path, tmp_path, "run")
    client = client_for(config)
    version = client.get_model_version("flight-delay-severity", "1")
    run = client.get_run(version.run_id)
    score = run.data.metrics["val_macro_f1"]

    result = register_and_promote(
        settings(config),
        version.run_id,
        score + delta,
        0.0,
        run.data.tags["data.validation_sha256"],
    )

    assert result["promoted_to_champion"] is (delta > 0)
    assert aliases_of(config) == {"staging": "2", "champion": champion_version}
    assert report["mlflow"]["version"] == "1"


@pytest.mark.parametrize(
    "candidate, dummy, current, sha, promoted",
    [
        (0.4, 0.3, None, "a", True),
        (0.3, 0.3, None, "a", False),
        (0.2, 0.3, None, "a", False),
        (0.4, None, None, "a", True),
        (0.5, 0.3, {"macro_f1": 0.4, "validation_sha256": "a"}, "a", True),
        (0.4, 0.3, {"macro_f1": 0.4, "validation_sha256": "a"}, "a", False),
        (0.3, 0.2, {"macro_f1": 0.4, "validation_sha256": "a"}, "a", False),
        (0.5, 0.3, {"macro_f1": 0.4, "validation_sha256": "a"}, "b", False),
    ],
)
def test_promotion_rule(candidate, dummy, current, sha, promoted):
    assert decide_promotion(candidate, dummy, current, sha)[0] is promoted


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("baseline", "base/baseline"),
        ("ablations", "base/ablations"),
        ("sequential", "base/search"),
    ],
)
def test_each_mode_has_its_own_experiment(mode, expected):
    assert experiment_name("base", mode) == expected


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown experiment mode"):
        experiment_name("base", "other")


def test_environment_overrides_the_configured_uri(monkeypatch):
    config = {
        "training": {"tracking": {"enabled": True}},
        "mlflow": {
            "tracking_uri": "http://configured:5000",
            "experiment_name": "base",
            "registered_model_name": "model",
        },
    }
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://vm1:5000")

    assert load_tracking_config(config).tracking_uri == "http://vm1:5000"


def test_tracking_is_off_when_not_enabled():
    assert load_tracking_config({"training": {}}) is None


def test_no_tracking_flag_skips_mlflow(tracked):
    path, config, tmp_path = tracked

    report = main(
        ["--config", str(path), "--output-dir", str(tmp_path / "r"), "--no-tracking"]
    )

    assert "mlflow" not in report
    assert (
        client_for(config).search_experiments(
            filter_string="name = 'flight-delay/baseline'"
        )
        == []
    )


def test_unreachable_server_fails_before_training(tracked, monkeypatch):
    path, config, tmp_path = tracked
    config["mlflow"]["tracking_uri"] = "http://127.0.0.1:1"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "failed"

    def forbidden(*args, **kwargs):
        raise AssertionError("training started")

    monkeypatch.setattr("src.models.experiments._run", forbidden)
    with pytest.raises(RuntimeError, match="MLflow server unreachable"):
        main(["--config", str(path), "--output-dir", str(output)])

    assert json.loads((output / "status.json").read_text())["state"] == "failed"


def test_commit_comes_from_the_environment_without_git(monkeypatch):
    def no_git(*args, **kwargs):
        raise OSError("git not installed")

    monkeypatch.setattr(subprocess, "check_output", no_git)
    monkeypatch.setenv("GIT_COMMIT", "abc123")

    assert artifacts.provenance()["git_commit"] == "abc123"
