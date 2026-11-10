"""MLflow experiment tracking and model registry for training runs (Issue #6).

Each execution is a parent run with one nested run per candidate, in an experiment
named after the training mode. The selected candidate becomes a new registered
version with the ``staging`` alias, and takes the ``champion`` alias when it beats
the Dummy reference and the current champion on validation Macro F1.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Optional

import mlflow
import requests
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types.schema import ColSpec, Schema

from src.data.contract import SPECS
from src.models import artifacts

MODE_EXPERIMENTS = {
    "baseline": "baseline",
    "ablations": "ablations",
    "sequential": "search",
}
STAGING_ALIAS = "staging"
CHAMPION_ALIAS = "champion"
# Small airports give unstable scores, so they do not define the worst subgroup.
MIN_SUBGROUP_SUPPORT = 100
PINNED_PACKAGES = ("numpy", "pandas", "scikit-learn", "lightgbm", "joblib")
SUBGROUPS = {"OP_UNIQUE_CARRIER": "carrier", "ORIGIN": "origin"}
SOURCE_DIRECTORY = Path(__file__).resolve().parents[1]
CANDIDATE_FILES = ("config.json", "metrics.json", "cv.json", "manifest.json")
# Integers are declared as doubles because optional columns can hold NaN.
SIGNATURE_TYPES = {
    "int": "double",
    "float": "double",
    "code": "string",
    "text": "string",
    "date": "datetime",
}


@dataclass(frozen=True)
class TrackingConfig:
    tracking_uri: str
    experiment_base: str
    registered_model: str


def load_tracking_config(config: dict) -> Optional[TrackingConfig]:
    """Tracking settings, or None when disabled. The environment overrides the URI."""
    if not config["training"].get("tracking", {}).get("enabled", False):
        return None
    settings = config["mlflow"]
    return TrackingConfig(
        os.environ.get("MLFLOW_TRACKING_URI") or settings["tracking_uri"],
        settings["experiment_name"],
        settings["registered_model_name"],
    )


def check_server(tracking_uri: str) -> None:
    """Fail before training starts when the MLflow server cannot be reached."""
    if not tracking_uri.startswith("http"):
        return
    try:
        requests.get(
            f"{tracking_uri.rstrip('/')}/health", timeout=10
        ).raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            f"MLflow server unreachable at {tracking_uri}: {error}"
        ) from error


def experiment_name(base: str, mode: str) -> str:
    if mode not in MODE_EXPERIMENTS:
        raise ValueError(f"Unknown experiment mode: {mode}")
    return f"{base}/{MODE_EXPERIMENTS[mode]}"


def decide_promotion(candidate_f1, dummy_f1, current, validation_sha256):
    """Apply the promotion rule. ``current`` is None or the champion's f1 and hash."""
    if dummy_f1 is not None and candidate_f1 <= dummy_f1:
        return False, f"Macro F1 {candidate_f1:.6f} does not beat Dummy {dummy_f1:.6f}"
    if current is None:
        return True, "First champion"
    if current["validation_sha256"] != validation_sha256:
        return False, "Champion was evaluated on different validation data"
    if candidate_f1 <= current["macro_f1"]:
        return False, (
            f"Macro F1 {candidate_f1:.6f} does not beat champion "
            f"{current['macro_f1']:.6f}"
        )
    return True, f"Macro F1 {candidate_f1:.6f} beats {current['macro_f1']:.6f}"


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        flat = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}{key}."))
        return flat
    return {prefix[:-1]: str(value)}


def _metrics(result):
    validation = result["validation"]
    metrics = {
        "cv_macro_f1": result["cv_macro_f1"],
        "cv_log_loss": result["cv_log_loss"],
        "fit_seconds": validation["fit_seconds"],
        "inference_seconds": validation["inference_seconds"],
        "model_bytes": validation["model_bytes"],
    }
    for name in ("macro_f1", "log_loss", "accuracy", "balanced_accuracy"):
        metrics[f"val_{name}"] = validation[name]
    for class_name, row in validation["per_class"].items():
        for name in ("f1", "precision", "recall"):
            metrics[f"val_{name}_{class_name}"] = row[name]
    for column, label in SUBGROUPS.items():
        scores = [
            group["macro_f1"]
            for group in validation["subgroups"][column].values()
            if group["support"] >= MIN_SUBGROUP_SUPPORT
        ]
        if scores:
            metrics[f"val_worst_macro_f1_{label}"] = min(scores)
            metrics[f"val_median_macro_f1_{label}"] = median(scores)
    return metrics


def _classification_report(validation):
    lines = [f"{'class':<16}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"]
    for name, row in validation["per_class"].items():
        lines.append(
            f"{name:<16}{row['precision']:>10.4f}{row['recall']:>10.4f}"
            f"{row['f1']:>10.4f}{row['support']:>10d}"
        )
    lines += [
        "",
        f"macro_f1: {validation['macro_f1']:.6f}",
        f"accuracy: {validation['accuracy']:.6f}",
        f"balanced_accuracy: {validation['balanced_accuracy']:.6f}",
        f"log_loss: {validation['log_loss']:.6f}",
    ]
    return "\n".join(lines) + "\n"


def _confusion_matrix(validation):
    classes = validation["class_order"]
    rows = [",".join(["actual \\ predicted", *classes])]
    for name, counts in zip(classes, validation["confusion_matrix"]):
        rows.append(",".join([name, *map(str, counts)]))
    return "\n".join(rows) + "\n"


def _tags(snapshot, mode):
    provenance = snapshot["provenance"]
    commit = provenance["git_commit"] or "unknown"
    return {
        "mode": mode,
        "git_commit": commit,
        "git_dirty": str(provenance["git_dirty"]),
        "mlflow.source.git.commit": commit,
        "data.train_sha256": snapshot["splits"]["train"]["sha256"],
        "data.validation_sha256": snapshot["splits"]["validation"]["sha256"],
        "data.split_manifest_sha256": snapshot["split_manifest"]["sha256"],
    }


def model_signature(columns) -> ModelSignature:
    """Input schema from the data contract, which knows the optional columns."""
    return ModelSignature(
        inputs=Schema(
            [
                ColSpec(
                    type=SIGNATURE_TYPES[SPECS[column].kind],
                    name=column,
                    required=SPECS[column].request == "required",
                )
                for column in columns
                if column in SPECS
            ]
        ),
        outputs=Schema([ColSpec(type="string")]),
    )


def _log_candidate(result, directory, filename, serialization, snapshot, columns, tags):
    """Log one candidate; returns its model type, used by the promotion rule."""
    effective = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    experiment = effective["training"]["experiments"][0]
    folds = json.loads((directory / "cv.json").read_text(encoding="utf-8"))
    validation = result["validation"]

    mlflow.set_tags(tags)
    mlflow.log_params(_flatten({k: v for k, v in experiment.items() if k != "name"}))
    mlflow.log_metrics(_metrics(result))
    for fold in folds:
        mlflow.log_metric("cv_fold_macro_f1", fold["metrics"]["macro_f1"], fold["fold"])
    for name in CANDIDATE_FILES:
        mlflow.log_artifact(str(directory / name))
    mlflow.log_artifact(str(directory / filename))
    mlflow.log_text(_classification_report(validation), "classification_report.txt")
    mlflow.log_text(_confusion_matrix(validation), "confusion_matrix.csv")

    model = artifacts.load_model(directory / filename, serialization)
    versions = snapshot["provenance"]["versions"]
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=model_signature(columns),
        pip_requirements=[
            f"{package}=={versions[package]}"
            for package in PINNED_PACKAGES
            if versions.get(package)
        ],
        code_paths=[str(SOURCE_DIRECTORY)],
    )
    return experiment["model"]


def _current_champion(client, name):
    version = client.get_registered_model(name).aliases.get(CHAMPION_ALIAS)
    if version is None:
        return None
    run = client.get_run(client.get_model_version(name, version).run_id)
    return {
        "version": version,
        "macro_f1": run.data.metrics["val_macro_f1"],
        "validation_sha256": run.data.tags["data.validation_sha256"],
    }


def register_and_promote(tracking, run_id, candidate_f1, dummy_f1, validation_sha256):
    """Register a selected run as ``staging`` and apply the promotion rule."""
    name = tracking.registered_model
    client = MlflowClient(tracking.tracking_uri)
    version = mlflow.register_model(f"runs:/{run_id}/model", name).version
    client.set_registered_model_alias(name, STAGING_ALIAS, version)
    promoted, reason = decide_promotion(
        candidate_f1,
        dummy_f1,
        _current_champion(client, name),
        validation_sha256,
    )
    if promoted:
        client.set_registered_model_alias(name, CHAMPION_ALIAS, version)
    client.set_model_version_tag(name, version, "promotion", reason)
    return {
        "registered_model": name,
        "version": str(version),
        "promoted_to_champion": promoted,
        "promotion_reason": reason,
    }


def log_training(tracking, config, output, report, snapshot, columns):
    """Log a completed training run and register its selected candidate."""
    settings = config["training"]
    mode = settings.get("mode", "baseline")
    serialization = settings.get("serialization", "joblib")
    filename = artifacts.model_filename(serialization)
    tags = _tags(snapshot, mode)
    mlflow.set_tracking_uri(tracking.tracking_uri)
    name = experiment_name(tracking.experiment_base, mode)
    mlflow.set_experiment(name)

    run_ids, dummy_scores = {}, []
    with mlflow.start_run(run_name=Path(output).name) as parent:
        mlflow.set_tags({**tags, "selected_candidate": report["selected"]})
        mlflow.log_params(
            {
                "mode": mode,
                "seed": settings["seed"],
                "sample_size": settings.get("sample_size"),
                "cv_splits": settings["cv_splits"],
                "threads": settings["threads"],
            }
        )
        mlflow.log_dict(report, "summary.json")
        for result in report["results"]:
            with mlflow.start_run(run_name=result["name"], nested=True) as child:
                run_ids[result["name"]] = child.info.run_id
                model_type = _log_candidate(
                    result,
                    Path(output) / result["name"],
                    filename,
                    serialization,
                    snapshot,
                    columns,
                    tags,
                )
            if model_type == "dummy":
                dummy_scores.append(result["validation"]["macro_f1"])
        selected = next(r for r in report["results"] if r["name"] == report["selected"])
        mlflow.log_metrics(
            {
                "selected_cv_macro_f1": selected["cv_macro_f1"],
                "selected_val_macro_f1": selected["validation"]["macro_f1"],
            }
        )
        if "test" in report:
            mlflow.log_metrics(
                {
                    "test_macro_f1": report["test"]["macro_f1"],
                    "test_log_loss": report["test"]["log_loss"],
                }
            )
        registry = register_and_promote(
            tracking,
            run_ids[report["selected"]],
            selected["validation"]["macro_f1"],
            max(dummy_scores) if dummy_scores else None,
            tags["data.validation_sha256"],
        )
        mlflow.set_tags(
            {
                "registered_version": registry["version"],
                "promotion": registry["promotion_reason"],
            }
        )
    return {"experiment": name, "run_id": parent.info.run_id, **registry}
