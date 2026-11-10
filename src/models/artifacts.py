"""Local persistence boundary; add formats here without changing the runner."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import joblib
import pandas as pd

SERIALIZERS = {"joblib": (joblib.dump, joblib.load)}


def model_filename(format="joblib"):
    if format not in SERIALIZERS:
        raise ValueError(f"Unknown serialization format: {format}")
    return "pipeline." + format


def save_model(model, path, format="joblib"):
    _atomic(path, lambda temporary: SERIALIZERS[format][0](model, temporary))
    return Path(path).stat().st_size


def load_model(path, format="joblib"):
    return SERIALIZERS[format][1](path)


def write_json(path, payload):
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic(
        path, lambda temporary: Path(temporary).write_text(content, encoding="utf-8")
    )


def _atomic(path, writer):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name, suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def source_fingerprint(root=None, config_path=None):
    """Fingerprint relevant source contents, including untracked files and deletions."""
    root = Path(root or Path(__file__).resolve().parents[2])
    files = set((root / "src").rglob("*.py")) | set((root / "configs").rglob("*.yaml"))
    files |= set(root.glob("*requirements*.txt"))
    hashes = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
        if p.is_file()
    }
    if config_path is not None:
        hashes["effective-input-config"] = hashlib.sha256(
            Path(config_path).read_bytes()
        ).hexdigest()
    encoded = json.dumps(hashes, sort_keys=True).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": hashes}


def file_identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(Path(path).resolve()),
        "sha256": digest.hexdigest(),
        "bytes": Path(path).stat().st_size,
    }


def provenance(config_path=None):
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=root
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, cwd=root
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    versions = {"python": platform.python_version()}
    for package in (
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "lightgbm",
        "joblib",
        "threadpoolctl",
        "imbalanced-learn",
        "tsfresh",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "versions": versions,
        "source_fingerprint": source_fingerprint(root, config_path),
    }


def describe_pipeline(pipeline):
    """Resolved estimator defaults and preprocessing overrides, JSON-ready."""
    estimator = pipeline.named_steps["model"]
    result = {
        "estimator": type(estimator).__name__,
        "params": estimator.get_params(deep=False),
        "feature_steps": list(pipeline.named_steps),
    }
    steps = pipeline.named_steps
    if "columns" in steps:
        columns = steps["columns"]
        numeric = columns.named_transformers_.get("numeric")
        if numeric is not None:
            imputer, scaler = (
                numeric.named_steps["imputer"],
                numeric.named_steps["scaler"],
            )
            result["numeric_columns"] = list(columns.transformers_[0][2])
            result["imputation"] = {
                "strategy": getattr(imputer, "strategy", "knn"),
                "keep_empty_features": imputer.keep_empty_features,
                "missing_values": "NaN",
            }
            result["scaling"] = (
                {"type": "passthrough"}
                if scaler == "passthrough"
                else {
                    "type": type(scaler).__name__,
                    "params": scaler.get_params(deep=False),
                }
            )
    result["column_steps"] = {}
    for name, branch, columns in steps["columns"].transformers_:
        if name == "remainder":
            continue
        result["column_steps"][name] = {}
        for step_name, step in branch.steps:
            if step == "passthrough":
                continue
            params = step.get_params(deep=False)
            if "missing_values" in params:
                params["missing_values"] = (
                    "pd.NA" if params["missing_values"] is pd.NA else "NaN"
                )
            params = {k: str(v) if callable(v) else v for k, v in params.items()}
            result["column_steps"][name][step_name] = {
                "type": type(step).__name__,
                "params": params,
            }
    for name in ("engineering", "outliers", "tsfresh", "sampling"):
        if name in steps and steps[name] != "passthrough":
            result[name] = steps[name].get_params(deep=False)
    if "selection" in steps and steps["selection"] != "passthrough":
        selector = steps["selection"]
        result["selection"] = {
            "type": type(selector).__name__,
            "score_func": "f_classif",
        }
        result["selection"].update(
            {k: v for k, v in selector.get_params().items() if k != "score_func"}
        )
    return result


def training_diagnostics(pipeline, X):
    """Inspect fitted transforms without adding behavior to the pipeline."""
    from sklearn.pipeline import Pipeline

    transforms = Pipeline(
        [(name, step) for name, step in pipeline.steps[:-1] if name != "sampling"]
    )
    return {"transformed_features": transforms.transform(X.iloc[:1]).shape[1]}
