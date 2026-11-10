"""Framework-neutral experiment orchestration over the documented strategy contract."""

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    TimeSeriesSplit,
)
from sklearn.model_selection import cross_validate as sklearn_cross_validate
from threadpoolctl import threadpool_info, threadpool_limits

from src.data.preprocessor import (
    CLASSES,
)
from src.data.temporal_split import load_published_splits, load_split_config
from src.models import artifacts
from src.models.dataset import prepare_split, prepared_batches
from src.models.evaluation import EvaluationAccumulator, evaluate
from src.models.factory import create_model
from src.models.search import ablations, sequential
from src.models.tracking import check_server, load_tracking_config, log_training


def expanding_folds(X, n_splits):
    """Expanding windows over unique days, never split a day across partitions."""
    days = np.array(sorted(pd.to_datetime(X["FL_DATE"]).dt.normalize().unique()))
    if n_splits < 2 or len(days) < n_splits + 1:
        raise ValueError("Insufficient unique training days for temporal CV")
    dates = pd.to_datetime(X["FL_DATE"]).dt.normalize()
    for train, validation in TimeSeriesSplit(n_splits=n_splits).split(days):
        yield (
            np.flatnonzero(dates.isin(days[train])),
            np.flatnonzero(dates.isin(days[validation])),
        )


def _fit(experiment, preprocessing, seed, threads, X, y):
    if set(y) != set(CLASSES):
        raise ValueError("Every training partition must contain all target classes")
    model, order = create_model(experiment, preprocessing, seed, threads, CLASSES)
    with threadpool_limits(limits=threads):
        start = time.perf_counter()
        model.fit(X, y)
        duration = time.perf_counter() - start
        pools = threadpool_info()
    return model, tuple(model.classes_), duration, pools


def _evaluate(model, order, X, y, metadata, threads=1):
    with threadpool_limits(limits=threads):
        start = time.perf_counter()
        predictions, probabilities = model.predict(X), model.predict_proba(X)
        duration = time.perf_counter() - start
    metrics = evaluate(y, predictions, probabilities, order, metadata)
    metrics["inference_seconds"] = duration
    return metrics


def cross_validate(experiment, preprocessing, seed, threads, X, y, metadata, folds):
    """The single CV implementation used by baselines, ablations and A/B/C."""
    for a, _ in folds:
        if set(y.iloc[a]) != set(CLASSES):
            raise ValueError("Every training partition must contain all target classes")
    model, _ = create_model(experiment, preprocessing, seed, threads, CLASSES)
    reports = []

    def scoring(estimator, rows, labels):
        metrics = _evaluate(
            estimator,
            estimator.classes_,
            rows,
            labels,
            metadata.loc[rows.index],
            threads,
        )
        reports.append(metrics)
        return {"macro_f1": metrics["macro_f1"], "neg_log_loss": -metrics["log_loss"]}

    with threadpool_limits(limits=threads):
        scores = sklearn_cross_validate(
            model, X, y, cv=folds, scoring=scoring, n_jobs=1, error_score="raise"
        )
        pools = threadpool_info()
    results = [
        {
            "fold": number,
            "train_rows": len(a),
            "validation_rows": len(b),
            "train_end": str(X.iloc[a].FL_DATE.max()),
            "validation_start": str(X.iloc[b].FL_DATE.min()),
            "fit_seconds": float(scores["fit_time"][number]),
            "native_pools": pools,
            "metrics": reports[number],
        }
        for number, (a, b) in enumerate(folds)
    ]
    return {
        "cv_macro_f1": float(np.mean([f["metrics"]["macro_f1"] for f in results])),
        "cv_log_loss": float(np.mean([f["metrics"]["log_loss"] for f in results])),
        "folds": results,
    }


def _evaluate_split(model, order, path, period, preprocessing, settings):
    accumulator = EvaluationAccumulator(order)
    duration = 0.0
    counts = {"raw": 0, "eligible": 0, "used": 0, "classes": {c: 0 for c in CLASSES}}
    with threadpool_limits(limits=settings["threads"]):
        for X, y, metadata, raw_rows in prepared_batches(
            path,
            period,
            preprocessing,
            settings.get("evaluation_batch_size", 50000),
            (settings["history_column"],) if settings.get("history_column") else (),
        ):
            counts["raw"] += raw_rows
            if not len(y):
                continue
            start = time.perf_counter()
            predictions, probabilities = model.predict(X), model.predict_proba(X)
            duration += time.perf_counter() - start
            accumulator.update(y, predictions, probabilities, metadata)
            counts["eligible"] += len(y)
            counts["used"] += len(y)
            for c in CLASSES:
                counts["classes"][c] += int(y.eq(c).sum())
    metrics = accumulator.result()
    metrics["inference_seconds"] = duration
    return metrics, counts


def _run(
    config,
    config_path,
    preprocessing,
    output_dir,
    snapshot,
    final_test=False,
    tracking=None,
):
    settings = config["training"]
    serialization = settings.get("serialization", "joblib")
    filename = artifacts.model_filename(serialization)
    seed, threads = settings["seed"], settings["threads"]
    if threads < 1:
        raise ValueError("threads must be positive")
    paths = snapshot["paths"]
    boundaries = load_split_config(config_path)
    data = {
        "train": prepare_split(
            paths["train"],
            boundaries["train"],
            preprocessing,
            settings.get("sample_size"),
            seed,
            settings.get("read_batch_size", 50000),
            settings.get("memory_budget_mb", 1024),
            (settings["history_column"],) if settings.get("history_column") else (),
        )
    }
    X, y, _, _ = data["train"]
    folds = list(expanding_folds(X, settings["cv_splits"]))
    mode = settings.get("mode", "baseline")
    cache = {}
    output = Path(output_dir)
    reference_name = None
    search_records = []

    def score(candidate):
        return cross_validate(
            candidate, preprocessing, seed, threads, X, y, data["train"][2], folds
        )

    if mode == "baseline":
        experiments = settings["experiments"]
    elif mode == "ablations":
        experiments = ablations(settings["ablations"])
        reference_name = experiments[0]["name"]
    elif mode == "sequential":
        experiments = []

        def record(result):
            search_records.append(result)
            artifacts.write_json(output / "search.json", search_records)

        search_settings = settings["search"]
        references = search_settings["references"]
        if not references or set(references) - {"logistic", "lightgbm"}:
            raise ValueError("Search references must be logistic and/or lightgbm")
        for name, reference in references.items():
            if reference["model"] != name:
                raise ValueError("Search reference model does not match its key")
            best, _ = sequential(
                reference,
                search_settings,
                preprocessing,
                X,
                y,
                folds,
                seed,
                threads,
                record,
            )
            best["name"] = name + "_selected"
            experiments.append(best)
            cache[best["name"]] = score(best)
    else:
        raise ValueError(f"Unknown experiment mode: {mode}")
    names = [e["name"] for e in experiments]
    if not names or len(set(names)) != len(names):
        raise ValueError("Experiment names must be nonempty and unique")
    if any(
        Path(name).name != name or name in (".", "..") or "/" in name or "\\" in name
        for name in names
    ):
        raise ValueError("Experiment names must be plain directory names")
    results = []
    for experiment in experiments:
        directory = output / experiment["name"]
        directory.mkdir()
        artifacts.write_json(directory / "status.json", {"state": "running"})
        cv = cache.get(experiment["name"])
        if cv is None:
            cv = score(experiment)
        fold_results = cv["folds"]
        model, order, duration, pools = _fit(
            experiment, preprocessing, seed, threads, X, y
        )
        metrics, validation_counts = _evaluate_split(
            model,
            order,
            paths["validation"],
            boundaries["validation"],
            preprocessing,
            settings,
        )
        metrics["fit_seconds"] = duration
        size = artifacts.save_model(model, directory / filename, serialization)
        # Pipeline inspection belongs to artifact export, outside orchestration.
        diagnostics = artifacts.training_diagnostics(model, X)
        metrics.update(diagnostics)
        metrics["model_bytes"] = size
        summary = {
            "name": experiment["name"],
            "cv_macro_f1": cv["cv_macro_f1"],
            "cv_log_loss": cv["cv_log_loss"],
            "validation": metrics,
        }
        results.append(summary)
        effective = copy.deepcopy(config)
        resolved = artifacts.describe_pipeline(model)
        actual_experiment = copy.deepcopy(experiment)
        if "params" in resolved:
            actual_experiment["params"] = resolved["params"]
        effective["training"]["experiments"] = [actual_experiment]
        effective["resolved_model"] = resolved
        effective["resource_limits"] = {"native_threads": threads, "fit_pools": pools}
        artifacts.write_json(directory / "config.json", effective)
        artifacts.write_json(directory / "metrics.json", metrics)
        artifacts.write_json(directory / "cv.json", fold_results)
        artifacts.write_json(
            directory / "manifest.json",
            {
                **snapshot["provenance"],
                "state": "completed",
                "native_thread_limit": threads,
                "native_pools": pools,
                "seed": seed,
                "serialization": serialization,
                "model_bytes": size,
                "class_order": list(order),
                "splits": snapshot["splits"],
                "split_manifest": snapshot["split_manifest"],
                "counts": {"train": data["train"][3], "validation": validation_counts},
                "fit_seconds": duration,
                "inference_seconds": metrics["inference_seconds"],
            },
        )
        artifacts.write_json(directory / "status.json", {"state": "completed"})
    ranked = sorted(
        results, key=lambda r: (-r["cv_macro_f1"], r["cv_log_loss"], r["name"])
    )
    selected = ranked[0]["name"]
    report = {
        "selection": "CV macro F1 descending, log loss ascending",
        "selected": selected,
        "results": results,
    }
    if reference_name:
        reference_result = next(r for r in results if r["name"] == reference_name)
        report["reference"] = reference_name
        report["deltas"] = {
            r["name"]: {
                "cv_macro_f1": r["cv_macro_f1"] - reference_result["cv_macro_f1"],
                "cv_log_loss": r["cv_log_loss"] - reference_result["cv_log_loss"],
                "validation_macro_f1": r["validation"]["macro_f1"]
                - reference_result["validation"]["macro_f1"],
                "validation_log_loss": r["validation"]["log_loss"]
                - reference_result["validation"]["log_loss"],
            }
            for r in results
        }
    if mode == "sequential":
        report["search_candidates"] = len(search_records)
    if final_test:
        model = artifacts.load_model(output / selected / filename, serialization)
        manifest = json.loads((output / selected / "manifest.json").read_text())
        report["test"], report["test_counts"] = _evaluate_split(
            model,
            manifest["class_order"],
            paths["test"],
            boundaries["test"],
            preprocessing,
            settings,
        )
        report["test_identity"] = snapshot["splits"]["test"]
    if tracking is not None:
        report["mlflow"] = log_training(
            tracking, config, output, report, snapshot, list(X.columns)
        )
    report["state"] = "completed"
    artifacts.write_json(output / "summary.json", report)
    return report


def run(config, config_path, preprocessing, output_dir, final_test=False):
    """Publish explicit run state; only a completed root status denotes success."""
    if final_test and config["training"].get("mode", "baseline") != "baseline":
        raise ValueError("Final test is unavailable during ablations/search")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output}")
    output.mkdir(parents=True)
    artifacts.write_json(output / "status.json", {"state": "running"})
    try:
        settings = config["training"]
        tracking = load_tracking_config(config)
        if tracking is not None:
            check_server(tracking.tracking_uri)
        paths = load_published_splits(settings["manifest_path"])
        names = (
            ("train", "validation", "test") if final_test else ("train", "validation")
        )
        snapshot = {
            "paths": paths,
            "provenance": artifacts.provenance(config_path),
            "splits": {name: artifacts.file_identity(paths[name]) for name in names},
            "split_manifest": artifacts.file_identity(settings["manifest_path"]),
        }
        if settings.get("experiment_config_path"):
            snapshot["experiment_config"] = artifacts.file_identity(
                settings["experiment_config_path"]
            )
        artifacts.write_json(output / "requested_config.json", config)
        artifacts.write_json(
            output / "inputs.json", {k: v for k, v in snapshot.items() if k != "paths"}
        )
        report = _run(
            config,
            config_path,
            preprocessing,
            output_dir,
            snapshot,
            final_test,
            tracking,
        )
        artifacts.write_json(
            output / "status.json",
            {"state": "completed", "selected": report["selected"]},
        )
        return report
    except BaseException as error:
        for directory in output.iterdir():
            status = directory / "status.json"
            if directory.is_dir() and status.exists():
                state = json.loads(status.read_text(encoding="utf-8"))
                if state["state"] == "running":
                    artifacts.write_json(
                        status, {"state": "failed", "error": str(error)}
                    )
        artifacts.write_json(
            output / "status.json",
            {
                "state": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
