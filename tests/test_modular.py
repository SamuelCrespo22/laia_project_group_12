"""Small synthetic compositions; reuse the existing temporal integration fixture."""

import copy
import io
import json

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.preprocessor import CLASSES, load_config
from src.models import experiments
from src.models.factory import build_pipeline, create_estimator, create_model
from src.models.artifacts import describe_pipeline
from src.models.search import ablations, sequential
from src.models.train import main
from src.models.transformers import (
    FlightFeatures,
    OperationalFeatures,
    ZScoreOutliers,
)
from tests.test_training import environment as training_environment

environment = training_environment


@pytest.fixture
def records():
    n = 60
    X = pd.DataFrame(
        {
            "FL_DATE": pd.to_datetime(["2023-06-01"] * n),
            "CRS_DEP_TIME": [900 + i % 10 * 100 for i in range(n)],
            "CRS_ARR_TIME": [2100] * n,
            "CRS_ELAPSED_TIME": [None if i % 7 == 0 else 100.0 + i for i in range(n)],
            "DISTANCE": [None if i % 11 == 0 else 200.0 + i * 10 for i in range(n)],
            "OP_UNIQUE_CARRIER": ["AA", "DL"] * 30,
            "ORIGIN": [f"A{i:03}" for i in range(n)],
            "DEST": ["LAX"] * n,
        }
    )
    y = pd.Series(["on_time"] * 40 + ["moderate_delay"] * 10 + ["severe_delay"] * 10)
    return X, y


def modular():
    return {
        "name": "reference",
        "model": "logistic",
        "params": {"max_iter": 1000},
        "scaling": "standard",
        "features": {
            "mode": "modular",
            "outliers": {"method": "passthrough"},
            "encoding": "onehot",
            "missing": {"method": "simple", "strategy": "median"},
            "selection": {"method": "passthrough"},
        },
    }


def test_builder_accepts_estimator_without_model_registration(records):
    from sklearn.naive_bayes import GaussianNB

    X, y = records
    estimator = GaussianNB()
    pipeline = build_pipeline(estimator, {"scaling": "standard"}, load_config())
    assert type(pipeline) is Pipeline
    assert pipeline["model"] is estimator
    assert list(pipeline.named_steps) == [
        "features",
        "outliers",
        "engineering",
        "tsfresh",
        "columns",
        "selection",
        "sampling",
        "model",
    ]
    assert all(
        pipeline[name] == "passthrough"
        for name in (
            "outliers",
            "engineering",
            "tsfresh",
            "selection",
            "sampling",
        )
    )
    pipeline.set_params(model__var_smoothing=1e-8)
    pipeline.fit(X, y)
    assert pipeline["model"].var_smoothing == 1e-8
    assert pipeline.predict_proba(X).shape == (len(X), len(CLASSES))


@pytest.mark.parametrize(
    "missing,scaling,encode,selection,sampling",
    [
        ("simple", "standard", "onehot", "passthrough", "passthrough"),
        ("knn", "robust", "ordinal", "kbest", "over"),
        ("simple", "minmax", "onehot", "percentile", "under"),
        ("knn", "passthrough", "onehot", "passthrough", "over"),
    ],
)
def test_native_composition_sparse_sampling_serialization(
    records, missing, scaling, encode, selection, sampling
):
    X, y = records
    spec = modular()
    spec["scaling"] = scaling
    spec["features"].update(
        outliers={"method": "zscore", "threshold": 3},
        missing={"method": missing, "n_neighbors": 3},
        binning={
            "enabled": True,
            "columns": ["DISTANCE"],
            "n_bins": 3,
            "encode": encode,
        },
        selection={"method": selection, "k": 5, "percentile": 50},
    )
    spec["sampling"] = {"method": sampling}
    model, order = create_model(spec, load_config(), 42, 1, CLASSES)
    model = clone(model)
    if missing == "knn":
        model.set_params(columns__numeric__imputer__n_neighbors=2)
    model.fit(X, y)
    transformed = model["columns"].transform(model[:3].transform(X))
    assert sparse.issparse(transformed)
    if sampling != "passthrough":
        count = len(model["sampling"].sample_indices_)
        assert count == (120 if sampling == "over" else 30)
    future = X.iloc[:3].copy()
    future["ORIGIN"] = "UNSEEN"
    future["OP_UNIQUE_CARRIER"] = None
    proba = model.predict_proba(future)
    assert proba.shape == (3, 3) and np.isfinite(proba).all()
    assert len(model.predict(future)) == 3
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    np.testing.assert_array_equal(proba, joblib.load(buffer).predict_proba(future))
    assert order == tuple(sorted(CLASSES))
    # Export must support dtype classes and missing-value sentinels from native objects.
    json.dumps(describe_pipeline(model), allow_nan=False)


@pytest.mark.parametrize("index", range(5))
def test_baseline_native_equivalence(records, index):
    X, y = records
    spec = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))["training"][
        "experiments"
    ][index]
    preprocessing = load_config()
    model, _ = create_model(spec, preprocessing, 42, 1, CLASSES)
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "scaler",
                StandardScaler() if spec["model"] == "logistic" else "passthrough",
            ),
        ]
    )
    # Frozen original constructor choices: verify the new explicit YAML defaults
    # preserve the previous automatic multiclass backends and predictions.
    if spec["model"] == "dummy":
        original_params = {"strategy": "prior"}
    elif spec["model"] == "logistic":
        original_params = {
            "max_iter": 500,
            "class_weight": spec["params"]["class_weight"],
        }
    else:
        original_params = {
            "n_estimators": 100,
            "num_leaves": 31,
            "learning_rate": 0.1,
            "class_weight": spec["params"]["class_weight"],
            "verbosity": -1,
        }
    original = Pipeline(
        [
            ("features", FlightFeatures(preprocessing)),
            ("engineering", OperationalFeatures()),
            (
                "columns",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            numeric,
                            [
                                "MONTH",
                                "DAY_OF_WEEK",
                                "CRS_DEP_TIME",
                                "CRS_ARR_TIME",
                                "CRS_ELAPSED_TIME",
                                "DISTANCE",
                                "DISTANCE_GROUP",
                            ],
                        )
                    ],
                    remainder="drop",
                ),
            ),
            ("model", create_estimator(spec["model"], original_params, 42, 1)),
        ]
    )
    model.fit(X, y)
    original.fit(X, y)
    np.testing.assert_array_equal(model.predict(X), original.predict(X))
    np.testing.assert_array_equal(model.predict_proba(X), original.predict_proba(X))
    adapter = FlightFeatures(preprocessing)
    del (
        adapter.history_column
    )  # artifacts created before the optional parameter existed
    pd.testing.assert_frame_equal(
        adapter.transform(X), FlightFeatures(preprocessing).transform(X)
    )
    b = io.BytesIO()
    joblib.dump(adapter, b)
    b.seek(0)
    assert joblib.load(b).get_params()["history_column"] is None


def test_zscore_train_only_preserves_rows_and_nans():
    train = pd.DataFrame({"DISTANCE": [1.0, 2.0, 3.0, np.nan]}, index=[4, 4, 8, 1])
    transform = ZScoreOutliers(["DISTANCE"], threshold=2).fit(train)
    mean = transform.scaler_.mean_.copy()
    out = transform.transform(
        pd.DataFrame({"DISTANCE": [1000.0, 2.0, np.nan]}, index=[9, 9, 3])
    )
    assert out.index.tolist() == [9, 9, 3]
    assert out.DISTANCE.isna().tolist() == [True, False, True]
    np.testing.assert_array_equal(mean, transform.scaler_.mean_)


def test_tsfresh_real_history_causal_and_serializable(records):
    X, y = records
    X = X.iloc[:12].copy()
    y = pd.Series(list(CLASSES) * 4)
    X["history"] = [
        [
            {"timestamp": "2023-06-01 07:00", "value": float(i)},
            {"timestamp": "2023-06-01 08:00", "value": float(i + 2)},
        ]
        for i in range(len(X))
    ]
    spec = modular()
    spec["features"]["tsfresh"] = {
        "enabled": True,
        "history_column": "history",
        "statistics": ["mean", "standard_deviation", "length"],
    }
    model, _ = create_model(spec, load_config(), 42, 1, CLASSES)
    model.fit(X, y)
    features = model["tsfresh"].transform(model["features"].transform(X))
    np.testing.assert_allclose(features.tsfresh_mean, np.arange(12) + 1)
    np.testing.assert_allclose(features.tsfresh_standard_deviation, 1)
    np.testing.assert_allclose(features.tsfresh_length, 2)
    with pytest.raises(ValueError, match="explicit history"):
        model.predict(X.drop(columns="history"))
    future = X.copy()
    future.at[0, "history"] = [{"timestamp": "2023-06-02 00:00", "value": 9999.0}]
    with pytest.raises(ValueError, match="strictly before departure"):
        model.predict(future)
    b = io.BytesIO()
    joblib.dump(model, b)
    b.seek(0)
    np.testing.assert_allclose(model.predict_proba(X), joblib.load(b).predict_proba(X))


def test_ablations_one_family_and_sequential_propagation():
    reference = modular()
    specs = ablations(
        {
            "reference": reference,
            "families": {"normalization": ["robust"], "missing": [{"method": "knn"}]},
        }
    )
    assert specs[1]["features"] == reference["features"]
    assert specs[2]["scaling"] == reference["scaling"]
    assert reference == modular()
    observed = []

    def score(spec):
        observed.append(copy.deepcopy(spec))
        value = (
            int(spec["scaling"] == "robust")
            + int(spec["features"].get("selection", {}).get("method") == "kbest")
            + int(spec["params"].get("C") == 2)
        )
        return {"cv_macro_f1": value / 4, "cv_log_loss": 1.0 / (value + 1), "folds": []}

    history = []
    best, _ = sequential(
        reference,
        {
            "budget": 1,
            "spaces": {
                "A": {"normalization": ["robust"]},
                "B": {"selection": [{"method": "kbest", "k": 5}]},
                "C": {"params.C": [2]},
            },
            "stages": ["A", "B", "C"],
        },
        score,
        42,
        history.append,
    )
    assert best["scaling"] == "robust"
    assert (
        best["features"]["selection"]["method"] == "kbest" and best["params"]["C"] == 2
    )
    assert "C" not in reference["params"]
    assert len(history) == 6 and len(observed) == 4
    assert all("mask" not in record["config"] for record in history)


@pytest.mark.parametrize("mode", ["ablations", "sequential"])
def test_modes_use_same_cv_without_test_or_validation_selection(
    environment, tmp_path, monkeypatch, mode
):
    config_path, config, _, manifest = environment
    reference = modular()
    reference["features"]["encoding"] = "passthrough"
    if mode == "ablations":
        config["training"]["ablations"] = {
            "reference": reference,
            "families": {"normalization": ["robust"]},
        }
    else:
        lightgbm = copy.deepcopy(reference)
        lightgbm.update(
            model="lightgbm",
            params={"n_estimators": 3, "verbosity": -1, "min_child_samples": 2},
        )
        config["training"]["search"] = {
            "stages": ["A", "B", "C"],
            "budget": 1,
            "references": {"logistic": reference, "lightgbm": lightgbm},
            "spaces": {
                "A": {"normalization": ["robust"]},
                "B": {"selection": [{"method": "percentile", "percentile": 100}]},
                "C": {"imbalance": [{"method": "weights", "class_weight": "balanced"}]},
            },
        }
    config_path.write_text(yaml.safe_dump(config))
    # Never inspect held-out contents in either automatic mode.
    from pathlib import Path

    Path(manifest["splits"]["test"]["path"]).write_bytes(b"not a parquet")
    calls, validations = [], []
    original = experiments.cross_validate
    validate = experiments._evaluate_split

    def cv(*args):
        assert not validations  # all search selection happens before fixed validation
        calls.append(args[0]["model"])
        return original(*args)

    def validation(*args):
        validations.append(args[0])
        assert Path(args[2]).name == "validation.parquet"
        return validate(*args)

    if mode == "sequential":
        monkeypatch.setattr(experiments, "cross_validate", cv)
        monkeypatch.setattr(experiments, "_evaluate_split", validation)
    output = tmp_path / mode
    report = main(
        ["--config", str(config_path), "--mode", mode, "--output-dir", str(output)]
    )
    assert len(report["results"]) == 2 and "test" not in report
    if mode == "ablations":
        assert report["reference"] == "reference" and len(report["deltas"]) == 2
    else:
        assert set(calls) == {"logistic", "lightgbm"} and len(validations) == 2
        logs = json.loads((output / "search.json").read_text())
        assert len(logs) == 12 and all(r["state"] == "completed" for r in logs)
        assert all("validation" not in r or r["validation"] is None for r in logs)
        before = [(r["cv_macro_f1"], r["cv_log_loss"]) for r in report["results"]]
        raw = pd.read_parquet(manifest["splits"]["validation"]["path"])
        raw["ARR_DELAY"] = 99999.0
        raw.to_parquet(manifest["splits"]["validation"]["path"], index=False)
        validations.clear()
        second = main(
            [
                "--config",
                str(config_path),
                "--mode",
                mode,
                "--output-dir",
                str(tmp_path / "changed_validation"),
            ]
        )
        assert report["selected"] == second["selected"]
        assert before == [
            (r["cv_macro_f1"], r["cv_log_loss"]) for r in second["results"]
        ]
    with pytest.raises(ValueError, match="Final test"):
        main(
            [
                "--config",
                str(config_path),
                "--mode",
                mode,
                "--final-test",
                "--output-dir",
                str(tmp_path / "forbidden"),
            ]
        )


def test_history_projects_and_aligns_after_filtering(environment, tmp_path):
    path, config, preprocessing, manifest = environment
    from pathlib import Path
    from src.models.dataset import prepare_split

    raw = pd.read_parquet(manifest["splits"]["train"]["path"])
    raw["history"] = [
        [{"timestamp": str(date - pd.Timedelta(hours=1)), "value": float(i)}]
        for i, date in enumerate(raw.FL_DATE)
    ]
    source = Path(manifest["splits"]["train"]["path"])
    raw.to_parquet(source, index=False)
    X, y, _, _ = prepare_split(
        source,
        {"start": "2023-06-01", "end": "2023-07-01"},
        preprocessing,
        extra_columns=["history"],
    )
    assert len(X) == len(y) == 72 and "history" in X
    assert all(len(h) == 1 for h in X.history)
    config["training"]["history_column"] = "history"
    spec = modular()
    spec["features"]["tsfresh"] = {
        "enabled": True,
        "history_column": "history",
        "statistics": ["mean", "length"],
    }
    config["training"]["experiments"] = [spec]
    for name in ("validation", "test"):
        part = pd.read_parquet(manifest["splits"][name]["path"])
        part["history"] = [
            [{"timestamp": str(d - pd.Timedelta(hours=1)), "value": 5.0}]
            for d in part.FL_DATE
        ]
        part.to_parquet(manifest["splits"][name]["path"], index=False)
    path.write_text(yaml.safe_dump(config))
    report = main(
        ["--config", str(path), "--output-dir", str(tmp_path / "history_run")]
    )
    assert report["results"][0]["validation"]["support"] == 72


@pytest.mark.parametrize("stage", ["A", "B", "C"])
def test_isolated_stages_deterministic_and_log_loss_tie_break(stage):
    reference = modular()
    settings = {
        "stages": [stage],
        "budget": 1,
        "spaces": {
            "A": {"normalization": ["robust"]},
            "B": {"selection": [{"method": "percentile", "percentile": 100}]},
            "C": {"params.C": [2]},
        },
    }

    def score(spec):
        changed = spec != reference
        return {"cv_macro_f1": 0.3, "cv_log_loss": 0.5 if changed else 1.0, "folds": []}

    a, b = [], []
    first = sequential(reference, settings, score, 42, a.append)
    second = sequential(reference, settings, score, 42, b.append)
    assert first == second and a == b
    assert first[1]["cv_log_loss"] == 0.5


def test_cli_training_only_overlay(environment, tmp_path):
    path, config, _, _ = environment
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "training": {
                    "ablations": {
                        "reference": modular(),
                        "families": {"normalization": ["robust"]},
                    }
                }
            }
        )
    )
    output = tmp_path / "overlay_run"
    report = main(
        [
            "--config",
            str(path),
            "--experiment-config",
            str(overlay),
            "--mode",
            "ablations",
            "--output-dir",
            str(output),
        ]
    )
    assert len(report["results"]) == 2
    effective = json.loads((output / "requested_config.json").read_text())
    assert effective["temporal_split"] == config["temporal_split"]
    inputs = json.loads((output / "inputs.json").read_text())
    assert inputs["experiment_config"]["path"] == str(overlay.resolve())
    overlay.write_text(yaml.safe_dump({"data": {"raw_dir": "elsewhere"}}))
    with pytest.raises(ValueError, match="only change training"):
        main(["--config", str(path), "--experiment-config", str(overlay)])


def test_invalid_search_candidate_is_recorded_without_losing_reference():
    reference = modular()
    logs = []

    def score(spec):
        if spec != reference:
            raise ValueError("invalid native selector configuration")
        return {"cv_macro_f1": 0.3, "cv_log_loss": 0.8, "folds": []}

    best, result = sequential(
        reference,
        {
            "stages": ["B"],
            "budget": 1,
            "spaces": {"B": {"selection": [{"method": "kbest", "k": 999}]}},
        },
        score,
        42,
        logs.append,
    )
    assert best == reference and result["cv_macro_f1"] == 0.3
    assert logs[-1]["state"] == "failed"
    assert logs[-1]["config"]["features"]["selection"]["k"] == 999


def test_single_encoded_artifact_accepts_api_contract(records, tmp_path):
    """Prove inference compatibility without changing/calling the placeholder API."""
    from src.api.schemas import PredictionRequest, PredictionResponse
    from src.data.contract import records_to_frame
    from src.models.artifacts import save_model, load_model

    X, y = records
    model, order = create_model(modular(), load_config(), 42, 1, CLASSES)
    model.fit(X, y)
    request = PredictionRequest(
        instances=[
            {
                "FL_DATE": "2023-07-01",
                "OP_UNIQUE_CARRIER": "ZZ",
                "ORIGIN": "XXX",
                "DEST": "LAX",
                "CRS_DEP_TIME": 900,
            },
            {
                "FL_DATE": "2023-07-02",
                "OP_UNIQUE_CARRIER": "AA",
                "ORIGIN": "JFK",
                "DEST": "ORD",
                "CRS_DEP_TIME": 1800,
            },
        ]
    )
    frame = records_to_frame(request.instances)
    expected_labels, expected_proba = model.predict(frame), model.predict_proba(frame)
    path = tmp_path / "complete_encoded_pipeline.joblib"
    save_model(model, path)
    restored = load_model(path)
    np.testing.assert_array_equal(expected_labels, restored.predict(frame))
    np.testing.assert_array_equal(expected_proba, restored.predict_proba(frame))
    response = PredictionResponse(classifications=restored.predict(frame).tolist())
    assert len(response.classifications) == 2
    assert order == tuple(sorted(CLASSES))
    assert set(restored.named_steps["columns"].named_transformers_) >= {
        "numeric",
        "categorical",
    }
