"""Synthetic-only training integration; no BTS, network, API server or MLflow."""

import copy
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

from scripts.create_temporal_splits import assign_batch
from src.data.preprocessor import (
    CLASSES,
    build_features,
    build_training_frame,
    load_config,
)
from src.data.temporal_split import load_split_config
from src.models.artifacts import load_model, save_model, training_diagnostics
from src.models.evaluation import evaluate
from src.models.experiments import expanding_folds, prepare_split
from src.models.factory import create_model
from src.models.train import main
from src.models.transformers import RouteIQR


@pytest.fixture
def environment(tmp_path):
    config = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    config["training"]["cv_splits"] = 2
    rows = []
    for start in ("2023-06-01", "2023-07-01", "2023-10-01"):
        for day in pd.date_range(start, periods=12):
            for index, delay in enumerate([0, 14, 15, 59, 60, 120, 5, 5, None, 5, 5]):
                rows.append(
                    {
                        "FL_DATE": day,
                        "OP_UNIQUE_CARRIER": "AA" if start.endswith("06-01") else "ZZ",
                        "ORIGIN": "JFK" if index % 2 else "XXX",
                        "DEST": "LAX",
                        "CRS_DEP_TIME": 900 + index * 100,
                        "CRS_ARR_TIME": 2100,
                        "CRS_ELAPSED_TIME": 300.0 if index % 2 else None,
                        "DISTANCE": 1000.0 + index * 20,
                        "ARR_DELAY": delay,
                        "CANCELLED": 1 if index == 6 else None if index == 9 else 0,
                        "DIVERTED": 1 if index == 7 else None if index == 10 else 0,
                        "DEP_DELAY": 9999.0,
                        "TAXI_OUT": 9999.0,
                    }
                )
    raw = pd.DataFrame(rows)
    parts = assign_batch(pa.Table.from_pandas(raw).to_batches()[0], load_split_config())
    manifest = {"splits": {}}
    for name, table in parts.items():
        path = tmp_path / f"{name}.parquet"
        table.to_pandas().sample(frac=1, random_state=1).to_parquet(path, index=False)
        manifest["splits"][name] = {"path": str(path), "rows": table.num_rows}
    manifest_path = tmp_path / "temporal_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config["training"]["manifest_path"] = str(manifest_path)
    for experiment in config["training"]["experiments"]:
        if experiment["model"] == "lightgbm":
            experiment["params"]["n_estimators"] = 3
            experiment["params"]["min_child_samples"] = 2
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, config, load_config(path), manifest


def prepared(environment, name="train"):
    path, config, preprocessing, manifest = environment
    return prepare_split(
        manifest["splits"][name]["path"], load_split_config(path)[name], preprocessing
    )


def test_end_to_end_five_baselines(environment, tmp_path):
    path, _, _, _ = environment
    output = tmp_path / "run"
    report = main(["--config", str(path), "--output-dir", str(output), "--final-test"])
    assert len(report["results"]) == 5
    assert report["test"]["support"] == 72
    for result in report["results"]:
        directory = output / result["name"]
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["serialization"] == "joblib"
        assert manifest["counts"]["train"] == {
            "raw": 132,
            "eligible": 72,
            "used": 72,
            "classes": {c: 24 for c in CLASSES},
        }
        assert result["validation"]["transformed_features"] == 7
        assert manifest["model_bytes"] == (directory / "pipeline.joblib").stat().st_size
        restored = load_model(directory / "pipeline.joblib")
        X, y, metadata, _ = prepared(environment, "validation")
        metrics = evaluate(
            y,
            restored.predict(X),
            restored.predict_proba(X),
            manifest["class_order"],
            metadata,
        )
        assert metrics["macro_f1"] == result["validation"]["macro_f1"]
        folds = json.loads((directory / "cv.json").read_text())
        assert len(folds) == 2
        assert all(f["train_end"] < f["validation_start"] for f in folds)


def test_temporal_filtering_and_alignment(environment):
    X, y, meta, counts = prepared(environment)
    assert X.FL_DATE.is_monotonic_increasing
    assert X.index.equals(y.index) and X.index.equals(meta.index)
    assert counts["eligible"] == 72
    assert not {"ARR_DELAY", "DEP_DELAY", "TAXI_OUT", "CANCELLED", "DIVERTED"} & set(X)
    previous = set()
    for a, b in expanding_folds(X, 2):
        train_days, validation_days = set(X.iloc[a].FL_DATE), set(X.iloc[b].FL_DATE)
        assert train_days.isdisjoint(validation_days)
        assert max(train_days) < min(validation_days)
        assert previous <= train_days
        previous = train_days


@pytest.mark.parametrize("scaling", ["standard", "robust", "minmax", "passthrough"])
def test_scaling_train_only_serialization_and_seed(environment, tmp_path, scaling):
    _, config, preprocessing, _ = environment
    experiment = copy.deepcopy(config["training"]["experiments"][1])
    experiment["scaling"] = scaling
    X, y, _, _ = prepared(environment)
    model, order = create_model(experiment, preprocessing, 42, 1, CLASSES)
    model.fit(X, y)
    assert order == tuple(sorted(CLASSES))
    columns = model.named_steps["columns"]
    assert set(columns.transformers_[0][2]).isdisjoint(
        {"ORIGIN", "DEST", "OP_UNIQUE_CARRIER", "FL_DATE", "DEP_TIME_BLK"}
    )
    imputer = columns.named_transformers_["numeric"].named_steps["imputer"]
    before = imputer.statistics_.copy()
    changed = X.copy()
    changed["DISTANCE"] = 1e9
    changed["OP_UNIQUE_CARRIER"] = "UNSEEN"
    model.predict_proba(changed)
    np.testing.assert_array_equal(before, imputer.statistics_)
    np.testing.assert_allclose(before, X[columns.transformers_[0][2]].median())
    path = tmp_path / f"{scaling}.joblib"
    save_model(model, path)
    loaded = load_model(path)
    np.testing.assert_allclose(
        model.predict_proba(changed), loaded.predict_proba(changed)
    )
    other, _ = create_model(experiment, preprocessing, 42, 1, CLASSES)
    other.fit(X, y)
    np.testing.assert_allclose(model.predict_proba(X), other.predict_proba(X))
    # Actual classes, not target config order, define probability column mapping.
    np.testing.assert_array_equal(model.named_steps["model"].classes_, order)


def test_outlier_fallback_and_train_statistics():
    X = pd.DataFrame(
        {
            "ORIGIN": ["A"] * 4 + ["B"],
            "DEST": ["C"] * 5,
            "DISTANCE": [1.0, 2.0, 3.0, 4.0, 100.0],
        }
    )
    for action in ("flag", "deviation"):
        transformer = RouteIQR(
            columns=["DISTANCE"], min_group_size=3, action=action
        ).fit(X)
        stats = copy.deepcopy(transformer.global_)
        unseen = pd.DataFrame(
            {
                "ORIGIN": ["Z", "A", "B"],
                "DEST": ["C"] * 3,
                "DISTANCE": [1000.0, 1000.0, 1000.0],
            }
        )
        out = transformer.transform(unseen)
        assert len(out) == len(unseen)
        assert (out.iloc[:, -1] > 0).all()
        assert stats == transformer.global_
        assert ("B", "C") not in transformer.routes_["DISTANCE"]
        if action == "deviation":
            assert out.iloc[0, -1] == 1000 - transformer.global_["DISTANCE"][1]


def test_optional_engineering(environment):
    _, config, preprocessing, _ = environment
    experiment = copy.deepcopy(config["training"]["experiments"][1])
    experiment["features"]["engineering"] = {"cyclic": True, "interactions": True}
    experiment["features"]["outliers"]["enabled"] = True
    experiment["features"]["selection"] = {"enabled": True, "k": 5}
    X, y, _, _ = prepared(environment)
    model, _ = create_model(experiment, preprocessing, 42, 1, CLASSES)
    model.fit(X, y)
    assert training_diagnostics(model, X)["transformed_features"] == 5
    assert np.isfinite(model.predict_proba(X)).all()


def test_metrics_and_subgroups():
    classes = ["b", "a", "c"]
    y = np.array(["a", "a", "b", "b", "c", "c"])
    pred = np.array(["a", "b", "b", "b", "a", "c"])
    p = np.full((6, 3), 0.1)
    for i, label in enumerate(pred):
        p[i, classes.index(label)] = 0.8
    metadata = pd.DataFrame(
        {"OP_UNIQUE_CARRIER": ["AA"] * 3 + ["BB"] * 3, "ORIGIN": ["X"] * 6}
    )
    result = evaluate(y, pred, p, classes, metadata)
    assert result["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert result["accuracy"] == accuracy_score(y, pred)
    assert result["balanced_accuracy"] == balanced_accuracy_score(y, pred)
    assert result["log_loss"] == pytest.approx(
        log_loss(y, p[:, [1, 0, 2]], labels=sorted(classes))
    )
    assert result["per_class"]["a"]["support"] == 2
    assert result["subgroups"]["OP_UNIQUE_CARRIER"]["AA"]["support"] == 3
    assert (
        result["subgroups"]["ORIGIN"]["X"]["confusion_matrix"]
        == result["confusion_matrix"]
    )


def test_no_test_read_without_opt_in(environment, tmp_path):
    path, config, _, manifest = environment
    # Existing but invalid test file must not be opened during model selection.
    from pathlib import Path

    Path(manifest["splits"]["test"]["path"]).write_bytes(b"not parquet")
    config["training"]["experiments"] = config["training"]["experiments"][:1]
    path.write_text(yaml.safe_dump(config))
    report = main(
        ["--config", str(path), "--output-dir", str(tmp_path / "without_test")]
    )
    assert "test" not in report


@pytest.mark.parametrize("column", ["DEP_DELAY", "ORIGIN", "FL_DATE"])
def test_forbidden_numeric_selection(environment, column):
    _, config, preprocessing, _ = environment
    experiment = copy.deepcopy(config["training"]["experiments"][0])
    experiment["features"]["columns"] = [column]
    with pytest.raises(ValueError, match="numeric"):
        create_model(experiment, preprocessing, 42, 1, CLASSES)


def test_schema_date_validation_and_feature_leakage(environment):
    _, _, preprocessing, manifest = environment
    path = manifest["splits"]["train"]["path"]
    raw = pd.read_parquet(path)
    changed = raw.copy()
    changed["ARR_DELAY"] = 9999.0
    changed["DEP_DELAY"] = -9999.0
    pd.testing.assert_frame_equal(
        build_features(raw, preprocessing), build_features(changed, preprocessing)
    )
    bad_period = {"start": "2023-07-01", "end": "2023-10-01"}
    with pytest.raises(ValueError, match="dates"):
        prepare_split(path, bad_period, preprocessing)
    raw.drop(columns="ARR_DELAY").to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Missing raw"):
        prepare_split(path, bad_period, preprocessing)


def test_seeded_sampling(environment):
    path, _, preprocessing, manifest = environment
    period = load_split_config(path)["train"]
    args = (manifest["splits"]["train"]["path"], period, preprocessing)
    first = prepare_split(*args, sample_size=40, seed=7)
    second = prepare_split(*args, sample_size=40, seed=7)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_series_equal(first[1], second[1])


def test_runner_uses_model_contract(environment, tmp_path, monkeypatch):
    """Runner calls fit/predict/predict_proba without framework dispatch."""
    from src.models import artifacts, experiments

    instances = []

    from sklearn.base import BaseEstimator, ClassifierMixin

    class TestModel(ClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            self.train_days = set(X.FL_DATE)
            self.classes_ = np.array(CLASSES)
            self.priors = np.array([np.mean(y == c) for c in CLASSES])
            instances.append(self)
            return self

        def predict(self, X):
            assert self.train_days.isdisjoint(set(X.FL_DATE))
            return np.repeat(CLASSES[int(self.priors.argmax())], len(X))

        def predict_proba(self, X):
            return np.tile(self.priors, (len(X), 1))

    monkeypatch.setattr(
        experiments, "create_model", lambda *args: (TestModel(), CLASSES)
    )
    monkeypatch.setattr(artifacts, "training_diagnostics", lambda *args: {})
    monkeypatch.setattr(artifacts, "describe_pipeline", lambda *args: {})

    def test_persistence(model, path, format="joblib"):
        path.write_bytes(b"test-only-serializer")
        return path.stat().st_size

    monkeypatch.setattr(artifacts, "save_model", test_persistence)
    path, config, _, _ = environment
    config["training"]["experiments"] = [{"name": "contract", "model": "test"}]
    path.write_text(yaml.safe_dump(config))
    main(["--config", str(path), "--output-dir", str(tmp_path / "contract_run")])
    assert len(instances) == 3
    assert len({id(model) for model in instances}) == 3


def test_missing_lightgbm_fails_explicitly(environment, monkeypatch):
    import sys

    _, config, preprocessing, _ = environment
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    with pytest.raises(RuntimeError, match="LightGBM required"):
        create_model(
            config["training"]["experiments"][3], preprocessing, 42, 1, CLASSES
        )


def test_outlier_artifact_serialization(environment, tmp_path):
    _, config, preprocessing, _ = environment
    experiment = copy.deepcopy(config["training"]["experiments"][1])
    experiment["features"]["outliers"]["enabled"] = True
    experiment["features"]["outliers"]["action"] = "deviation"
    X, y, _, _ = prepared(environment)
    model, _ = create_model(experiment, preprocessing, 42, 1, CLASSES)
    model.fit(X, y)
    path = tmp_path / "route.joblib"
    save_model(model, path)
    unknown = X.copy()
    unknown["ORIGIN"] = "NEW"
    unknown["DISTANCE"] = 1e6
    np.testing.assert_allclose(
        model.predict_proba(unknown), load_model(path).predict_proba(unknown)
    )


def test_invalid_probability_contract():
    with pytest.raises(ValueError, match="probabilities"):
        evaluate(["a"], ["a"], [[0.5, 0.8]], ["a", "b"])
    with pytest.raises(ValueError, match="Metadata"):
        evaluate(["a"], ["a"], [[1.0, 0.0]], ["a", "b"], pd.DataFrame())


def test_frozen_features_and_minimal_request(environment, tmp_path):
    from src.data.contract import FlightRecord, records_to_frame

    path, config, preprocessing, _ = environment
    model, _ = create_model(
        config["training"]["experiments"][1], preprocessing, 42, 1, CLASSES
    )
    X, y, _, _ = prepared(environment)
    model.fit(X, y)
    request = records_to_frame(
        [
            FlightRecord(
                FL_DATE="2023-07-01",
                OP_UNIQUE_CARRIER="NEW",
                ORIGIN="ZZZ",
                DEST="XXX",
                CRS_DEP_TIME=900,
            )
        ]
    )
    expected = model.predict_proba(request)
    artifact = tmp_path / "frozen.joblib"
    save_model(model, artifact)
    config["preprocessing"]["feature_allowlist"] = ["DEP_DELAY"]
    path.write_text(yaml.safe_dump(config))
    np.testing.assert_allclose(expected, load_model(artifact).predict_proba(request))


def test_sample_before_features_projected_batches(environment, monkeypatch):
    from src.models import dataset

    _, _, preprocessing, manifest = environment
    path = manifest["splits"]["train"]["path"]
    original = dataset.pq.ParquetFile
    original_builder = dataset.build_training_frame
    batches, materialized = [], []

    class Reader:
        def __init__(self, path):
            self.reader = original(path)
            self.schema_arrow = self.reader.schema_arrow

        def iter_batches(self, batch_size, columns, use_threads):
            assert "DEP_DELAY" not in columns
            assert use_threads is False
            for batch in self.reader.iter_batches(
                batch_size=batch_size, columns=columns, use_threads=use_threads
            ):
                batches.append(batch.num_rows)
                yield batch

    def build(raw, config):
        materialized.append(len(raw))
        return original_builder(raw, config)

    monkeypatch.setattr(dataset.pq, "ParquetFile", Reader)
    monkeypatch.setattr(dataset, "build_training_frame", build)
    monkeypatch.setattr(
        pd, "read_parquet", lambda *a, **k: pytest.fail("Full read forbidden")
    )
    period = {"start": "2023-06-01", "end": "2023-07-01"}
    first = prepare_split(path, period, preprocessing, 40, 7, batch_size=7)
    second = prepare_split(path, period, preprocessing, 40, 7, batch_size=11)
    assert materialized == [40, 40]
    assert max(batches) <= 11
    pd.testing.assert_frame_equal(first[0], second[0])
    counts = first[0].FL_DATE.dt.normalize().value_counts()
    assert len(counts) == 12 and counts.max() - counts.min() <= 1
    with pytest.raises(ValueError, match="every eligible training day"):
        prepare_split(path, period, preprocessing, 3, batch_size=7)
    with pytest.raises(MemoryError, match="memory budget"):
        prepare_split(
            path, period, preprocessing, 40, batch_size=7, memory_budget_mb=0.00001
        )


@pytest.mark.parametrize(
    "column,value",
    [
        ("CRS_DEP_TIME", 1260),
        ("CRS_DEP_TIME", 2401),
        ("CRS_DEP_TIME", -1),
        ("CRS_DEP_TIME", 900.5),
        ("CRS_DEP_TIME", None),
        ("CRS_ARR_TIME", 1260),
        ("DISTANCE", -1),
        ("DISTANCE", np.inf),
        ("CRS_ELAPSED_TIME", np.inf),
        ("CRS_DEP_TIME", -np.inf),
    ],
)
def test_schedule_invalid_values(column, value):
    from src.models.transformers import validate_scheduled_features

    frame = pd.DataFrame(
        {
            "CRS_DEP_TIME": [900],
            "CRS_ARR_TIME": [2400],
            "DISTANCE": [100.0],
            "CRS_ELAPSED_TIME": [60.0],
        }
    )
    frame[column] = value
    with pytest.raises(ValueError):
        validate_scheduled_features(frame)


def test_schedule_missing_duration_extremes_and_severe_labels(environment):
    from src.models.transformers import validate_scheduled_features

    frame = pd.DataFrame(
        {
            "CRS_DEP_TIME": [0, 2400, 2359, 900],
            "CRS_ARR_TIME": [None] * 4,
            "DISTANCE": [0, 1e9, None, 10],
            "CRS_ELAPSED_TIME": [0, -1, None, 1e9],
        },
        index=[8, 3, 3, 9],
    )
    result = validate_scheduled_features(frame)
    assert result.index.equals(frame.index)
    assert result.CRS_ELAPSED_TIME.iloc[:3].isna().all()
    assert result.CRS_ELAPSED_TIME.iloc[3] == 1e9
    _, _, preprocessing, manifest = environment
    raw = pd.read_parquet(manifest["splits"]["train"]["path"])
    valid = raw[raw.CANCELLED.eq(0) & raw.DIVERTED.eq(0) & raw.ARR_DELAY.notna()].copy()
    valid["ARR_DELAY"] = 1e9
    assert (
        build_training_frame(valid, preprocessing)
        .delay_severity.eq("severe_delay")
        .all()
    )


def test_zero_iqr_unknown_routes_duplicate_indices():
    train = pd.DataFrame(
        {"ORIGIN": ["A"] * 4, "DEST": ["B"] * 4, "DISTANCE": [10.0] * 4},
        index=[8, 3, 3, 9],
    )
    probe = pd.DataFrame(
        {
            "ORIGIN": ["A", "Z", None, "A"],
            "DEST": ["B"] * 4,
            "DISTANCE": [10.0, 11.0, np.nan, 11.0],
        },
        index=[9, 2, 2, 7],
    )
    for action in ("flag", "deviation"):
        transformer = RouteIQR(
            columns=["DISTANCE"], min_group_size=3, action=action
        ).fit(train)
        out = transformer.transform(probe)
        pd.testing.assert_index_equal(out.index, probe.index)
        assert out.iloc[:, -1].tolist() == [0.0, 1.0, 0.0, 1.0]
        assert len(out) == len(probe)


def test_fold_statistics_and_native_threads(environment, monkeypatch):
    from sklearn.pipeline import Pipeline
    from threadpoolctl import threadpool_info

    from src.models import experiments

    _, config, preprocessing, _ = environment
    experiment = copy.deepcopy(config["training"]["experiments"][1])
    experiment["features"]["outliers"]["enabled"] = True
    experiment["features"]["outliers"]["min_group_size"] = 2
    experiment["features"]["selection"] = {"enabled": True, "k": 4}
    X, y, metadata, _ = prepared(environment)
    original = Pipeline.fit
    fitted = []

    def check_fit(self, rows, labels):
        assert all(p["num_threads"] <= 1 for p in threadpool_info())
        fitted.append(self)
        return original(self, rows, labels)

    monkeypatch.setattr(Pipeline, "fit", check_fit)
    for a, b in expanding_folds(X, 2):
        model, order, _, pools = experiments._fit(
            experiment, preprocessing, 42, 1, X.iloc[a], y.iloc[a]
        )
        assert all(p["num_threads"] <= 1 for p in pools)
        pipeline = model
        engineered = pipeline[:3].transform(X.iloc[a])
        columns = pipeline["columns"].transformers_[0][2]
        numeric = pipeline["columns"].named_transformers_["numeric"]
        np.testing.assert_allclose(
            numeric["imputer"].statistics_,
            engineered[columns].astype(float).median().fillna(0),
        )
        imputed = numeric["imputer"].transform(engineered[columns])
        np.testing.assert_allclose(numeric["scaler"].mean_, imputed.mean(axis=0))
        from sklearn.feature_selection import f_classif

        scores, _ = f_classif(pipeline["columns"].transform(engineered), y.iloc[a])
        np.testing.assert_allclose(
            pipeline["selection"].scores_, scores, equal_nan=True
        )
        scaler_state = numeric["scaler"].mean_.copy()
        selection_state = pipeline["selection"].scores_.copy()
        expected = RouteIQR(columns=pipeline["outliers"].columns, min_group_size=2).fit(
            pipeline["features"].transform(X.iloc[a])
        )
        assert pipeline["outliers"].global_ == expected.global_
        assert pipeline["outliers"].routes_ == expected.routes_
        state = copy.deepcopy(pipeline["outliers"].global_)
        future = X.iloc[b].copy()
        future["DISTANCE"] = 1e9
        experiments._evaluate(model, order, future, y.iloc[b], metadata.iloc[b], 1)
        assert pipeline["outliers"].global_ == state
        np.testing.assert_array_equal(numeric["scaler"].mean_, scaler_state)
        np.testing.assert_allclose(
            pipeline["selection"].scores_, selection_state, equal_nan=True
        )
        np.testing.assert_allclose(
            numeric["imputer"].statistics_,
            engineered[columns].astype(float).median().fillna(0),
        )
    assert fitted[0] is not fitted[1]
    assert fitted[0]["selection"] is not fitted[1]["selection"]


def test_streaming_evaluation_equals_integral():
    from src.models.evaluation import EvaluationAccumulator

    y = np.array(["a", "b", "c", "a", "c"])
    pred = np.array(["b", "b", "c", "a", "a"])
    proba = np.full((5, 3), 0.1)
    for i, c in enumerate(pred):
        proba[i, ["a", "b", "c"].index(c)] = 0.8
    meta = pd.DataFrame(
        {
            "ORIGIN": ["X", "Y", "X", "Z", "X"],
            "OP_UNIQUE_CARRIER": ["A", "A", "B", "B", "A"],
        }
    )
    accumulator = EvaluationAccumulator(["a", "b", "c"])
    for a, b in [(0, 2), (2, 3), (3, 5)]:
        accumulator.update(y[a:b], pred[a:b], proba[a:b], meta.iloc[a:b])
    expected, result = (
        evaluate(y, pred, proba, ["a", "b", "c"], meta),
        accumulator.result(),
    )
    assert result["confusion_matrix"] == expected["confusion_matrix"]
    assert result["macro_f1"] == expected["macro_f1"]
    assert result["log_loss"] == pytest.approx(expected["log_loss"])
    assert result["subgroups"]["ORIGIN"]["X"]["support"] == 3


def test_day_folds_match_sklearn_with_uneven_flight_counts():
    from sklearn.model_selection import TimeSeriesSplit

    days = pd.date_range("2023-01-01", periods=11)
    X = pd.DataFrame({"FL_DATE": np.repeat(days, np.arange(1, 12))})
    for (a, b), (train, validation) in zip(
        expanding_folds(X, 3), TimeSeriesSplit(n_splits=3).split(days)
    ):
        assert set(X.iloc[a].FL_DATE) == set(days[train])
        assert set(X.iloc[b].FL_DATE) == set(days[validation])
        assert X.iloc[a].FL_DATE.max() < X.iloc[b].FL_DATE.min()


def test_metrics_preserve_probability_order_and_missing_batch_classes():
    from sklearn.metrics import log_loss

    from src.models.evaluation import EvaluationAccumulator

    classes = ["c", "a", "b"]
    y = np.array(["a", "c", "a", "b"])
    probabilities = np.array(
        [[0.1, 0.8, 0.1], [0.6, 0.2, 0.2], [0.2, 0.3, 0.5], [0.1, 0.1, 0.8]]
    )
    pred = np.asarray(classes)[probabilities.argmax(axis=1)]
    accumulator = EvaluationAccumulator(classes)
    for index in range(len(y)):
        accumulator.update(
            y[index : index + 1],
            pred[index : index + 1],
            probabilities[index : index + 1],
        )
    report = accumulator.result()
    assert report["log_loss"] == pytest.approx(
        log_loss(y, probabilities[:, [1, 2, 0]], labels=["a", "b", "c"])
    )
    assert (
        report["confusion_matrix"]
        == evaluate(y, pred, probabilities, classes)["confusion_matrix"]
    )


def test_effective_config_hash_once_and_status(environment, tmp_path, monkeypatch):
    from sklearn.pipeline import Pipeline

    from src.models import artifacts

    path, config, _, _ = environment
    config["training"]["experiments"] = config["training"]["experiments"][1:2]
    config["training"]["experiments"][0]["params"].update(random_state=999, n_jobs=8)
    config["training"]["evaluation_batch_size"] = 7
    path.write_text(yaml.safe_dump(config))
    hashes = []
    identity = artifacts.file_identity
    original_fit = Pipeline.fit

    def fingerprint(p):
        hashes.append(str(p))
        return identity(p)

    def fit(self, X, y):
        assert len(hashes) == 3  # train, validation, split manifest before any fit
        return original_fit(self, X, y)

    monkeypatch.setattr(artifacts, "file_identity", fingerprint)
    monkeypatch.setattr(Pipeline, "fit", fit)
    output = tmp_path / "resolved"
    main(["--config", str(path), "--output-dir", str(output)])
    assert len(hashes) == len(set(hashes)) == 3
    effective = json.loads((output / "logistic/config.json").read_text())
    params = effective["training"]["experiments"][0]["params"]
    assert params["random_state"] == 42 and params["n_jobs"] == 1
    assert params["solver"] == "lbfgs"
    assert effective["resolved_model"]["imputation"]["strategy"] == "median"
    assert json.loads((output / "status.json").read_text())["state"] == "completed"
    manifest = json.loads((output / "logistic/manifest.json").read_text())
    assert len(manifest["source_fingerprint"]["sha256"]) == 64
    assert manifest["native_thread_limit"] == 1
    assert manifest["counts"]["validation"]["used"] == 72


def test_source_fingerprint_includes_untracked_changes(tmp_path):
    from src.models.artifacts import source_fingerprint

    (tmp_path / "src").mkdir()
    source = tmp_path / "src/new.py"
    source.write_text("answer = 1\n")
    before = source_fingerprint(tmp_path)
    assert before == source_fingerprint(tmp_path)
    source.write_text("answer = 2\n")
    assert before["sha256"] != source_fingerprint(tmp_path)["sha256"]
    source.unlink()
    assert before["sha256"] != source_fingerprint(tmp_path)["sha256"]


def test_failed_run_and_atomic_export(environment, tmp_path, monkeypatch):
    from sklearn.pipeline import Pipeline

    from src.models import artifacts

    path, _, _, _ = environment
    output = tmp_path / "failed"

    def fail(self, X, y):
        assert json.loads((output / "status.json").read_text())["state"] == "running"
        assert (
            json.loads((output / "dummy_prior/status.json").read_text())["state"]
            == "running"
        )
        raise RuntimeError("intentional training failure")

    monkeypatch.setattr(Pipeline, "fit", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        main(["--config", str(path), "--output-dir", str(output)])
    assert json.loads((output / "status.json").read_text())["state"] == "failed"
    assert (
        json.loads((output / "dummy_prior/status.json").read_text())["state"]
        == "failed"
    )
    assert not (output / "summary.json").exists()
    target = tmp_path / "atomic.json"
    artifacts.write_json(target, {"old": True})
    with pytest.raises(ValueError):
        artifacts.write_json(target, {"invalid": np.nan})
    assert json.loads(target.read_text()) == {"old": True}
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"previous-complete-model")

    def broken_writer(model, temporary):
        from pathlib import Path

        Path(temporary).write_bytes(b"partial")
        raise RuntimeError("serialization interrupted")

    monkeypatch.setitem(artifacts.SERIALIZERS, "broken", (broken_writer, None))
    with pytest.raises(RuntimeError, match="interrupted"):
        artifacts.save_model(None, model_path, "broken")
    assert model_path.read_bytes() == b"previous-complete-model"
    assert not list(tmp_path.glob(".*.tmp"))


def test_lightgbm_seeded_sampling_and_roundtrip(environment, tmp_path):
    _, config, preprocessing, _ = environment
    experiment = copy.deepcopy(config["training"]["experiments"][3])
    experiment["params"].update(subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
    X, y, _, _ = prepared(environment)
    first, order = create_model(experiment, preprocessing, 91, 1, CLASSES)
    second, _ = create_model(experiment, preprocessing, 91, 1, CLASSES)
    first.fit(X, y)
    second.fit(X, y)
    np.testing.assert_allclose(first.predict_proba(X), second.predict_proba(X))
    artifact = tmp_path / "lgbm.joblib"
    save_model(first, artifact)
    np.testing.assert_allclose(
        first.predict_proba(X), load_model(artifact).predict_proba(X)
    )
    assert order == tuple(sorted(CLASSES))


def test_cv_selection_invariant_to_fixed_validation(environment, tmp_path):
    path, config, _, manifest = environment
    config["training"]["experiments"] = config["training"]["experiments"][:2]
    path.write_text(yaml.safe_dump(config))
    first = main(["--config", str(path), "--output-dir", str(tmp_path / "first")])
    validation_path = manifest["splits"]["validation"]["path"]
    raw = pd.read_parquet(validation_path)
    raw["ARR_DELAY"] = 1e6
    raw["DISTANCE"] = 1e6
    raw.to_parquet(validation_path, index=False)
    second = main(["--config", str(path), "--output-dir", str(tmp_path / "second")])
    assert first["selected"] == second["selected"]
    assert [(r["cv_macro_f1"], r["cv_log_loss"]) for r in first["results"]] == [
        (r["cv_macro_f1"], r["cv_log_loss"]) for r in second["results"]
    ]
