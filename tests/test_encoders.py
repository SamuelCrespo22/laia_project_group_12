"""Feature encoder tests (Issue #4, roadmap 1.4). Synthetic fixtures only."""

import io

import joblib
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.contract import FlightRecord, records_to_frame
from src.data.preprocessor import DEFAULT_CONFIG_PATH, build_features
from src.features.encoders import (
    FeatureConfig,
    build_feature_encoder,
    default_feature_config,
    load_feature_config,
)

MODEL_TYPES = ["linear", "tree"]
CONFIG = FeatureConfig(
    categorical=(
        "OP_UNIQUE_CARRIER",
        "ORIGIN",
        "DEST",
        "DEP_TIME_BLK",
        "ARR_TIME_BLK",
    ),
    numeric=(
        "CRS_ELAPSED_TIME",
        "DISTANCE",
        "MONTH",
        "DAY_OF_WEEK",
        "DISTANCE_GROUP",
    ),
    min_frequency=0.05,
)


def record(carrier="AA", origin="JFK", dest="LAX", **overrides):
    values = {
        "FL_DATE": "2023-05-15",
        "OP_UNIQUE_CARRIER": carrier,
        "ORIGIN": origin,
        "DEST": dest,
        "CRS_DEP_TIME": 900,
        "CRS_ARR_TIME": 1215,
        "CRS_ELAPSED_TIME": 375,
        "DISTANCE": 2475,
    }
    values.update(overrides)
    return {key: value for key, value in values.items() if value is not None}


def features(records):
    return build_features(records_to_frame([FlightRecord(**r) for r in records]))


def dense(matrix):
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


@pytest.fixture
def train_frame():
    """Two common carrier/origin pairs and one category below the threshold."""
    return features(
        [record("AA", "JFK", "LAX", CRS_ELAPSED_TIME=100, DISTANCE=500)] * 40
        + [record("DL", "ATL", "ORD", CRS_ELAPSED_TIME=300, DISTANCE=1500)] * 40
        + [record("ZZ", "RRR", "LAX", CRS_ELAPSED_TIME=200, DISTANCE=1000)]
    )


def fitted(model_type, frame):
    return build_feature_encoder(model_type, CONFIG).fit(frame)


def categories_of(encoder, column):
    transformer = encoder.named_transformers_["categorical"]
    return set(transformer.categories_[list(CONFIG.categorical).index(column)])


def infrequent_of(encoder, column):
    transformer = encoder.named_transformers_["categorical"]
    infrequent = transformer.infrequent_categories_
    values = infrequent[list(CONFIG.categorical).index(column)]
    return set() if values is None else set(values)


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_unseen_categories_do_not_fail(model_type, train_frame):
    encoder = fitted(model_type, train_frame)
    known = features([record()])

    output = dense(encoder.transform(features([record("Q9", "NEW", "XYZ")])))

    assert output.shape == dense(encoder.transform(known)).shape
    assert np.isfinite(output).all()


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_missing_optional_fields_do_not_fail(model_type, train_frame):
    encoder = fitted(model_type, train_frame)
    frame = features([record(CRS_ARR_TIME=None, CRS_ELAPSED_TIME=None, DISTANCE=None)])
    assert frame[["ARR_TIME_BLK", "DISTANCE", "DISTANCE_GROUP"]].isna().all(axis=None)

    assert np.isfinite(dense(encoder.transform(frame))).all()


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_missing_categorical_values_do_not_fail(model_type, train_frame):
    encoder = fitted(model_type, train_frame)
    frame = features([record()])
    frame[["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]] = pd.NA

    assert np.isfinite(dense(encoder.transform(frame))).all()


def test_one_hot_groups_rare_and_unseen_categories(train_frame):
    encoder = fitted("linear", train_frame)
    names = list(encoder.get_feature_names_out())

    assert infrequent_of(encoder, "ORIGIN") == {"RRR"}
    assert "categorical__ORIGIN_RRR" not in names
    position = names.index("categorical__ORIGIN_infrequent_sklearn")

    rare = dense(encoder.transform(features([record(origin="RRR")])))
    unseen = dense(encoder.transform(features([record(origin="NEW")])))
    common = dense(encoder.transform(features([record(origin="JFK")])))

    assert rare[0, position] == 1
    assert unseen[0, position] == 1
    assert common[0, position] == 0


def test_ordinal_uses_a_distinct_code_for_unseen_categories(train_frame):
    encoder = fitted("tree", train_frame)
    column = list(CONFIG.categorical).index("ORIGIN")

    rare = dense(encoder.transform(features([record(origin="RRR")])))[0, column]
    unseen = dense(encoder.transform(features([record(origin="NEW")])))[0, column]
    common = dense(encoder.transform(features([record(origin="JFK")])))[0, column]

    assert infrequent_of(encoder, "ORIGIN") == {"RRR"}
    assert unseen == -1
    assert rare >= 0
    assert rare != common


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_encoder_learns_only_from_the_data_it_is_fitted_on(model_type, train_frame):
    later_frame = features([record("B6", "SEA", "BOS", DISTANCE=9000)] * 500)
    encoder = fitted(model_type, train_frame)
    before = dense(encoder.transform(train_frame))

    encoder.transform(later_frame)

    assert "SEA" not in categories_of(encoder, "ORIGIN")
    assert "B6" not in categories_of(encoder, "OP_UNIQUE_CARRIER")
    assert np.array_equal(before, dense(encoder.transform(train_frame)))


def test_missing_numbers_use_the_training_median(train_frame):
    encoder = fitted("tree", train_frame)
    frame = features([record(DISTANCE=None)])
    column = len(CONFIG.categorical) + list(CONFIG.numeric).index("DISTANCE")

    assert (
        dense(encoder.transform(frame))[0, column] == train_frame["DISTANCE"].median()
    )


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_fitted_encoder_survives_joblib(model_type, train_frame):
    encoder = fitted(model_type, train_frame)
    frame = features([record(), record("Q9", "NEW", "XYZ")])
    buffer = io.BytesIO()

    joblib.dump(encoder, buffer)
    buffer.seek(0)

    restored = joblib.load(buffer)
    assert np.array_equal(
        dense(restored.transform(frame)), dense(encoder.transform(frame))
    )


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_batch_output_keeps_row_order(model_type, train_frame):
    encoder = fitted(model_type, train_frame)
    records = [record("DL", "ATL", "ORD"), record("Q9", "NEW", "XYZ"), record()]

    batch = dense(encoder.transform(features(records)))
    row_by_row = np.vstack([dense(encoder.transform(features([r]))) for r in records])

    assert np.array_equal(batch, row_by_row)


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_only_configured_columns_reach_the_output(model_type, train_frame):
    names = fitted(model_type, train_frame).get_feature_names_out()

    for excluded in ("FL_DATE", "CRS_DEP_TIME", "CRS_ARR_TIME"):
        assert not any(excluded in name for name in names)


def test_tree_output_has_categorical_columns_first(train_frame):
    names = list(fitted("tree", train_frame).get_feature_names_out())
    expected = [f"categorical__{column}" for column in CONFIG.categorical]

    assert names[: len(CONFIG.categorical)] == expected


@pytest.mark.parametrize(
    "model_type, classifier, fit_params",
    [
        ("linear", LogisticRegression(max_iter=200), {}),
        (
            "tree",
            LGBMClassifier(n_estimators=3, min_child_samples=1, verbose=-1),
            {"classifier__categorical_feature": list(range(len(CONFIG.categorical)))},
        ),
    ],
)
def test_encoder_feeds_a_classifier_with_unseen_categories(
    model_type, classifier, fit_params, train_frame
):
    labels = ["on_time"] * 40 + ["severe_delay"] * 40 + ["moderate_delay"]
    pipeline = Pipeline(
        [
            ("encoder", build_feature_encoder(model_type, CONFIG)),
            ("classifier", classifier),
        ]
    ).fit(train_frame, labels, **fit_params)

    predictions = pipeline.predict(features([record("Q9", "NEW", "XYZ"), record()]))

    assert len(predictions) == 2
    assert set(predictions) <= set(labels)


def test_unknown_model_type_is_rejected():
    with pytest.raises(ValueError, match="model_type"):
        build_feature_encoder("forest", CONFIG)


def test_project_config_is_valid():
    config = default_feature_config()

    assert "FL_DATE" not in config.categorical + config.numeric
    assert 0 < config.min_frequency < 1


def write_config(tmp_path, **changes):
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["features"].update(changes)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"numeric": ["ARR_DELAY"]}, "not in the allowlist"),
        ({"numeric": ["ORIGIN"]}, "more than once"),
        ({"categorical": [], "numeric": []}, "No feature columns"),
        ({"min_frequency": 0}, "min_frequency"),
        ({"min_frequency": 100}, "min_frequency"),
    ],
)
def test_invalid_feature_config_is_rejected(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        load_feature_config(write_config(tmp_path, **changes))
