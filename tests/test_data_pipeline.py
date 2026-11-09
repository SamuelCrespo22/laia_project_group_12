"""Preprocessing pipeline tests (Issue #3, roadmap 1.3). Synthetic fixtures only."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

import src.api.app as api_app
from src.data import make_dataset
from src.data.contract import (
    FlightRecord,
    find_post_departure_columns,
    records_to_frame,
)
from src.data.preprocessor import (
    DEFAULT_CONFIG_PATH,
    TARGET_COLUMN,
    build_features,
    build_training_frame,
    compute_delay_severity,
    default_config,
    filter_valid_flights,
    label_delay_severity,
    load_config,
    validate_allowlist,
)

# Post-departure columns listed in Issue #3; DEP_DELAY*, ARR_DELAY* and DIV_*
# are expanded to real TranStats field names.
ISSUE_POST_DEPARTURE = [
    "DEP_TIME",
    "DEP_DELAY",
    "DEP_DELAY_NEW",
    "DEP_DELAY_GROUP",
    "DEP_DEL15",
    "TAXI_OUT",
    "WHEELS_OFF",
    "WHEELS_ON",
    "TAXI_IN",
    "ARR_TIME",
    "ARR_DELAY",
    "ARR_DELAY_NEW",
    "ARR_DELAY_GROUP",
    "ARR_DEL15",
    "ACTUAL_ELAPSED_TIME",
    "AIR_TIME",
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "CANCELLATION_CODE",
    "DIV_AIRPORT_LANDINGS",
    "DIV_REACHED_DEST",
    "DIV_ARR_DELAY",
    "DIV1_AIRPORT",
    "DIV1_WHEELS_ON",
    "FIRST_DEP_TIME",
    "TOTAL_ADD_GTIME",
    "LONGEST_ADD_GTIME",
]
ISSUE_ALLOWLIST = [
    "FL_DATE",
    "MONTH",
    "DAY_OF_WEEK",
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "DEP_TIME_BLK",
    "CRS_ARR_TIME",
    "ARR_TIME_BLK",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
    "DISTANCE_GROUP",
]


def raw_flights(n: int = 1, **overrides) -> pd.DataFrame:
    """Raw-like flights, typed like the ingested Parquet files."""
    base = {
        "YEAR": 2023,
        "MONTH": 1,
        "DAY_OF_MONTH": 2,
        "DAY_OF_WEEK": 1,
        "FL_DATE": pd.Timestamp("2023-01-02"),
        "OP_UNIQUE_CARRIER": "AA",
        "TAIL_NUM": "N123AA",
        "ORIGIN": "JFK",
        "DEST": "LAX",
        "CRS_DEP_TIME": 900,
        "CRS_ARR_TIME": 1215,
        "CRS_ELAPSED_TIME": 375.0,
        "DISTANCE": 2475.0,
        "DISTANCE_GROUP": 10,
        "CANCELLED": 0.0,
        "DIVERTED": 0.0,
        "ARR_DELAY": 35.0,
        "DEP_DELAY": 20.0,
        "TAXI_OUT": 18.0,
    }
    frame = pd.DataFrame(
        {
            k: (
                [overrides.get(k, v)] * n
                if not isinstance(overrides.get(k), list)
                else overrides[k]
            )
            for k, v in base.items()
        }
    )
    dtypes = {
        "FL_DATE": "datetime64[ns]",
        "OP_UNIQUE_CARRIER": "string",
        "TAIL_NUM": "string",
        "ORIGIN": "string",
        "DEST": "string",
        "CRS_ELAPSED_TIME": "Float64",
        "DISTANCE": "Float64",
        "CANCELLED": "Float64",
        "DIVERTED": "Float64",
        "ARR_DELAY": "Float64",
        "DEP_DELAY": "Float64",
        "TAXI_OUT": "Float64",
    }
    return frame.astype({c: dtypes.get(c, "Int64") for c in frame.columns})


# ------------------------------------------------------------------ target
def test_compute_delay_severity():
    assert compute_delay_severity(-10) == "on_time"
    assert compute_delay_severity(0) == "on_time"
    assert compute_delay_severity(14) == "on_time"
    assert compute_delay_severity(15) == "moderate_delay"
    assert compute_delay_severity(59) == "moderate_delay"
    assert compute_delay_severity(60) == "severe_delay"
    assert compute_delay_severity(120) == "severe_delay"
    assert pd.isna(compute_delay_severity(np.nan))


def test_thresholds_come_from_config(tmp_path):
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    config["target"]["thresholds"] = {"on_time_max": 10, "moderate_delay_max": 30}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    labels = label_delay_severity(pd.Series([9.0, 10.0, 29.0, 30.0]), load_config(path))
    assert labels.tolist() == [
        "on_time",
        "moderate_delay",
        "moderate_delay",
        "severe_delay",
    ]


# --------------------------------------------------------------- filtering
def test_filter_valid_flights():
    df = pd.DataFrame(
        {
            "FL_DATE": ["2023-01-01"] * 5,
            "CANCELLED": [0, 1, 0, 0, np.nan],
            "DIVERTED": [0, 0, 1, 0, 0],
            "ARR_DELAY": [10.0, np.nan, 50.0, np.nan, 5.0],
        }
    )
    filtered = filter_valid_flights(df)
    assert len(filtered) == 1
    assert filtered.iloc[0]["ARR_DELAY"] == 10.0


def test_training_frame_drops_ineligible_flights_and_labels_the_rest():
    raw = raw_flights(
        4,
        OP_UNIQUE_CARRIER=["AA", "DL", "UA", "WN"],
        CANCELLED=[0.0, 1.0, 0.0, 0.0],
        DIVERTED=[0.0, 0.0, 1.0, 0.0],
        ARR_DELAY=[10.0, None, 50.0, None],
    )
    data = build_training_frame(raw)
    assert data["OP_UNIQUE_CARRIER"].tolist() == ["AA"]
    assert data[TARGET_COLUMN].tolist() == ["on_time"]


# ----------------------------------------------------------------- leakage
def test_contract_flags_every_post_departure_column_of_the_issue():
    assert find_post_departure_columns(ISSUE_POST_DEPARTURE) == ISSUE_POST_DEPARTURE


def test_only_allowlisted_columns_reach_the_features():
    raw = raw_flights()
    for column in ISSUE_POST_DEPARTURE:
        if column not in raw.columns:
            raw[column] = 1.0
    data = build_training_frame(raw)
    assert list(data.columns) == ISSUE_ALLOWLIST + [TARGET_COLUMN]
    assert find_post_departure_columns(list(data.columns)) == []
    assert data.loc[0, TARGET_COLUMN] == "moderate_delay"


def test_configured_allowlist_is_the_issue_allowlist():
    assert list(default_config().feature_allowlist) == ISSUE_ALLOWLIST


@pytest.mark.parametrize(
    "columns",
    [
        ["FL_DATE", "DEP_DELAY"],  # post-departure pattern
        ["FL_DATE", "TAIL_NUM"],  # post-departure in the contract
        ["FL_DATE", "CANCELLED"],  # outcome flag
        ["FL_DATE", "WEATHER"],  # not in the contract
        ["FL_DATE", "FL_DATE"],  # duplicate
        [],
    ],
)
def test_allowlist_rejects_unsafe_columns(columns):
    with pytest.raises(ValueError):
        validate_allowlist(columns)


def test_inference_drops_post_departure_columns():
    request = raw_flights().drop(columns=["CANCELLED", "DIVERTED"])
    features = build_features(request)
    assert list(features.columns) == ISSUE_ALLOWLIST


# ---------------------------------------------------------------- features
def test_derived_features():
    raw = raw_flights(
        6,
        FL_DATE=[
            pd.Timestamp(d)
            for d in [
                "2023-01-02",
                "2023-01-08",
                "2023-02-01",
                "2023-12-31",
                "2022-07-04",
                "2022-07-05",
            ]
        ],
        CRS_DEP_TIME=[0, 559, 600, 1259, 2359, 2400],
        DISTANCE=[31.0, 249.0, 250.0, 2499.0, 2500.0, 4983.0],
    )
    features = build_features(raw)
    assert features["MONTH"].tolist() == [1, 1, 2, 12, 7, 7]
    assert features["DAY_OF_WEEK"].tolist() == [1, 7, 3, 7, 1, 2]
    assert features["DEP_TIME_BLK"].tolist() == [
        "0001-0559",
        "0001-0559",
        "0600-0659",
        "1200-1259",
        "2300-2359",
        "0001-0559",
    ]
    assert features["DISTANCE_GROUP"].tolist() == [1, 1, 2, 10, 11, 11]


def test_provided_values_fill_only_missing_sources():
    request = pd.DataFrame(
        {
            "FL_DATE": [pd.Timestamp("2023-01-02")] * 2,
            "CRS_DEP_TIME": [900, None],
            "DEP_TIME_BLK": ["0000-0000", "1400-1459"],
        }
    )
    features = build_features(request)
    assert features["DEP_TIME_BLK"].tolist() == ["0900-0959", "1400-1459"]
    assert features["ORIGIN"].isna().all()


def test_features_depend_on_their_own_record_only():
    raw = raw_flights(
        3,
        OP_UNIQUE_CARRIER=["AA", "DL", "AA"],
        CRS_DEP_TIME=[900, 1400, 2400],
        DISTANCE=[2475.0, 600.0, None],
    )
    batch = build_features(raw)
    one_by_one = pd.concat([build_features(raw.iloc[[i]]) for i in range(len(raw))])
    pd.testing.assert_frame_equal(batch, one_by_one)


def test_serving_and_training_build_identical_features():
    raw = raw_flights(2, ORIGIN=["ORD", "JFK"], CRS_DEP_TIME=[1400, 900])
    requests = [
        {
            "FL_DATE": "1/2/2023 12:00:00 AM",
            "OP_UNIQUE_CARRIER": "aa",
            "ORIGIN": origin,
            "DEST": "LAX",
            "CRS_DEP_TIME": dep,
            "CRS_ARR_TIME": "1215",
            "CRS_ELAPSED_TIME": 375,
            "DISTANCE": 2475,
            "ARR_DELAY": 99,
        }
        for origin, dep in [("ORD", 1400), ("JFK", 900)]
    ]
    served = build_features(records_to_frame([FlightRecord(**r) for r in requests]))
    trained = build_features(raw)
    pd.testing.assert_frame_equal(served, trained)


def test_classify_endpoint_uses_the_training_feature_function(monkeypatch):
    calls = []

    def spy(frame, config=None):
        calls.append(frame)
        return build_features(frame, config)

    monkeypatch.setattr(api_app, "build_features", spy)
    payload = {
        "instances": [
            {
                "FL_DATE": "2023-01-02",
                "OP_UNIQUE_CARRIER": "AA",
                "ORIGIN": "JFK",
                "DEST": "LAX",
                "CRS_DEP_TIME": 900,
            }
        ]
    }
    assert TestClient(api_app.app).post("/classify", json=payload).status_code == 200
    assert len(calls) == 1


# ------------------------------------------------------------ make_dataset
def make_raw_dir(path: Path) -> None:
    path.mkdir()
    raw_flights(3, CANCELLED=[0.0, 1.0, 0.0], ARR_DELAY=[5.0, None, 70.0]).to_parquet(
        path / "flights_2023_01.parquet", index=False
    )
    raw_flights(
        2,
        FL_DATE=[pd.Timestamp("2023-02-01")] * 2,
        DIVERTED=[0.0, 1.0],
        ARR_DELAY=[20.0, None],
    ).to_parquet(path / "flights_2023_02.parquet", index=False)


def make_config(tmp_path: Path) -> Path:
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    config["data"]["raw_dir"] = str(tmp_path / "raw")
    config["preprocessing"]["output_path"] = str(
        tmp_path / "processed" / "dataset.parquet"
    )
    config["preprocessing"]["manifest_path"] = str(
        tmp_path / "processed" / "manifest.json"
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_make_dataset_builds_the_dataset_reproducibly(tmp_path):
    make_raw_dir(tmp_path / "raw")
    config_path = make_config(tmp_path)
    output = tmp_path / "processed" / "dataset.parquet"

    make_dataset.main(["--config", str(config_path)])
    first = (
        output.read_bytes(),
        (tmp_path / "processed" / "manifest.json").read_bytes(),
    )
    manifest = make_dataset.build_dataset(config_path)
    second = (
        output.read_bytes(),
        (tmp_path / "processed" / "manifest.json").read_bytes(),
    )

    assert first == second
    data = pd.read_parquet(output)
    assert list(data.columns) == ISSUE_ALLOWLIST + [TARGET_COLUMN]
    assert data[TARGET_COLUMN].tolist() == ["on_time", "severe_delay", "moderate_delay"]
    assert manifest["rows_raw"] == 5 and manifest["rows_out"] == 3
    assert manifest["rows_dropped"] == {
        "cancelled": 1,
        "diverted": 1,
        "missing_arr_delay": 0,
    }
    assert manifest["class_counts"] == {
        "on_time": 1,
        "moderate_delay": 1,
        "severe_delay": 1,
    }
    assert sorted(p.name for p in output.parent.iterdir()) == [
        "dataset.parquet",
        "manifest.json",
    ]


def test_make_dataset_requires_raw_files(tmp_path):
    (tmp_path / "raw").mkdir()
    with pytest.raises(FileNotFoundError):
        make_dataset.build_dataset(make_config(tmp_path))
