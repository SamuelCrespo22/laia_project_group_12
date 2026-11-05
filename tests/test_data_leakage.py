import pytest
import pandas as pd
import numpy as np

from src.data.preprocessor import (
    find_forbidden_columns,
    assert_no_leakage_columns,
    select_scheduled_features,
    preprocess_flight_data_leakage_safe,
    SCHEDULED_FEATURES,
    TARGET_COLUMN
)


def test_keeps_only_allowed_features():
    df = pd.DataFrame(
        {
            "FL_DATE": ["2023-01-01"],
            "OP_UNIQUE_CARRIER": ["AA"],
            "ORIGIN": ["JFK"],
            "DEST": ["LAX"],
            "CRS_DEP_TIME": [900],
            "CRS_ARR_TIME": [1215],
            "CRS_ELAPSED_TIME": [375],
            "DISTANCE": [2475],

            # Required for training filtering/target creation.
            "CANCELLED": [0],
            "DIVERTED": [0],
            "ARR_DELAY": [35.0],

            # Leakage columns that must not reach the model.
            "DEP_TIME": [905],
            "DEP_DELAY": [5],
            "DEP_DEL15": [0],
            "TAXI_OUT": [18],
            "WHEELS_OFF": [923],
            "WHEELS_ON": [1200],
            "TAXI_IN": [7],
            "ARR_TIME": [1207],
            "ARR_DELAY_NEW": [35],
            "ARR_DEL15": [1],
            "ACTUAL_ELAPSED_TIME": [367],
            "AIR_TIME": [340],
            "CARRIER_DELAY": [10],
            "WEATHER_DELAY": [0],
            "NAS_DELAY": [25],
            "SECURITY_DELAY": [0],
            "LATE_AIRCRAFT_DELAY": [0],
            "CANCELLATION_CODE": [np.nan],
            "DIV_AIRPORT_LANDINGS": [0],
            "FIRST_DEP_TIME": [np.nan],
            "TOTAL_ADD_GTIME": [np.nan],
            "LONGEST_ADD_GTIME": [np.nan],
        }
    )

    processed = preprocess_flight_data_leakage_safe(
        df,
        is_training=True,
    )

    expected_columns = SCHEDULED_FEATURES + [TARGET_COLUMN]

    assert list(processed.columns) == expected_columns
    assert processed.iloc[0][TARGET_COLUMN] == "moderate_delay"


def test_removes_forbidden_columns():
    df = pd.DataFrame(
        {
            "FL_DATE": ["2023-01-01"],
            "OP_UNIQUE_CARRIER": ["AA"],
            "ORIGIN": ["JFK"],
            "DEST": ["LAX"],
            "CRS_DEP_TIME": [900],
            "CRS_ARR_TIME": [1215],
            "CRS_ELAPSED_TIME": [375],
            "DISTANCE": [2475],
            "CANCELLED": [0],
            "DIVERTED": [0],
            "ARR_DELAY": [70.0],
            "DEP_DELAY": [15],
            "TAXI_OUT": [20],
            "ARR_TIME": [1230],
            "CARRIER_DELAY": [15],
        }
    )

    processed = preprocess_flight_data_leakage_safe(
        df,
        is_training=True,
    )

    forbidden_columns = find_forbidden_columns(processed.columns)

    assert forbidden_columns == []
    assert "ARR_DELAY" not in processed.columns
    assert "DEP_DELAY" not in processed.columns
    assert "TAXI_OUT" not in processed.columns
    assert "ARR_TIME" not in processed.columns
    assert "CARRIER_DELAY" not in processed.columns


def test_cancelled_diverted_and_missing_arr_delay():
    df = pd.DataFrame(
        {
            "FL_DATE": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
            ],
            "OP_UNIQUE_CARRIER": ["AA", "DL", "UA", "WN"],
            "ORIGIN": ["JFK", "ATL", "ORD", "LAX"],
            "DEST": ["LAX", "JFK", "SFO", "LAS"],
            "CRS_DEP_TIME": [900, 1000, 1100, 1200],
            "CRS_ARR_TIME": [1215, 1315, 1415, 1515],
            "CRS_ELAPSED_TIME": [375, 195, 255, 75],
            "DISTANCE": [2475, 760, 1846, 236],
            "CANCELLED": [0, 1, 0, 0],
            "DIVERTED": [0, 0, 1, 0],
            "ARR_DELAY": [10.0, np.nan, 50.0, np.nan],
        }
    )

    processed = preprocess_flight_data_leakage_safe(
        df,
        is_training=True,
    )

    assert len(processed) == 1
    assert processed.iloc[0]["OP_UNIQUE_CARRIER"] == "AA"
    assert processed.iloc[0][TARGET_COLUMN] == "on_time"


def test_inference_rejects_forbidden_columns():
    df = pd.DataFrame(
        {
            "FL_DATE": ["2025-05-15"],
            "OP_UNIQUE_CARRIER": ["AA"],
            "ORIGIN": ["JFK"],
            "DEST": ["LAX"],
            "CRS_DEP_TIME": [900],
            "CRS_ARR_TIME": [1215],
            "CRS_ELAPSED_TIME": [375],
            "DISTANCE": [2475],

            # This should not be accepted at inference time.
            "ARR_DELAY": [30],
        }
    )

    with pytest.raises(AssertionError):
        preprocess_flight_data_leakage_safe(
            df,
            is_training=False,
            reject_forbidden_input_at_inference=True,
        )


def test_inference_keeps_only_scheduled_features():
    df = pd.DataFrame(
        {
            "FL_DATE": ["2025-05-15"],
            "OP_UNIQUE_CARRIER": ["AA"],
            "ORIGIN": ["JFK"],
            "DEST": ["LAX"],
            "CRS_DEP_TIME": [900],
            "CRS_ARR_TIME": [1215],
            "CRS_ELAPSED_TIME": [375],
            "DISTANCE": [2475],
        }
    )

    processed = preprocess_flight_data_leakage_safe(
        df,
        is_training=False,
    )

    assert list(processed.columns) == SCHEDULED_FEATURES