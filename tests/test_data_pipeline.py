import pandas as pd
import numpy as np
from src.data.preprocessor import (
    compute_delay_severity,
    filter_valid_flights,
    preprocess_flight_data,
    TARGET_COLUMN
)


def test_compute_delay_severity():
    assert compute_delay_severity(-10) == "on_time"
    assert compute_delay_severity(0) == "on_time"
    assert compute_delay_severity(14) == "on_time"
    assert compute_delay_severity(15) == "moderate_delay"
    assert compute_delay_severity(59) == "moderate_delay"
    assert compute_delay_severity(60) == "severe_delay"
    assert compute_delay_severity(120) == "severe_delay"
    assert pd.isna(compute_delay_severity(np.nan))


def test_filter_valid_flights():
    raw_data = {
        "FL_DATE": ["2023-01-01", "2023-01-01", "2023-01-01", "2023-01-01"],
        "CANCELLED": [0, 1, 0, 0],
        "DIVERTED": [0, 0, 1, 0],
        "ARR_DELAY": [10.0, np.nan, np.nan, np.nan]
    }
    df = pd.DataFrame(raw_data)
    filtered = filter_valid_flights(df)
    assert len(filtered) == 1
    assert filtered.iloc[0]["ARR_DELAY"] == 10.0


def test_preprocess_flight_data_training():
    raw_data = {
        "FL_DATE": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "CANCELLED": [0, 0, 0],
        "DIVERTED": [0, 0, 0],
        "ARR_DELAY": [5.0, 30.0, 95.0]
    }
    df = pd.DataFrame(raw_data)
    processed = preprocess_flight_data(df, is_training=True)
    assert TARGET_COLUMN in processed.columns
    assert list(processed[TARGET_COLUMN]) == ["on_time", "moderate_delay", "severe_delay"]

