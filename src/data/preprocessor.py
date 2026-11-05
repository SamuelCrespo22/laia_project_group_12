import pandas as pd
import numpy as np
from typing import Tuple, List

import re
from typing import Sequence

# Allowed features known strictly prior to scheduled departure
SCHEDULED_FEATURES = [
    "FL_DATE",
    "OP_UNIQUE_CARRIER",  # or OP_CARRIER depending on BTS release
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
]

TARGET_COLUMN = "delay_severity"


def compute_delay_severity(arr_delay: float) -> str:
    """
    Categorizes arrival delay into three mutually exclusive categories:
    - on_time: ARR_DELAY < 15
    - moderate_delay: 15 <= ARR_DELAY < 60
    - severe_delay: ARR_DELAY >= 60
    """
    if pd.isna(arr_delay):
        return np.nan
    if arr_delay < 15:
        return "on_time"
    elif arr_delay < 60:
        return "moderate_delay"
    else:
        return "severe_delay"


def filter_valid_flights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out cancelled and diverted flights, ensuring valid arrival delay.
    Only flights that were neither cancelled nor diverted and have a valid recorded
    arrival delay will be considered for model development and evaluation.
    """
    mask = (
        (df.get("CANCELLED", 0) == 0) &
        (df.get("DIVERTED", 0) == 0) &
        (df["ARR_DELAY"].notna())
    )
    return df[mask].copy()


def preprocess_flight_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Applies filtering and target creation if is_training=True.
    Extracts time-based attributes and handles missing values.
    """
    clean_df = df.copy()

    if is_training:
        clean_df = filter_valid_flights(clean_df)
        clean_df[TARGET_COLUMN] = clean_df["ARR_DELAY"].apply(compute_delay_severity)

    return clean_df


# Columns that are known only after departure, after arrival,
# or as a consequence of disruption/cancellation/diversion.
FORBIDDEN_LEAKAGE_PATTERNS = [
    r"^DEP_TIME$",
    r"^DEP_DELAY.*",
    r"^DEP_DEL15$",
    r"^TAXI_OUT$",
    r"^WHEELS_OFF$",
    r"^WHEELS_ON$",
    r"^TAXI_IN$",
    r"^ARR_TIME$",
    r"^ARR_DELAY.*",
    r"^ARR_DEL15$",
    r"^ACTUAL_ELAPSED_TIME$",
    r"^AIR_TIME$",
    r"^CARRIER_DELAY$",
    r"^WEATHER_DELAY$",
    r"^NAS_DELAY$",
    r"^SECURITY_DELAY$",
    r"^LATE_AIRCRAFT_DELAY$",
    r"^CANCELLATION_CODE$",
    r"^DIV_.*",
    r"^FIRST_DEP_TIME$",
    r"^TOTAL_ADD_GTIME$",
    r"^LONGEST_ADD_GTIME$",
]


def find_forbidden_columns(
    columns: Sequence[str],
    forbidden_patterns: Sequence[str] = FORBIDDEN_LEAKAGE_PATTERNS,
) -> List[str]:
    """
    Returns columns matching known leakage-prone post-departure patterns.
    """
    forbidden_columns = []

    for column in columns:
        for pattern in forbidden_patterns:
            if re.fullmatch(pattern, column):
                forbidden_columns.append(column)
                break

    return forbidden_columns


def assert_no_leakage_columns(
    columns: Sequence[str],
    forbidden_patterns: Sequence[str] = FORBIDDEN_LEAKAGE_PATTERNS) -> None:
    """
    Raises an AssertionError if forbidden leakage columns are present.
    """
    forbidden_columns = find_forbidden_columns(
        columns=columns,
        forbidden_patterns=forbidden_patterns,
    )

    if forbidden_columns:
        raise AssertionError(
            f"Forbidden leakage columns found: {sorted(forbidden_columns)}"
        )


def select_scheduled_features(df: pd.DataFrame, include_target: bool = False) -> pd.DataFrame:
    """
    Keeps only features known before scheduled departure.

    If include_target=True, also keeps TARGET_COLUMN.

    This uses the existing SCHEDULED_FEATURES and TARGET_COLUMN constants already
    defined in this file.
    """
    output_columns = list(SCHEDULED_FEATURES)

    if include_target:
        output_columns.append(TARGET_COLUMN)

    missing_columns = [
        column for column in output_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required scheduled feature/target columns: {missing_columns}"
        )

    selected = df.loc[:, output_columns].copy()

    feature_columns = [
        column for column in selected.columns if column != TARGET_COLUMN
    ]

    assert_no_leakage_columns(feature_columns)

    return selected


def preprocess_flight_data_leakage_safe(df: pd.DataFrame, is_training: bool = True, reject_forbidden_input_at_inference: bool = True) -> pd.DataFrame:

    # Accepts only scheduled features;

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if is_training:
        processed = preprocess_flight_data(df, is_training=True)

        leakage_safe = select_scheduled_features(
            processed,
            include_target=True,
        )

    else:
        if reject_forbidden_input_at_inference:
            assert_no_leakage_columns(df.columns)

        processed = preprocess_flight_data(df, is_training=False)

        leakage_safe = select_scheduled_features(
            processed,
            include_target=False,
        )

    sort_columns = [
        column
        for column in ["FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "CRS_DEP_TIME"]
        if column in leakage_safe.columns
    ]

    if sort_columns:
        leakage_safe = leakage_safe.sort_values(sort_columns).reset_index(drop=True)
    else:
        leakage_safe = leakage_safe.reset_index(drop=True)

    return leakage_safe
