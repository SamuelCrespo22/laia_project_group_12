import pandas as pd
import numpy as np
from typing import Tuple, List

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

