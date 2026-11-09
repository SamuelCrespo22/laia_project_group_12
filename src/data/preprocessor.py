"""Leakage-safe preprocessing shared by training and serving (Issue #3, roadmap 1.3).

* ``filter_valid_flights``: drops cancelled, diverted and ``ARR_DELAY``-less
  flights (training only).
* ``label_delay_severity``: ``ARR_DELAY`` -> ``on_time`` / ``moderate_delay`` /
  ``severe_delay`` with the thresholds in ``configs/config.yaml``.
* ``build_features``: keeps only the allowlisted pre-departure columns. It is
  used by ``src.data.make_dataset`` (training) and by ``/classify`` (serving),
  so both see the same features (no train/serve skew). Every feature is
  computed from its own record only, never from other flights.

The allowlist lives in ``configs/config.yaml``; which columns are known before
the scheduled departure is decided by the data contract
(``src/data/contract.py``). A column that is not pre-departure there cannot be
allowlisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from src.data.contract import SPECS, find_post_departure_columns, pandas_dtype

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"

TARGET_COLUMN = "delay_severity"
CLASSES = ("on_time", "moderate_delay", "severe_delay")
# Used to filter flights and build the target; never model inputs.
LABEL_SOURCE_COLUMNS = ("CANCELLED", "DIVERTED", "ARR_DELAY")
# Allowlisted columns derived from another column of the same record.
DERIVATION_SOURCES = {
    "MONTH": "FL_DATE",
    "DAY_OF_WEEK": "FL_DATE",
    "DEP_TIME_BLK": "CRS_DEP_TIME",
    "ARR_TIME_BLK": "CRS_ARR_TIME",
    "DISTANCE_GROUP": "DISTANCE",
}
DISTANCE_GROUP_MILES = 250  # BTS DistanceGroup: 250-mile intervals...
DISTANCE_GROUP_MAX = 11  # ...the last one being 2500 miles or more.


@dataclass(frozen=True)
class PreprocessingConfig:
    feature_allowlist: tuple[str, ...]
    on_time_max: float
    moderate_delay_max: float
    raw_dir: Path
    output_path: Path
    manifest_path: Path


def validate_allowlist(columns: Sequence[str]) -> tuple[str, ...]:
    """Reject allowlists with duplicates, unknown or post-departure columns."""
    columns = tuple(columns)
    if not columns:
        raise ValueError("The feature allowlist is empty")
    duplicates = sorted({c for c in columns if columns.count(c) > 1})
    if duplicates:
        raise ValueError(f"Duplicated allowlist columns: {duplicates}")
    unknown = [c for c in columns if c not in SPECS]
    if unknown:
        raise ValueError(f"Allowlist columns not in the data contract: {unknown}")
    leaking = sorted(
        set(find_post_departure_columns(columns))
        | {c for c in columns if not SPECS[c].pre_departure}
    )
    if leaking:
        raise ValueError(f"Allowlist contains post-departure columns: {leaking}")
    return columns


def load_config(path: Optional[Path | str] = None) -> PreprocessingConfig:
    """Read the preprocessing parameters; relative paths are relative to the cwd."""
    config = yaml.safe_load(
        Path(path or DEFAULT_CONFIG_PATH).read_text(encoding="utf-8")
    )
    target = config["target"]
    if target["name"] != TARGET_COLUMN or tuple(target["classes"]) != CLASSES:
        raise ValueError(
            f"configs target must be {TARGET_COLUMN} with classes {CLASSES}"
        )
    on_time_max = float(target["thresholds"]["on_time_max"])
    moderate_delay_max = float(target["thresholds"]["moderate_delay_max"])
    if not on_time_max < moderate_delay_max:
        raise ValueError("on_time_max must be lower than moderate_delay_max")
    preprocessing = config["preprocessing"]
    return PreprocessingConfig(
        feature_allowlist=validate_allowlist(preprocessing["feature_allowlist"]),
        on_time_max=on_time_max,
        moderate_delay_max=moderate_delay_max,
        raw_dir=Path(config["data"]["raw_dir"]),
        output_path=Path(preprocessing["output_path"]),
        manifest_path=Path(preprocessing["manifest_path"]),
    )


@lru_cache(maxsize=1)
def default_config() -> PreprocessingConfig:
    return load_config()


# ------------------------------------------------------------------- target
def label_delay_severity(
    arr_delay: pd.Series, config: Optional[PreprocessingConfig] = None
) -> pd.Series:
    """``ARR_DELAY`` (minutes) -> severity class; null delays stay null."""
    config = config or default_config()
    delay = pd.to_numeric(arr_delay).astype("Float64")
    labels = np.select(
        [
            delay.lt(config.on_time_max).to_numpy(dtype=bool, na_value=False),
            delay.lt(config.moderate_delay_max).to_numpy(dtype=bool, na_value=False),
            delay.notna().to_numpy(dtype=bool),
        ],
        list(CLASSES),
        default=None,
    )
    return pd.Series(labels, index=arr_delay.index, dtype="string", name=TARGET_COLUMN)


def compute_delay_severity(
    arr_delay: float, config: Optional[PreprocessingConfig] = None
):
    """Scalar version of ``label_delay_severity``; returns NaN for a null delay."""
    if pd.isna(arr_delay):
        return np.nan
    return label_delay_severity(pd.Series([arr_delay]), config).iloc[0]


# ---------------------------------------------------------------- filtering
def _equals_zero(column: pd.Series) -> np.ndarray:
    return column.eq(0).to_numpy(dtype=bool, na_value=False)


def filter_valid_flights(df: pd.DataFrame) -> pd.DataFrame:
    """Keep flights neither cancelled nor diverted and with a recorded ``ARR_DELAY``.

    A null ``CANCELLED``/``DIVERTED`` flag counts as not proven valid: dropped.
    """
    missing = [c for c in LABEL_SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns needed to filter flights: {missing}")
    mask = (
        _equals_zero(df["CANCELLED"])
        & _equals_zero(df["DIVERTED"])
        & df["ARR_DELAY"].notna().to_numpy(dtype=bool)
    )
    return df.loc[mask].copy()


# ----------------------------------------------------------------- features
def _time_block(hhmm: pd.Series) -> pd.Series:
    """BTS time blocks: ``0001-0559`` for 00:00-05:59, then one per hour.

    2400 (midnight at the end of the day) falls in the first block like 0000.
    """
    hour = (pd.to_numeric(hhmm).astype("Int64") // 100) % 24
    blocks = {h: ("0001-0559" if h < 6 else f"{h:02d}00-{h:02d}59") for h in range(24)}
    return hour.map(blocks, na_action="ignore").astype("string")


def _distance_group(distance: pd.Series) -> pd.Series:
    group = pd.to_numeric(distance).astype("Float64") // DISTANCE_GROUP_MILES + 1
    return group.clip(upper=DISTANCE_GROUP_MAX).astype("Int64")


def _derive(name: str, source: pd.Series) -> pd.Series:
    if name in ("MONTH", "DAY_OF_WEEK"):
        fl_date = pd.to_datetime(source)
        values = (
            fl_date.dt.month if name == "MONTH" else fl_date.dt.dayofweek + 1
        )  # BTS: Monday=1
        return values.astype("Int64")
    if name in ("DEP_TIME_BLK", "ARR_TIME_BLK"):
        return _time_block(source)
    return _distance_group(source)


def _typed(name: str, values: pd.Series) -> pd.Series:
    spec = SPECS[name]
    if spec.kind == "date":
        return pd.to_datetime(values).astype("datetime64[ns]")
    if spec.kind in ("int", "float"):
        values = pd.to_numeric(values)
    return values.astype(pandas_dtype(spec))


def build_features(
    frame: pd.DataFrame, config: Optional[PreprocessingConfig] = None
) -> pd.DataFrame:
    """Allowlisted features of each record, in allowlist order and input row order.

    Derivable columns (calendar fields, time blocks, distance group) are
    computed from their source column when it is present, so training data
    (where the blocks are not stored) and requests get identical values; a
    provided value is used only where the source is null. Allowlisted columns
    absent from the input are null. Every other column is dropped.
    """
    config = config or default_config()
    features = pd.DataFrame(index=frame.index)
    for name in config.feature_allowlist:
        provided = frame[name] if name in frame.columns else None
        source = DERIVATION_SOURCES.get(name)
        values = provided
        if source in frame.columns:
            values = _derive(name, frame[source])
            if provided is not None:
                values = values.where(values.notna(), _typed(name, provided))
        if values is None:
            values = pd.Series(pd.NA, index=frame.index)
        features[name] = _typed(name, values)
    assert_no_leakage_columns(features.columns)
    return features


def assert_no_leakage_columns(columns: Sequence[str]) -> None:
    """Raise if any column is only known after the scheduled departure."""
    leaking = find_post_departure_columns(list(columns))
    if leaking:
        raise AssertionError(
            f"Post-departure columns reached the features: {sorted(leaking)}"
        )


def build_training_frame(
    raw: pd.DataFrame, config: Optional[PreprocessingConfig] = None
) -> pd.DataFrame:
    """Raw flights -> eligible flights -> allowlisted features + target."""
    config = config or default_config()
    eligible = filter_valid_flights(raw)
    data = build_features(eligible, config)
    data[TARGET_COLUMN] = label_delay_severity(eligible["ARR_DELAY"], config)
    return data.reset_index(drop=True)
