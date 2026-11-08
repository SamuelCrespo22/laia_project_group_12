
"""Deterministic temporal train, validation and test splitting."""

import json

from datetime import date
from pathlib import Path

import pandas as pd
import yaml


SPLIT_NAMES = ("train", "validation", "test")


def load_split_config(config_path="configs/config.yaml"):
    """Load temporal split settings from the project configuration."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or "temporal_split" not in config:
        raise ValueError("Missing temporal_split configuration")

    split_config = config["temporal_split"]
    date_column = split_config.get("date_column")

    if date_column != "FL_DATE":
        raise ValueError("Temporal splitting must use BTS column FL_DATE")

    boundaries = {}

    for name in SPLIT_NAMES:
        if name not in split_config:
            raise ValueError(f"Missing split configuration: {name}")

        settings = split_config[name]

        try:
            start = date.fromisoformat(settings["start"])
            end = date.fromisoformat(settings["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid dates for split: {name}") from exc

        if start >= end:
            raise ValueError(f"Invalid interval for split: {name}")

        boundaries[name] = (pd.Timestamp(start), pd.Timestamp(end))

    for previous, current in zip(SPLIT_NAMES, SPLIT_NAMES[1:]):
        previous_end = boundaries[previous][1]
        current_start = boundaries[current][0]

        if previous_end != current_start:
            raise ValueError(
                f"Splits {previous} and {current} must be contiguous "
                "and non-overlapping"
            )

    return split_config


def temporal_split(df, split_config):
    """Assign flights to chronological, non-overlapping periods."""
    date_column = split_config["date_column"]

    if date_column not in df.columns:
        raise ValueError(f"Missing required column: {date_column}")

    dates = pd.to_datetime(df[date_column], errors="raise")

    if dates.isna().any():
        raise ValueError("FL_DATE contains missing values")

    splits = {}

    for name in SPLIT_NAMES:
        start = pd.Timestamp(split_config[name]["start"])
        end = pd.Timestamp(split_config[name]["end"])

        mask = (dates >= start) & (dates < end)
        splits[name] = df.loc[mask].copy()

    assigned_count = sum(len(part) for part in splits.values())

    if assigned_count != len(df):
        raise ValueError(
            f"{len(df) - assigned_count} flights fall outside "
            "the configured temporal boundaries"
        )

    return splits



def load_published_splits(
    manifest_path="data/processed/temporal_split_manifest.json",
):
    """Resolve and validate the published temporal dataset paths."""
    manifest_file = Path(manifest_path).resolve()

    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"Temporal split manifest not found: {manifest_file}"
        )

    with manifest_file.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    paths = {}

    for name in SPLIT_NAMES:
        try:
            stored_path = Path(manifest["splits"][name]["path"])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Missing dataset path for split: {name}"
            ) from exc

        if stored_path.is_absolute():
            resolved_path = stored_path
        else:
            # Paths generated from the repository root start with
            # data/processed/. Resolve them relative to that root.
            repository_root = manifest_file.parent.parent.parent
            resolved_path = repository_root / stored_path

        resolved_path = resolved_path.resolve()

        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"Published {name} dataset not found: {resolved_path}"
            )

        paths[name] = resolved_path

    return paths
