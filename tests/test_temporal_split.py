
"""Unit tests for deterministic temporal data splitting."""

import pandas as pd
import pytest
import json

from src.data.temporal_split import (
    load_split_config,
    load_published_splits,
    temporal_split,
)


@pytest.fixture
def split_config():
    """Create an independent configuration for synthetic tests."""
    return {
        "date_column": "FL_DATE",
        "train": {
            "start": "2022-01-01",
            "end": "2023-07-01",
        },
        "validation": {
            "start": "2023-07-01",
            "end": "2023-10-01",
        },
        "test": {
            "start": "2023-10-01",
            "end": "2024-01-01",
        },
    }


@pytest.fixture
def sample_flights():
    """Small synthetic dataset with dates at split boundaries."""
    return pd.DataFrame({
        "FL_DATE": [
            "2022-01-01",
            "2023-06-30",
            "2023-07-01",
            "2023-09-30",
            "2023-10-01",
            "2023-12-31",
        ],
        "ARR_DELAY": [0, 10, 20, 30, 60, 90],
    })


def test_temporal_split_sizes(sample_flights, split_config):
    """Each split receives the expected number of flights."""
    result = temporal_split(sample_flights, split_config)

    assert len(result["train"]) == 2
    assert len(result["validation"]) == 2
    assert len(result["test"]) == 2


def test_temporal_split_boundaries(sample_flights, split_config):
    """Boundary dates belong to the correct temporal periods."""
    result = temporal_split(sample_flights, split_config)

    assert result["train"]["FL_DATE"].tolist() == [
        "2022-01-01",
        "2023-06-30",
    ]

    assert result["validation"]["FL_DATE"].tolist() == [
        "2023-07-01",
        "2023-09-30",
    ]

    assert result["test"]["FL_DATE"].tolist() == [
        "2023-10-01",
        "2023-12-31",
    ]


def test_temporal_split_no_overlap(sample_flights, split_config):
    """A flight cannot appear in more than one split."""
    result = temporal_split(sample_flights, split_config)

    train_indices = set(result["train"].index)
    validation_indices = set(result["validation"].index)
    test_indices = set(result["test"].index)

    assert train_indices.isdisjoint(validation_indices)
    assert train_indices.isdisjoint(test_indices)
    assert validation_indices.isdisjoint(test_indices)


def test_temporal_split_deterministic(sample_flights, split_config):
    """Repeated execution produces identical results."""
    first = temporal_split(sample_flights, split_config)
    second = temporal_split(sample_flights, split_config)

    for name in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(first[name], second[name])


def test_temporal_split_missing_date_column(split_config):
    """Reject data without the required BTS date column."""
    df = pd.DataFrame({"ARR_DELAY": [0, 30, 90]})

    with pytest.raises(ValueError, match="FL_DATE"):
        temporal_split(df, split_config)


def test_temporal_split_missing_date(split_config):
    """Reject records with missing flight dates."""
    df = pd.DataFrame({
        "FL_DATE": ["2023-01-01", None],
        "ARR_DELAY": [0, 30],
    })

    with pytest.raises(ValueError, match="missing values"):
        temporal_split(df, split_config)


def test_temporal_split_outside_boundaries(split_config):
    """Reject records outside the configured date range."""
    df = pd.DataFrame({
        "FL_DATE": ["2021-12-31", "2023-01-01"],
        "ARR_DELAY": [0, 30],
    })

    with pytest.raises(ValueError, match="outside"):
        temporal_split(df, split_config)


def test_load_split_config_valid(tmp_path):
    """Read a valid configuration from YAML."""
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        """
temporal_split:
  date_column: FL_DATE
  train:
    start: "2022-01-01"
    end: "2023-07-01"
  validation:
    start: "2023-07-01"
    end: "2023-10-01"
  test:
    start: "2023-10-01"
    end: "2024-01-01"
""",
        encoding="utf-8",
    )

    result = load_split_config(config_file)

    assert result["date_column"] == "FL_DATE"
    assert result["train"]["start"] == "2022-01-01"


def test_load_split_config_rejects_overlap(tmp_path):
    """Reject overlapping temporal boundaries."""
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        """
temporal_split:
  date_column: FL_DATE
  train:
    start: "2022-01-01"
    end: "2023-08-01"
  validation:
    start: "2023-07-01"
    end: "2023-10-01"
  test:
    start: "2023-10-01"
    end: "2024-01-01"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-overlapping"):
        load_split_config(config_file)



def test_load_published_splits_valid(tmp_path):
    """Resolve all three published dataset paths."""

    processed_dir = tmp_path / "data" / "processed"
    run_dir = processed_dir / "runs" / "test_run"

    run_dir.mkdir(parents=True)

    for name in ("train", "validation", "test"):
        (run_dir / f"{name}.parquet").touch()

    manifest = {
        "splits": {
            name: {
                "path": str(
                    run_dir / f"{name}.parquet"
                ),
                "rows": 2,
            }
            for name in ("train", "validation", "test")
        }
    }

    manifest_path = processed_dir / "temporal_split_manifest.json"

    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    paths = load_published_splits(manifest_path)

    assert set(paths) == {"train", "validation", "test"}

    for name, path in paths.items():
        assert path.is_absolute()
        assert path.is_file()
        assert path.name == f"{name}.parquet"


def test_load_published_splits_missing_manifest(tmp_path):
    """Reject a missing manifest file."""

    manifest_path = tmp_path / "missing_manifest.json"

    with pytest.raises(FileNotFoundError, match="manifest"):
        load_published_splits(manifest_path)


def test_load_published_splits_missing_dataset(tmp_path):
    """Reject a manifest referencing a nonexistent dataset."""

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    manifest = {
        "splits": {
            name: {
                "path": str(
                    processed_dir / f"{name}.parquet"
                ),
                "rows": 2,
            }
            for name in ("train", "validation", "test")
        }
    }

    manifest_path = processed_dir / "temporal_split_manifest.json"

    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="dataset"):
        load_published_splits(manifest_path)
