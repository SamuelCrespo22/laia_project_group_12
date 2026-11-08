
"""Integration tests for temporal Parquet dataset generation."""

import json

import pandas as pd
import pytest
import yaml

from scripts.create_temporal_splits import build_splits


@pytest.fixture
def synthetic_environment(tmp_path):
    """Create isolated raw data, output directory and YAML config."""

    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    raw_dir.mkdir()
    processed_dir.mkdir()

    flights = pd.DataFrame({
        "FL_DATE": pd.to_datetime([
            "2022-01-01",
            "2023-06-30",
            "2023-07-01",
            "2023-09-30",
            "2023-10-01",
            "2023-12-31",
        ]),
        "ARR_DELAY": [
            0.0, 20.0, 70.0, 10.0, 45.0, 100.0
        ],
        "CANCELLED": [
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0
        ],
        "DIVERTED": [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
        "OP_UNIQUE_CARRIER": [
            "AA", "AA", "DL", "DL", "UA", "UA"
        ],
        "ORIGIN": [
            "JFK", "JFK", "ATL", "ATL", "ORD", "ORD"
        ],
    })

    # Two files with identical schemas.
    flights.iloc[:3].to_parquet(
        raw_dir / "flights_part1.parquet",
        index=False,
    )

    flights.iloc[3:].to_parquet(
        raw_dir / "flights_part2.parquet",
        index=False,
    )

    config = {
        "data": {
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
        },
        "temporal_split": {
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
            "output_dir": str(processed_dir),
            "report_class_distribution": True,
            "batch_size": 2,
            "compression": "zstd",
        },
    }

    config_path = tmp_path / "config.yaml"

    config_path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    return config_path, processed_dir


def test_create_temporal_splits(synthetic_environment):
    """Verify persisted splits and eligible-flight class counts."""

    config_path, processed_dir = synthetic_environment

    report = build_splits(config_path)

    manifest_path = processed_dir / "temporal_split_manifest.json"

    assert manifest_path.exists()

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    train = pd.read_parquet(manifest["splits"]["train"]["path"])
    validation = pd.read_parquet(
        manifest["splits"]["validation"]["path"]
    )
    test = pd.read_parquet(manifest["splits"]["test"]["path"])

    assert len(train) == 2
    assert len(validation) == 2
    assert len(test) == 2

    assert len(train) + len(validation) + len(test) == 6

    # Original columns and cancelled flights are preserved.
    assert "OP_UNIQUE_CARRIER" in train.columns
    assert "ORIGIN" in validation.columns
    assert test["CANCELLED"].sum() == 1.0

    # Class distributions exclude cancelled flights.
    assert report["train"]["eligible_rows"] == 2
    assert report["validation"]["eligible_rows"] == 2
    assert report["test"]["eligible_rows"] == 1

    assert report["train"]["classes"]["on_time"]["count"] == 1
    assert report["train"]["classes"]["moderate_delay"]["count"] == 1

    assert report["validation"]["classes"]["on_time"]["count"] == 1
    assert report["validation"]["classes"]["severe_delay"]["count"] == 1

    assert report["test"]["classes"]["severe_delay"]["count"] == 1

    report_path = pd.io.common.stringify_path(
        manifest["class_distribution"]
    )
    from pathlib import Path
    report_path = Path(report_path)

    assert report_path.exists()

    saved_report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    assert saved_report == report


def test_failed_run_preserves_previous_manifest(synthetic_environment):
    """A failed run must not replace the previously published version."""

    config_path, processed_dir = synthetic_environment

    # First execution succeeds.
    build_splits(config_path)

    manifest_path = processed_dir / "temporal_split_manifest.json"

    original_manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    original_run_id = original_manifest["run_id"]

    # Corrupt one input Parquet file before the second execution.
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    raw_dir = config_path.parent / "raw"
    corrupted_file = raw_dir / "flights_part2.parquet"

    corrupted_file.write_bytes(b"invalid parquet data")

    # The second execution must fail.
    with pytest.raises(Exception):
        build_splits(config_path)

    # The active manifest must remain unchanged.
    current_manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert current_manifest["run_id"] == original_run_id
    assert current_manifest == original_manifest

    # Previously published datasets must still be readable.
    for name in ("train", "validation", "test"):
        path = original_manifest["splits"][name]["path"]

        dataset = pd.read_parquet(path)

        assert len(dataset) == 2
