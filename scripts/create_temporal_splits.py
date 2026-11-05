
"""Create reproducible temporal Parquet splits from BTS flight data."""
import os
import uuid
import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.temporal_split import SPLIT_NAMES, load_split_config


CLASS_NAMES = ("on_time", "moderate_delay", "severe_delay")


def assign_batch(batch, config):
    """Assign a record batch to non-overlapping temporal periods."""
    table = pa.Table.from_batches([batch])
    dates = table.column(config["date_column"])

    if pc.any(pc.is_null(dates)).as_py():
        raise ValueError("FL_DATE contains missing values")

    result = {}

    for name in SPLIT_NAMES:
        start = pa.scalar(
            config[name]["start"],
            type=pa.string(),
        )
        end = pa.scalar(
            config[name]["end"],
            type=pa.string(),
        )

        start = pc.cast(start, pa.timestamp("ns"))
        end = pc.cast(end, pa.timestamp("ns"))

        mask = pc.and_(
            pc.greater_equal(dates, start),
            pc.less(dates, end),
        )

        result[name] = table.filter(mask)

    if sum(part.num_rows for part in result.values()) != table.num_rows:
        raise ValueError("Flights outside configured temporal boundaries")

    return result


def count_classes(table):
    """Count delay classes among eligible, completed flights."""
    if table.num_rows == 0:
        return Counter(), 0

    cancelled = table.column("CANCELLED")
    diverted = table.column("DIVERTED")
    delays = table.column("ARR_DELAY")

    valid = pc.and_(
        pc.and_(
            pc.equal(cancelled, 0),
            pc.equal(diverted, 0),
        ),
        pc.is_valid(delays),
    )

    valid_delays = pc.filter(delays, valid)
    total = len(valid_delays)

    counts = Counter({
        "on_time": pc.sum(
            pc.less(valid_delays, 15)
        ).as_py() or 0,
        "moderate_delay": pc.sum(
            pc.and_(
                pc.greater_equal(valid_delays, 15),
                pc.less(valid_delays, 60),
            )
        ).as_py() or 0,
        "severe_delay": pc.sum(
            pc.greater_equal(valid_delays, 60)
        ).as_py() or 0,
    })

    return counts, total


def build_splits(config_path="configs/config.yaml"):
    """Stream raw Parquet files into three temporal datasets."""
    config = load_split_config(config_path)

    with Path(config_path).open(encoding="utf-8") as file:
        import yaml
        full_config = yaml.safe_load(file)

    raw_dir = Path(full_config["data"]["raw_dir"])
    output_dir = Path(config["output_dir"])
    batch_size = int(config["batch_size"])
    compression = config["compression"]

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    files = sorted(raw_dir.rglob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No Parquet files in {raw_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex
    final_run_dir = runs_dir / run_id

    temp_dir = Path(
        tempfile.mkdtemp(prefix=".temporal_splits_", dir=runs_dir)
    )

    writers = {}
    counts = {name: 0 for name in SPLIT_NAMES}
    class_counts = {name: Counter() for name in SPLIT_NAMES}
    eligible_counts = {name: 0 for name in SPLIT_NAMES}

    try:
        expected_schema = pq.ParquetFile(files[0]).schema_arrow

        for name in SPLIT_NAMES:
            writers[name] = pq.ParquetWriter(
                temp_dir / f"{name}.parquet",
                expected_schema,
                compression=compression,
            )

        for file in files:
            parquet_file = pq.ParquetFile(file)

            if not parquet_file.schema_arrow.equals(
                expected_schema, check_metadata=False
            ):
                raise ValueError(f"Schema mismatch: {file}")

            for batch in parquet_file.iter_batches(batch_size=batch_size):
                parts = assign_batch(batch, config)

                for name, table in parts.items():
                    if table.num_rows == 0:
                        continue

                    writers[name].write_table(table)
                    counts[name] += table.num_rows

                    if config.get("report_class_distribution", True):
                        current, eligible = count_classes(table)
                        class_counts[name].update(current)
                        eligible_counts[name] += eligible

            print(f"Processed: {file.name}")

        for writer in writers.values():
            writer.close()
        writers.clear()

        report = {}

        for name in SPLIT_NAMES:
            eligible = eligible_counts[name]

            report[name] = {
                "start": config[name]["start"],
                "end_exclusive": config[name]["end"],
                "total_rows": counts[name],
                "eligible_rows": eligible,
                "classes": {
                    cls: {
                        "count": class_counts[name][cls],
                        "percentage": round(
                            100 * class_counts[name][cls] / eligible, 2
                        ) if eligible else 0.0,
                    }
                    for cls in CLASS_NAMES
                },
            }

        with (temp_dir / "class_distribution.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(report, file, indent=2)

        # Verify outputs before publication.
        total_written = sum(counts.values())

        if total_written == 0:
            raise ValueError("No flights were written")

        if sum(eligible_counts.values()) > total_written:
            raise ValueError("Eligible flight count exceeds total rows")

        for name in SPLIT_NAMES:
            output_file = temp_dir / f"{name}.parquet"

            if not output_file.exists():
                raise FileNotFoundError(output_file)

            metadata = pq.read_metadata(output_file)

            if metadata.num_rows != counts[name]:
                raise ValueError(
                    f"Row count mismatch for {name}: "
                    f"{metadata.num_rows} != {counts[name]}"
                )

        # Publish the complete run as an immutable directory.
        temp_dir.rename(final_run_dir)

        manifest = {
            "run_id": run_id,
            "run_dir": str(final_run_dir),
            "total_rows": total_written,
            "splits": {
                name: {
                    "path": str(final_run_dir / f"{name}.parquet"),
                    "rows": counts[name],
                }
                for name in SPLIT_NAMES
            },
            "class_distribution": str(
                final_run_dir / "class_distribution.json"
            ),
        }

        manifest_path = output_dir / "temporal_split_manifest.json"
        temporary_manifest = output_dir / (
            f".temporal_split_manifest_{run_id}.tmp"
        )

        with temporary_manifest.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)

        # Atomically replace the active manifest.
        os.replace(temporary_manifest, manifest_path)

        print(f"\nPublished run: {run_id}")
        print(f"Manifest: {manifest_path}")

        print("\n=== TEMPORAL SPLIT SUMMARY ===")

        for name in SPLIT_NAMES:
            print(
                f"{name.upper()}: "
                f"{counts[name]:,} total flights, "
                f"{eligible_counts[name]:,} eligible flights"
            )

        print(f"\nSaved to: {final_run_dir}")
        return report

    finally:
        for writer in writers.values():
            writer.close()

        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML configuration",
    )
    args = parser.parse_args()
    build_splits(args.config)


if __name__ == "__main__":
    main()
