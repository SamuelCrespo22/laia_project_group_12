"""Build the model dataset: raw monthly Parquet -> eligible flights -> features + target.

Usage (from the repository root)::

    python -m src.data.make_dataset [--config configs/config.yaml]

Reads every ``flights_*.parquet`` in ``data.raw_dir`` one month at a time (in
file-name order), applies ``build_training_frame`` and streams the result to
``preprocessing.output_path``. The output is written to a temporary file and
moved into place only when complete. A manifest records the input files, row
counts at each step, the class distribution and the output SHA-256. Nothing
time-dependent is written, so re-runs on the same raw data give byte-identical
files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.preprocessor import (
    CLASSES,
    DERIVATION_SOURCES,
    LABEL_SOURCE_COLUMNS,
    TARGET_COLUMN,
    PreprocessingConfig,
    build_training_frame,
    load_config,
)

LOG = logging.getLogger("bts.make_dataset")


def raw_files(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.glob("flights_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No flights_*.parquet files in {raw_dir}")
    return files


def input_columns(path: Path, config: PreprocessingConfig) -> list[str]:
    """Columns to read: allowlisted features, their derivation sources and label sources."""
    available = pq.read_schema(path).names
    missing = [c for c in LABEL_SOURCE_COLUMNS if c not in available]
    if missing:
        raise ValueError(f"{path.name} lacks columns needed for the target: {missing}")
    wanted = set(config.feature_allowlist) | set(LABEL_SOURCE_COLUMNS)
    wanted |= {DERIVATION_SOURCES[c] for c in config.feature_allowlist if c in DERIVATION_SOURCES}
    return [c for c in available if c in wanted]


def drop_counts(raw: pd.DataFrame) -> dict[str, int]:
    """Rows removed by each filter, applied in order: cancelled, diverted, no ARR_DELAY."""
    cancelled = raw["CANCELLED"].ne(0).to_numpy(dtype=bool, na_value=True)
    diverted = ~cancelled & raw["DIVERTED"].ne(0).to_numpy(dtype=bool, na_value=True)
    no_delay = ~cancelled & ~diverted & raw["ARR_DELAY"].isna().to_numpy(dtype=bool)
    return {
        "cancelled": int(cancelled.sum()),
        "diverted": int(diverted.sum()),
        "missing_arr_delay": int(no_delay.sum()),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def build_dataset(config_path: Optional[Path | str] = None) -> dict:
    config = load_config(config_path)
    files = raw_files(config.raw_dir)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.output_path.with_name(config.output_path.name + ".tmp")

    writer: Optional[pq.ParquetWriter] = None
    schema: Optional[pa.Schema] = None
    inputs, dropped, classes = [], Counter(), Counter()
    rows_raw = rows_out = 0
    try:
        for path in files:
            raw = pd.read_parquet(path, columns=input_columns(path, config))
            data = build_training_frame(raw, config)
            table = pa.Table.from_pandas(data, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(tmp, schema, compression="zstd")
            writer.write_table(table.cast(schema))

            dropped.update(drop_counts(raw))
            classes.update(data[TARGET_COLUMN].value_counts().to_dict())
            rows_raw += len(raw)
            rows_out += len(data)
            inputs.append({"file": path.name, "rows_raw": len(raw), "rows_out": len(data)})
            LOG.info("%s: %d raw rows -> %d rows", path.name, len(raw), len(data))
    except BaseException:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)
        raise
    writer.close()
    os.replace(tmp, config.output_path)

    manifest = {
        "output": config.output_path.name,
        "sha256": sha256(config.output_path),
        "feature_columns": list(config.feature_allowlist),
        "target_column": TARGET_COLUMN,
        "thresholds": {
            "on_time_max": config.on_time_max,
            "moderate_delay_max": config.moderate_delay_max,
        },
        "rows_raw": rows_raw,
        "rows_dropped": dict(sorted(dropped.items())),
        "rows_out": rows_out,
        "class_counts": {name: int(classes.get(name, 0)) for name in CLASSES},
        "inputs": inputs,
    }
    _write_json(config.manifest_path, manifest)
    LOG.info(
        "Wrote %s (%d of %d rows, sha256 %s)",
        config.output_path,
        rows_out,
        rows_raw,
        manifest["sha256"],
    )
    return manifest


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_dataset(args.config)


if __name__ == "__main__":
    main()
