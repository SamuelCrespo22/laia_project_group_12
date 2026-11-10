"""Projected batch reads and deterministic day-stratified training samples."""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.preprocessor import (
    CLASSES,
    DERIVATION_SOURCES,
    LABEL_SOURCE_COLUMNS,
    TARGET_COLUMN,
    assert_no_leakage_columns,
    build_training_frame,
    filter_valid_flights,
)
from src.models.transformers import validate_scheduled_features


def raw_batches(path, period, preprocessing, batch_size=50000, extra_columns=()):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    parquet = pq.ParquetFile(path)
    required = set(preprocessing.feature_allowlist) - set(DERIVATION_SOURCES)
    required |= set(LABEL_SOURCE_COLUMNS) | {
        "FL_DATE",
        "ORIGIN",
        "DEST",
        "OP_UNIQUE_CARRIER",
    }
    required |= {
        DERIVATION_SOURCES[c]
        for c in preprocessing.feature_allowlist
        if c in DERIVATION_SOURCES
    }
    assert_no_leakage_columns(extra_columns)
    if set(extra_columns) & (
        set(preprocessing.feature_allowlist) | set(LABEL_SOURCE_COLUMNS)
    ):
        raise ValueError("History columns must be separate from the flight contract")
    required |= set(extra_columns)
    if required - set(parquet.schema_arrow.names):
        raise ValueError(
            f"Missing raw columns: {sorted(required - set(parquet.schema_arrow.names))}"
        )
    # Read outcomes for eligibility/target; never load other forbidden columns.
    columns = sorted(
        required
        | (set(preprocessing.feature_allowlist) & set(parquet.schema_arrow.names))
    )
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=columns, use_threads=False
    ):
        raw = batch.to_pandas()
        dates = pd.to_datetime(raw["FL_DATE"], errors="raise")
        if (
            dates.isna().any()
            or dates.dt.tz is not None
            or not (
                (dates >= pd.Timestamp(period["start"]))
                & (dates < pd.Timestamp(period["end"]))
            ).all()
        ):
            raise ValueError("Split dates violate configured temporal boundaries")
        raw["FL_DATE"] = dates
        for column in LABEL_SOURCE_COLUMNS:
            raw[column] = pd.to_numeric(raw[column], errors="raise")
            values = raw[column].dropna().astype(float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite values: {column}")
            if column != "ARR_DELAY" and not values.isin([0, 1]).all():
                raise ValueError(f"Invalid outcome flag: {column}")
        yield validate_scheduled_features(raw)


def _budget(frame_bytes, memory_mb):
    # Reserve headroom for typed features, fold copies, transforms and estimators.
    if memory_mb <= 0 or frame_bytes * 8 > memory_mb * 1024**2:
        raise MemoryError(
            "Training retention exceeds memory budget (8x dataframe estimate); "
            "reduce sample_size or increase memory_budget_mb"
        )


def _day_quotas(counts, size):
    if size < len(counts):
        raise ValueError("sample_size must cover every eligible training day")
    days = sorted(counts)
    remaining = np.array([counts[d] - 1 for d in days], dtype=float)
    if remaining.sum() == 0:
        return {d: 1 for d in days}
    shares = remaining * (size - len(days)) / remaining.sum()
    allocated = np.floor(shares).astype(int)
    extra = size - len(days) - allocated.sum()
    order = np.argsort(-(shares - allocated), kind="stable")
    allocated[order[:extra]] += 1
    return {d: int(n + 1) for d, n in zip(days, allocated)}


def prepare_split(
    path,
    period,
    preprocessing,
    sample_size=None,
    seed=42,
    batch_size=50000,
    memory_budget_mb=1024,
    extra_columns=(),
):
    """Sample raw eligible records before feature materialization, stratified by day.

    Two passes for bounded samples: day counts, then per-day random-priority
    reservoirs. RNG advances in file order, independently of batch boundaries.
    No validation/test statistics enter training sample allocation.
    """
    if sample_size is not None and (
        not isinstance(sample_size, int) or sample_size < 1
    ):
        raise ValueError("sample_size must be a positive integer or null")
    raw_count = eligible_count = 0
    counts, chunks = {}, []
    retained_bytes = 0
    for raw in raw_batches(path, period, preprocessing, batch_size, extra_columns):
        raw_count += len(raw)
        eligible = filter_valid_flights(raw)
        eligible_count += len(eligible)
        if sample_size is not None:
            for day, number in eligible.FL_DATE.dt.normalize().value_counts().items():
                counts[day] = counts.get(day, 0) + int(number)
        else:
            retained_bytes += int(eligible.memory_usage(deep=True).sum())
            _budget(retained_bytes, memory_budget_mb)
            chunks.append(eligible)
    if not eligible_count:
        raise ValueError("Empty split after filtering")
    if sample_size is not None:
        quotas = _day_quotas(counts, min(sample_size, eligible_count))
        rng = np.random.default_rng(seed)
        reservoirs, sizes = {}, {}
        for raw in raw_batches(path, period, preprocessing, batch_size, extra_columns):
            eligible = filter_valid_flights(raw)
            eligible = eligible.assign(_priority=rng.random(len(eligible)))
            for day, group in eligible.groupby(
                eligible.FL_DATE.dt.normalize(), sort=False
            ):
                previous = reservoirs.get(day)
                candidates = group if previous is None else pd.concat([previous, group])
                reservoirs[day] = candidates.nsmallest(quotas[day], "_priority")
                sizes[day] = int(reservoirs[day].memory_usage(deep=True).sum())
            _budget(sum(sizes.values()), memory_budget_mb)
        chunks = [reservoirs[d].drop(columns="_priority") for d in sorted(reservoirs)]
    selected = pd.concat(chunks, ignore_index=True)
    frame = training_frame(selected, preprocessing, extra_columns)
    frame = frame.sort_values("FL_DATE", kind="stable").reset_index(drop=True)
    X, y = frame.drop(columns=TARGET_COLUMN), frame[TARGET_COLUMN]
    assert_no_leakage_columns(X.columns)
    if not X.index.equals(y.index) or y.isna().any():
        raise ValueError("X/y alignment or target error")
    return (
        X,
        y,
        X[["OP_UNIQUE_CARRIER", "ORIGIN"]].copy(),
        {
            "raw": raw_count,
            "eligible": eligible_count,
            "used": len(frame),
            "classes": {c: int(y.eq(c).sum()) for c in CLASSES},
        },
    )


def training_frame(raw, preprocessing, extra_columns=()):
    """Delegate all flight filtering/labels/features to Issue 1.3, align history."""
    frame = build_training_frame(raw, preprocessing)
    if extra_columns:
        eligible = filter_valid_flights(raw).reset_index(drop=True)
        for column in extra_columns:
            frame[column] = eligible[column]
    return frame


def prepared_batches(path, period, preprocessing, batch_size=50000, extra_columns=()):
    """Integral evaluation with bounded feature/prediction materialization."""
    for raw in raw_batches(path, period, preprocessing, batch_size, extra_columns):
        frame = training_frame(raw, preprocessing, extra_columns)
        X, y = frame.drop(columns=TARGET_COLUMN), frame[TARGET_COLUMN]
        assert_no_leakage_columns(X.columns)
        yield X, y, X[["OP_UNIQUE_CARRIER", "ORIGIN"]], len(raw)
