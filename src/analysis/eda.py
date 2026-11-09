"""Reusable, side-effect-free EDA helpers for LAIA Issue 1.2.

Large Polars aggregations stay in the notebook; these helpers operate on small
aggregated counts and have no dependency on the raw BTS dataset.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

TARGET_SEVERITY = "TARGET_SEVERITY"
SEVERITY_CLASSES = ("on_time", "moderate_delay", "severe_delay")
MISSING_CATEGORY = "__MISSING__"


def severity_label(value, moderate_min: float = 15, severe_min: float = 60):
    """Return the 3-class target, or None for missing/non-finite delay."""
    if moderate_min >= severe_min:
        raise ValueError("moderate_min must be smaller than severe_min")
    if value is None:
        return None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(value):
        return None
    if value < moderate_min:
        return SEVERITY_CLASSES[0]
    if value < severe_min:
        return SEVERITY_CLASSES[1]
    return SEVERITY_CLASSES[2]


def polars_severity_expr(
    column: str = "ARR_DELAY", moderate_min: float = 15,
    severe_min: float = 60, alias: str = TARGET_SEVERITY,
):
    """Vectorised version of severity_label for Polars DataFrame/LazyFrame."""
    if moderate_min >= severe_min:
        raise ValueError("moderate_min must be smaller than severe_min")
    import polars as pl

    v = pl.col(column)
    return (
        pl.when(v.is_null() | (~v.is_finite())).then(pl.lit(None, dtype=pl.String))
        .when(v < moderate_min).then(pl.lit(SEVERITY_CLASSES[0]))
        .when(v < severe_min).then(pl.lit(SEVERITY_CLASSES[1]))
        .otherwise(pl.lit(SEVERITY_CLASSES[2]))
        .alias(alias)
    )


def categorical_comparison(
    train_counts: Mapping[str, int], comparison_counts: Mapping[str, int],
    min_count: int = 100, min_share: float = 0.001,
):
    """Compare categorical frequencies; return unseen and rare TRAIN categories.

    All counts including the MISSING sentinel contribute to denominators.
    """
    if min_count < 1 or not (0 <= min_share <= 1):
        raise ValueError("Invalid category rarity thresholds")
    n_train = sum(train_counts.values())
    n_comp = sum(comparison_counts.values())
    if n_train <= 0 or n_comp <= 0:
        raise ValueError("Both periods must contain observations")
    for val in (*train_counts.values(), *comparison_counts.values()):
        if val < 0:
            raise ValueError("Negative category frequency")
    cats = sorted(set(train_counts) | set(comparison_counts))
    result = {
        "train_total": n_train,
        "comparison_total": n_comp,
        "train_cardinality": sum(k != MISSING_CATEGORY for k, v in train_counts.items() if v > 0),
        "comparison_cardinality": sum(k != MISSING_CATEGORY for k, v in comparison_counts.items() if v > 0),
        "new_in_comparison": sorted(
            k for k, v in comparison_counts.items()
            if k != MISSING_CATEGORY and v > 0 and train_counts.get(k, 0) == 0
        ),
        "rare_in_train": sorted(
            k for k, v in train_counts.items()
            if k != MISSING_CATEGORY and v > 0
            and (v < min_count or (v / n_train) < min_share)
        ),
        "total_variation_distance": 0.5 * sum(
            abs(train_counts.get(k, 0) / n_train - comparison_counts.get(k, 0) / n_comp)
            for k in cats
        ),
        "frequency_rows": [
            {"category": k,
             "train_count": train_counts.get(k, 0),
             "comparison_count": comparison_counts.get(k, 0),
             "train_share": train_counts.get(k, 0) / n_train,
             "comparison_share": comparison_counts.get(k, 0) / n_comp}
            for k in cats
        ],
    }
    return result


def build_reference_statistics(
    *,
    metadata: dict,
    population_count: int,
    categorical_counts: Mapping[str, Mapping[str, int]],
    numeric_bin_counts: Mapping[str, Mapping[str, int]],
    missing_counts: Mapping[str, int],
    numeric_summary: Mapping[str, dict],
) -> dict:
    """Produce a JSON-serializable train-only reference, from aggregated data."""
    if population_count <= 0:
        raise ValueError("Training population is empty")

    def distribution(counts):
        if any(v < 0 for v in counts.values()):
            raise ValueError("Negative histogram frequency")
        if sum(counts.values()) != population_count:
            raise ValueError("Histogram counts do not match training population")
        return {
            k: {"count": int(v), "share": v / population_count}
            for k, v in sorted(counts.items())
        }

    if any(v < 0 or v > population_count for v in missing_counts.values()):
        raise ValueError("Invalid missing-value counts")
    return {
        "metadata": metadata,
        "n_flights": int(population_count),
        "categorical": {k: distribution(v) for k, v in sorted(categorical_counts.items())},
        "numeric_histograms": {k: distribution(v) for k, v in sorted(numeric_bin_counts.items())},
        "numeric_summary": dict(sorted(numeric_summary.items())),
        "missingness": {
            k: {"count": int(v), "share": v / population_count}
            for k, v in sorted(missing_counts.items())
        },
    }


def write_json_deterministic(path: str | Path, data: dict) -> None:
    """Write stable UTF-8 JSON (no execution-time timestamp)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(contents, encoding="utf-8")
    tmp_path.replace(path)
