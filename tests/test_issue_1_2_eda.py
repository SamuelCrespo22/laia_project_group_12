"""Issue 1.2: small synthetic BTS fixtures. No real Parquet data needed.

These tests reproduce the EDA transformations and verify their expected semantics.
For full integration coverage, move the relevant notebook transformations into
importable project functions and invoke those functions here.
"""

import json
from datetime import date

import pytest

pl = pytest.importorskip(
    "polars", reason="Install project requirements to run Polars EDA fixture tests"
)


def fixture_flights():
    return pl.DataFrame(
        {
            "FL_DATE": [
                date(2022, 12, 31),
                date(2022, 12, 31),
                date(2023, 1, 1),
                date(2023, 1, 1),
                date(2023, 1, 2),
                date(2023, 1, 2),
            ],
            "YEAR": [2022, 2022, 2023, 2023, 2023, 2023],
            "OP_UNIQUE_CARRIER": ["AA", "AA", "BB", "AA", "BB", "AA"],
            "ORIGIN": ["JFK", "JFK", "BOS", "JFK", "BOS", "JFK"],
            "DEST": ["BOS", "BOS", "JFK", "BOS", "JFK", "BOS"],
            "CANCELLED": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "DIVERTED": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "ARR_DELAY": [-3.0, 15.0, None, None, 60.0, None],
            "DEP_DELAY": [0.0, 22.0, None, 11.0, 61.0, 0.0],
            "CRS_DEP_TIME": [830.0, 2400.0, 2500.0, 1265.0, 1200.0, 0.0],
            "CRS_ARR_TIME": [1010.0, 100.0, 0.0, 2400.0, 1430.0, 200.0],
            "CRS_ELAPSED_TIME": [100.0, 100.0, 60.0, 90.0, 110.0, 80.0],
            "DISTANCE": [200.0, 200.0, 190.0, 200.0, 190.0, 200.0],
            "TAIL_NUM": ["N1", "N1", None, "N2", "N2", "N1"],
        }
    )


def test_arrival_target_filters_cancelled_diverted_and_missing():
    df = fixture_flights().lazy()
    eligible = df.filter(
        (pl.col("CANCELLED") == 0)
        & (pl.col("DIVERTED") == 0)
        & pl.col("ARR_DELAY").is_not_null()
    )
    result = eligible.with_columns(
        (pl.col("ARR_DELAY") >= 15).cast(pl.Int8).alias("TARGET_DELAY_15")
    ).collect()
    assert result.height == 3
    assert result["TARGET_DELAY_15"].to_list() == [0, 1, 1]
    assert result["ARR_DELAY"].to_list() == [-3.0, 15.0, 60.0]


def test_three_class_boundaries_from_configured_thresholds():
    moderate, severe = 15, 60
    values = pl.DataFrame(
        {"ARR_DELAY": [-10, 0, 14.999, 15, 59.999, 60, 120]},
        schema={"ARR_DELAY": pl.Float64},
    )
    labels = values.select(
        pl.when(pl.col("ARR_DELAY") < moderate)
        .then(pl.lit("on_time"))
        .when(pl.col("ARR_DELAY") < severe)
        .then(pl.lit("moderate_delay"))
        .otherwise(pl.lit("severe_delay"))
        .alias("DELAY_SEVERITY")
    )["DELAY_SEVERITY"].to_list()
    assert labels == [
        "on_time",
        "on_time",
        "on_time",
        "moderate_delay",
        "moderate_delay",
        "severe_delay",
        "severe_delay",
    ]


def test_missing_values_depend_on_status():
    data = fixture_flights()
    assert data.filter(pl.col("CANCELLED") == 1)["ARR_DELAY"].null_count() == 1
    assert data.filter(pl.col("DIVERTED") == 1)["ARR_DELAY"].null_count() == 1
    # One uncompleted/missing-target record is neither cancelled nor diverted.
    assert (
        data.filter(
            (pl.col("CANCELLED") == 0)
            & (pl.col("DIVERTED") == 0)
            & pl.col("ARR_DELAY").is_null()
        ).height
        == 1
    )


def test_float_coded_binary_status_is_checked_without_is_in_cast():
    df = pl.DataFrame({"CANCELLED": [0.0, 1.0, 0.5, -1.0, None]})
    v = pl.col("CANCELLED")
    invalid = df.select(
        (v.is_not_null() & (v != 0) & (v != 1))
        .fill_null(False)
        .sum()
        .alias("n_invalid")
    )
    assert invalid["n_invalid"][0] == 2


def test_hhmm_validation_includes_2400_and_rejects_2460():
    df = pl.DataFrame({"T": [0.0, 930.0, 2359.0, 2400.0, 1265.0, 2460.0, -1.0, None]})
    v = pl.col("T")
    invalid = v.is_not_null() & (
        (v < 0) | (v > 2400) | ((v % 100) >= 60) | (((v // 100) == 24) & (v != 2400))
    )
    assert df.select(invalid.fill_null(False).alias("invalid"))[
        "invalid"
    ].to_list() == [False, False, False, False, True, True, True, False]


def test_new_category_and_rare_thresholds():
    df = fixture_flights()
    counts = df.group_by("OP_UNIQUE_CARRIER").agg(
        pl.len().alias("flights"),
        (pl.col("YEAR") == 2022).sum().alias("flights_2022"),
        (pl.col("YEAR") == 2023).sum().alias("flights_2023"),
    )
    total = df.height
    counts = counts.with_columns(
        ((pl.col("flights") < 3) | (pl.col("flights") / total < 0.1)).alias("rare"),
        ((pl.col("flights_2022") == 0) & (pl.col("flights_2023") > 0)).alias(
            "new_in_2023"
        ),
    )
    bb = counts.filter(pl.col("OP_UNIQUE_CARRIER") == "BB").row(0, named=True)
    assert bb["new_in_2023"] is True
    assert bb["rare"] is True


def test_exclusive_train_end_prevents_2023_leakage(tmp_path):
    df = fixture_flights()
    train = df.filter(
        (pl.col("FL_DATE") >= date(2022, 1, 1)) & (pl.col("FL_DATE") < date(2023, 1, 1))
    )
    assert train.height == 2
    assert train["YEAR"].unique().to_list() == [2022]
    # Synthetic reference contains no future categories and is JSON-serializable.
    counts = train.group_by("OP_UNIQUE_CARRIER").len()
    stats = {r["OP_UNIQUE_CARRIER"]: r["len"] for r in counts.iter_rows(named=True)}
    output = tmp_path / "reference.json"
    output.write_text(json.dumps(stats), encoding="utf-8")
    assert json.loads(output.read_text()) == {"AA": 2}


def test_scheduled_arrival_traffic_excludes_target_from_volume():
    df = fixture_flights()
    arrivals = df.select(
        pl.len().alias("scheduled_arrivals"),
        (
            (pl.col("CANCELLED") == 0)
            & (pl.col("DIVERTED") == 0)
            & pl.col("ARR_DELAY").is_not_null()
        )
        .sum()
        .alias("eligible_arrivals"),
    )
    assert arrivals["scheduled_arrivals"][0] == 6
    assert arrivals["eligible_arrivals"][0] == 3
