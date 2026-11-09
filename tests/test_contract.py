"""Data contract (Issue #12): request validation, raw value rules and leakage lists."""

import logging
from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.app import app
from src.data import contract
from src.data.contract import FlightRecord, records_to_frame

VALID = {
    "FL_DATE": "2025-05-15",
    "OP_UNIQUE_CARRIER": "AA",
    "ORIGIN": "JFK",
    "DEST": "LAX",
    "CRS_DEP_TIME": 900,
}


def record(**changes):
    data = {**VALID, **changes}
    return FlightRecord.model_validate({k: v for k, v in data.items() if v is not ...})


def error_fields(exc_info):
    return {error["loc"][0] for error in exc_info.value.errors() if error["loc"]}


# ------------------------------------------------------------- valid records


def test_minimal_record_with_only_required_fields_is_valid():
    flight = record()
    assert flight.FL_DATE == date(2025, 5, 15)
    assert flight.CRS_DEP_TIME == 900
    assert flight.DISTANCE is None


def test_required_fields_are_the_agreed_five():
    assert set(contract.REQUIRED_FIELDS) == {
        "FL_DATE",
        "OP_UNIQUE_CARRIER",
        "ORIGIN",
        "DEST",
        "CRS_DEP_TIME",
    }


def test_missing_optional_fields_become_null_in_frame():
    frame = records_to_frame([record()])
    assert list(frame.columns) == contract.REQUEST_COLUMNS
    assert frame["CRS_ARR_TIME"].isna().all()
    assert frame["DEP_TIME_BLK"].isna().all()


def test_unknown_and_post_departure_fields_are_ignored():
    flight = record(ARR_DELAY=75.0, DEP_TIME=915, TAXI_OUT=12, SOMETHING_NEW="x")
    frame = records_to_frame([flight])
    assert contract.find_post_departure_columns(frame.columns) == []
    assert "SOMETHING_NEW" not in frame.columns


def test_unseen_categories_are_accepted():
    flight = record(OP_UNIQUE_CARRIER="ZZ", ORIGIN="XYZ", DEST="QQQ")
    assert (flight.OP_UNIQUE_CARRIER, flight.ORIGIN, flight.DEST) == (
        "ZZ",
        "XYZ",
        "QQQ",
    )


def test_codes_are_stripped_and_upper_cased():
    assert record(ORIGIN=" jfk ").ORIGIN == "JFK"


def test_op_carrier_stands_in_for_missing_op_unique_carrier():
    flight = record(OP_UNIQUE_CARRIER=..., OP_CARRIER="DL")
    assert flight.OP_UNIQUE_CARRIER == "DL"


@pytest.mark.parametrize(
    "value", ["2025-05-15", "5/15/2025 12:00:00 AM", "5/15/2025", "2025-05-15T00:00:00"]
)
def test_fl_date_accepts_iso_and_transtats_formats(value):
    assert record(FL_DATE=value).FL_DATE == date(2025, 5, 15)


@pytest.mark.parametrize("value", [900, "0900", "900", 900.0])
def test_hhmm_accepts_integer_like_values(value):
    assert record(CRS_DEP_TIME=value).CRS_DEP_TIME == 900


def test_frame_preserves_order_and_raw_dtypes():
    flights = [
        record(ORIGIN=code, DISTANCE=100.0 * i, CRS_ARR_TIME=1200)
        for i, code in enumerate(["AAA", "BBB", "CCC"])
    ]
    frame = records_to_frame(flights)
    assert frame["ORIGIN"].tolist() == ["AAA", "BBB", "CCC"]
    assert str(frame["CRS_DEP_TIME"].dtype) == "Int64"
    assert str(frame["DISTANCE"].dtype) == "Float64"
    assert str(frame["ORIGIN"].dtype) == "string"
    assert pd.api.types.is_datetime64_any_dtype(frame["FL_DATE"])


# ----------------------------------------------------------- invalid records


@pytest.mark.parametrize(
    "field", ["FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "CRS_DEP_TIME"]
)
@pytest.mark.parametrize("value", [..., None, "", "   "])
def test_required_field_missing_null_or_empty_is_rejected(field, value):
    with pytest.raises(ValidationError) as exc_info:
        record(**{field: value})
    assert field in error_fields(exc_info)


@pytest.mark.parametrize(
    "field,value",
    [
        ("CRS_DEP_TIME", "nine"),
        ("CRS_DEP_TIME", True),
        ("CRS_DEP_TIME", 9.5),
        ("FL_DATE", "15/05/2025"),
        ("FL_DATE", 20250515),
        ("ORIGIN", 123),
        ("DISTANCE", "far"),
        ("MONTH", [5]),  # optional fields of the wrong type
    ],
)
def test_wrong_type_is_rejected(field, value):
    with pytest.raises(ValidationError) as exc_info:
        record(**{field: value})
    assert field in error_fields(exc_info)


@pytest.mark.parametrize("value", [2460, 1260, -5, 2401])
def test_required_field_out_of_range_is_rejected(value):
    with pytest.raises(ValidationError) as exc_info:
        record(CRS_DEP_TIME=value)
    assert "CRS_DEP_TIME" in error_fields(exc_info)


@pytest.mark.parametrize(
    "field,value",
    [
        ("CRS_ARR_TIME", 2575),
        ("DAY_OF_WEEK", 9),
        ("MONTH", 13),
        ("DISTANCE", -5),
        ("CRS_ELAPSED_TIME", -10),
        ("DISTANCE_GROUP", 0),
        ("DEP_TIME_BLK", "morning"),
    ],
)
def test_optional_field_out_of_range_becomes_null_with_warning(field, value, caplog):
    caplog.set_level(logging.WARNING, logger="bts.contract")
    assert getattr(record(**{field: value}), field) is None
    assert field in caplog.text


# ------------------------------------------------------------- value rules

RULE_CASES = [
    ("CRS_DEP_TIME", [0, 1, 59, 60, 1259, 1260, 2359, 2360, 2400, 2401, -1]),
    ("CANCELLED", [0.0, 1.0, 0.5, 2.0]),
    ("DISTANCE", [0.0, 10.0, -0.1]),
    ("MONTH", [0, 1, 12, 13]),
    ("DEP_TIME_BLK", ["0600-0659", "0001-0559", "06-07", "x"]),
]


@pytest.mark.parametrize("name,values", RULE_CASES)
def test_scalar_and_vectorized_rules_agree(name, values):
    spec = contract.SPECS[name]
    vectorized = contract._valid_mask(spec, pd.Series(values)).tolist()
    scalar = [contract._value_ok(spec, value) for value in values]
    assert vectorized == scalar


def raw_frame(**changes):
    frame = pd.DataFrame(
        {
            "YEAR": [2022, 2022],
            "MONTH": [1, 1],
            "DAY_OF_MONTH": [1, 2],
            "FL_DATE": pd.to_datetime(["2022-01-01", "2022-01-02"]),
            "CRS_DEP_TIME": [900, 2400],
            "CRS_ARR_TIME": [1100, 30],
            "CANCELLED": [0.0, 1.0],
            "DIVERTED": [0.0, 0.0],
            "CRS_ELAPSED_TIME": [120.0, None],
            "DISTANCE": [500.0, 800.0],
        }
    )
    for column, value in changes.items():
        frame.loc[0, column] = value
    return frame


def test_raw_values_valid_frame_passes():
    contract.validate_values(raw_frame())


@pytest.mark.parametrize(
    "column,value",
    [("DIVERTED", 3.0), ("DISTANCE", -1.0), ("CRS_ARR_TIME", 2575), ("MONTH", None)],
)
def test_raw_values_violations_raise(column, value):
    with pytest.raises(ValueError):
        contract.validate_values(raw_frame(**{column: value}))


def test_raw_negative_scheduled_duration_is_logged_not_rejected(caplog):
    caplog.set_level(logging.WARNING, logger="bts.contract")
    contract.validate_values(raw_frame(CRS_ELAPSED_TIME=-5.0))
    assert "CRS_ELAPSED_TIME" in caplog.text


# ------------------------------------------------------------------- leakage


def test_post_departure_columns_are_detected():
    columns = [
        "DEP_TIME",
        "DEP_DELAY_NEW",
        "DEP_DEL15",
        "ARR_DELAY_GROUP",
        "WHEELS_ON",
        "DIV_AIRPORT_LANDINGS",
        "CANCELLATION_CODE",
        "TAIL_NUM",
        "CANCELLED",
        "CRS_DEP_TIME",
        "DEP_TIME_BLK",
        "ARR_TIME_BLK",
        "ORIGIN",
        "DISTANCE",
    ]
    assert contract.find_post_departure_columns(columns) == columns[:9]


def test_no_request_column_is_post_departure():
    assert contract.find_post_departure_columns(contract.REQUEST_COLUMNS) == []


def test_every_raw_column_has_a_source_or_is_derived():
    assert [s.name for s in contract.RAW_SPECS if s.source is None] == ["FL_DATE"]


# ----------------------------------------------------------------------- API

client = TestClient(app)


def test_api_accepts_record_with_only_required_fields():
    response = client.post(
        "/classify", json={"instances": [VALID, {**VALID, "EXTRA": 1}]}
    )
    assert response.status_code == 200
    assert len(response.json()["classifications"]) == 2


def test_api_error_names_instance_and_field():
    bad = {k: v for k, v in VALID.items() if k != "ORIGIN"}
    response = client.post("/classify", json={"instances": [VALID, bad]})
    assert response.status_code == 422
    locations = [error["loc"] for error in response.json()["detail"]]
    assert ["body", "instances", 1, "ORIGIN"] in locations
