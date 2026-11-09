"""Data contract for raw BTS flight records (Issue #12, roadmap 1.6).

``COLUMNS`` is the single definition of every column the project knows about:
its TranStats form name, PREZIP source heading, type, valid range, whether it
is known before the scheduled departure, and its role in ``/classify``
requests. Everything else is derived from it:

* ingestion (``scripts/ingest_bts.py``): ``COLUMN_MAP``, ``EXPECTED``,
  ``INTEGER_COLUMNS``, ``FLOAT_COLUMNS``, ``RAW_VALUE_CHECK_COLUMNS`` and
  ``validate_values``;
* API requests (``src/api/schemas.py``): ``FlightRecord`` and
  ``records_to_frame``;
* leakage checks (Issue #3): ``POST_DEPARTURE_PATTERNS`` and
  ``find_post_departure_columns``.

Request policy (justified in ``docs/data_decisions.md``):

* required field missing, null, of the wrong type or out of range -> the
  record is rejected (validation error naming the instance and the field);
* optional field missing -> null;
* optional field of the wrong type -> rejected;
* optional field out of range -> null, with a warning in the log;
* unknown fields, including post-departure ones -> ignored, never returned.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Annotated, Any, Literal, Optional, Sequence

import pandas as pd
from pydantic import (BaseModel, BeforeValidator, ConfigDict, Field,
                      create_model, model_validator)

LOG = logging.getLogger("bts.contract")

Kind = Literal["int", "float", "code", "text", "date"]
RequestRole = Literal["required", "optional", "ignored"]
RawPolicy = Literal["error", "warn"]


@dataclass(frozen=True)
class ColumnSpec:
    """Contract for one column.

    kind: ``code`` values are stripped and upper-cased (carrier, airport and
    state codes); ``text`` values are only stripped (city and state names).
    source: PREZIP heading; ``None`` when derived (FL_DATE) or request-only.
    pre_departure: known before the scheduled departure. Only these columns
    may be sent to the model; the others are listed by POST_DEPARTURE_PATTERNS.
    request: role in /classify requests. Post-departure columns are always
    ``ignored``.
    in_raw: stored in the monthly Parquet files.
    min/max/allowed/hhmm/pattern: value rules, shared by ingestion and API.
    raw_policy: what ingestion does with a rule violation (``None``: the rule
    is applied only to requests).
    raw_required: ingestion rejects null values.
    """

    name: str
    kind: Kind
    source: Optional[str] = None
    pre_departure: bool = True
    request: RequestRole = "optional"
    in_raw: bool = True
    min: Optional[float] = None
    max: Optional[float] = None
    allowed: Optional[frozenset] = None
    hhmm: bool = False
    pattern: Optional[str] = None
    raw_policy: Optional[RawPolicy] = None
    raw_required: bool = False

    def describe(self) -> str:
        rules = []
        if self.allowed is not None:
            rules.append(f"one of {sorted(self.allowed)}")
        if self.min is not None:
            rules.append(f">= {self.min:g}")
        if self.max is not None:
            rules.append(f"<= {self.max:g}")
        if self.hhmm:
            rules.append("HHMM time 0000-2359 or 2400")
        if self.pattern:
            rules.append(f"format {self.pattern}")
        return ", ".join(rules) or "no value rule"


def _post(name: str, kind: Kind, source: str, **rules) -> ColumnSpec:
    """Post-departure column: kept in raw data, never accepted as model input."""
    return ColumnSpec(
        name, kind, source, pre_departure=False, request="ignored", **rules
    )


# Order matters: the raw Parquet schema follows the in_raw columns in this order.
COLUMNS: tuple[ColumnSpec, ...] = (
    # Calendar (derivable from FL_DATE, so optional in requests)
    ColumnSpec("YEAR", "int", "Year", min=1987, raw_required=True),
    ColumnSpec("QUARTER", "int", "Quarter", min=1, max=4),
    ColumnSpec("MONTH", "int", "Month", min=1, max=12, raw_required=True),
    ColumnSpec("DAY_OF_MONTH", "int", "DayofMonth", min=1, max=31, raw_required=True),
    ColumnSpec("DAY_OF_WEEK", "int", "DayOfWeek", min=1, max=7),
    # Carrier
    ColumnSpec("OP_UNIQUE_CARRIER", "code", "Reporting_Airline", request="required"),
    ColumnSpec("OP_CARRIER_AIRLINE_ID", "int", "DOT_ID_Reporting_Airline"),
    # The recorded tail is the aircraft that actually flew; swaps happen up to departure.
    _post("TAIL_NUM", "code", "Tail_Number"),
    ColumnSpec("OP_CARRIER_FL_NUM", "int", "Flight_Number_Reporting_Airline", min=1),
    # Origin
    ColumnSpec("ORIGIN_AIRPORT_ID", "int", "OriginAirportID"),
    ColumnSpec("ORIGIN_AIRPORT_SEQ_ID", "int", "OriginAirportSeqID"),
    ColumnSpec("ORIGIN_CITY_MARKET_ID", "int", "OriginCityMarketID"),
    ColumnSpec("ORIGIN", "code", "Origin", request="required"),
    ColumnSpec("ORIGIN_CITY_NAME", "text", "OriginCityName"),
    ColumnSpec("ORIGIN_STATE_ABR", "code", "OriginState"),
    ColumnSpec("ORIGIN_STATE_NM", "text", "OriginStateName"),
    ColumnSpec("ORIGIN_WAC", "int", "OriginWac"),
    # Destination
    ColumnSpec("DEST_AIRPORT_ID", "int", "DestAirportID"),
    ColumnSpec("DEST_AIRPORT_SEQ_ID", "int", "DestAirportSeqID"),
    ColumnSpec("DEST_CITY_MARKET_ID", "int", "DestCityMarketID"),
    ColumnSpec("DEST", "code", "Dest", request="required"),
    ColumnSpec("DEST_CITY_NAME", "text", "DestCityName"),
    ColumnSpec("DEST_STATE_ABR", "code", "DestState"),
    ColumnSpec("DEST_STATE_NM", "text", "DestStateName"),
    ColumnSpec("DEST_WAC", "int", "DestWac"),
    # Schedule and operations
    ColumnSpec(
        "CRS_DEP_TIME",
        "int",
        "CRSDepTime",
        request="required",
        hhmm=True,
        raw_policy="error",
    ),
    _post("DEP_TIME", "int", "DepTime"),
    _post("DEP_DELAY", "float", "DepDelay"),
    _post("TAXI_OUT", "float", "TaxiOut"),
    _post("TAXI_IN", "float", "TaxiIn"),
    ColumnSpec("CRS_ARR_TIME", "int", "CRSArrTime", hhmm=True, raw_policy="error"),
    _post("ARR_TIME", "int", "ArrTime"),
    _post("ARR_DELAY", "float", "ArrDelay"),
    _post(
        "CANCELLED", "float", "Cancelled", allowed=frozenset({0, 1}), raw_policy="error"
    ),
    _post(
        "DIVERTED", "float", "Diverted", allowed=frozenset({0, 1}), raw_policy="error"
    ),
    # Six negative values exist in the 2022/2023 source: logged, kept for preprocessing.
    ColumnSpec("CRS_ELAPSED_TIME", "float", "CRSElapsedTime", min=0, raw_policy="warn"),
    ColumnSpec("FLIGHTS", "int", "Flights", min=1),
    ColumnSpec("DISTANCE", "float", "Distance", min=0, raw_policy="error"),
    ColumnSpec("DISTANCE_GROUP", "int", "DistanceGroup", min=1, max=11),
    _post("FIRST_DEP_TIME", "int", "FirstDepTime"),
    _post("TOTAL_ADD_GTIME", "float", "TotalAddGTime"),
    _post("LONGEST_ADD_GTIME", "float", "LongestAddGTime"),
    # Derived at ingestion from YEAR/MONTH/DAY_OF_MONTH
    ColumnSpec("FL_DATE", "date", request="required", raw_required=True),
    # Request-only: TranStats form fields not stored in the raw files
    ColumnSpec("OP_CARRIER", "code", in_raw=False),
    ColumnSpec("DEP_TIME_BLK", "code", in_raw=False, pattern=r"\d{4}-\d{4}"),
    ColumnSpec("ARR_TIME_BLK", "code", in_raw=False, pattern=r"\d{4}-\d{4}"),
)
SPECS: dict[str, ColumnSpec] = {spec.name: spec for spec in COLUMNS}

# ---------------------------------------------------------------- ingestion
RAW_SPECS = tuple(spec for spec in COLUMNS if spec.in_raw)
COLUMN_MAP = {spec.source: spec.name for spec in RAW_SPECS if spec.source}
EXPECTED = [spec.name for spec in RAW_SPECS]
INTEGER_COLUMNS = {spec.name for spec in RAW_SPECS if spec.kind == "int"}
FLOAT_COLUMNS = {spec.name for spec in RAW_SPECS if spec.kind == "float"}
RAW_VALUE_CHECK_COLUMNS = [
    spec.name for spec in RAW_SPECS if spec.raw_required or spec.raw_policy
]

# ------------------------------------------------------------------ requests
REQUIRED_FIELDS = [spec.name for spec in COLUMNS if spec.request == "required"]
OPTIONAL_FIELDS = [spec.name for spec in COLUMNS if spec.request == "optional"]
REQUEST_COLUMNS = [spec.name for spec in COLUMNS if spec.request != "ignored"]

# ------------------------------------------------------------------- leakage
# Post-departure TranStats form fields that are not ingested.
_NOT_INGESTED_POST_DEPARTURE = (
    r"DEP_DELAY.*",
    r"DEP_DEL15",
    r"WHEELS_OFF",
    r"WHEELS_ON",
    r"ARR_DELAY.*",
    r"ARR_DEL15",
    r"ACTUAL_ELAPSED_TIME",
    r"AIR_TIME",
    r"CARRIER_DELAY",
    r"WEATHER_DELAY",
    r"NAS_DELAY",
    r"SECURITY_DELAY",
    r"LATE_AIRCRAFT_DELAY",
    r"CANCELLATION_CODE",
    r"DIV_.*",
    r"DIV\d+_.*",
)
POST_DEPARTURE_PATTERNS: tuple[str, ...] = tuple(
    dict.fromkeys(
        [rf"^{re.escape(spec.name)}$" for spec in COLUMNS if not spec.pre_departure]
        + [rf"^{pattern}$" for pattern in _NOT_INGESTED_POST_DEPARTURE]
    )
)


def find_post_departure_columns(columns: Sequence[str]) -> list[str]:
    """Return the columns that are only known after the scheduled departure."""
    return [
        column
        for column in columns
        if any(re.fullmatch(pattern, column) for pattern in POST_DEPARTURE_PATTERNS)
    ]


def _check_contract_consistency() -> None:
    for spec in COLUMNS:
        if not spec.pre_departure and spec.request != "ignored":
            raise AssertionError(
                f"{spec.name} is post-departure but accepted in requests"
            )
    leaked = find_post_departure_columns([s.name for s in COLUMNS if s.pre_departure])
    if leaked:
        raise AssertionError(
            f"Pre-departure columns match post-departure patterns: {leaked}"
        )


_check_contract_consistency()


# --------------------------------------------------------------- value rules
# The same rules exist twice: vectorized (ingestion, millions of rows) and
# scalar (per request value). tests/test_contract.py checks they agree.


def _valid_mask(spec: ColumnSpec, values: pd.Series) -> pd.Series:
    """Vectorized rule check on non-null values."""
    ok = pd.Series(True, index=values.index)
    if spec.allowed is not None:
        ok &= values.isin(spec.allowed)
    if spec.min is not None:
        ok &= values >= spec.min
    if spec.max is not None:
        ok &= values <= spec.max
    if spec.hhmm:
        ok &= ((values >= 0) & (values <= 2359) & (values % 100 < 60)) | (
            values == 2400
        )
    if spec.pattern:
        ok &= values.astype(str).str.fullmatch(spec.pattern)
    return ok.astype(bool)


def _value_ok(spec: ColumnSpec, value: Any) -> bool:
    """Scalar rule check on a non-null, already parsed value."""
    if spec.allowed is not None and value not in spec.allowed:
        return False
    if spec.min is not None and value < spec.min:
        return False
    if spec.max is not None and value > spec.max:
        return False
    if spec.hhmm and not ((0 <= value <= 2359 and value % 100 < 60) or value == 2400):
        return False
    if spec.pattern and not re.fullmatch(spec.pattern, str(value)):
        return False
    return True


def validate_values(frame: pd.DataFrame) -> None:
    """Validate raw ingested values without filtering cancelled/diverted flights."""
    for spec in RAW_SPECS:
        if not (spec.raw_required or spec.raw_policy):
            continue
        column = frame[spec.name]
        if spec.raw_required and column.isna().any():
            raise ValueError(f"Missing required values: {spec.name}")
        if spec.raw_policy is None:
            continue
        values = column.dropna()
        invalid = int((~_valid_mask(spec, values)).sum())
        if not invalid:
            continue
        message = (
            f"{invalid} {spec.name} values violate the contract ({spec.describe()})"
        )
        if spec.raw_policy == "error":
            raise ValueError(message)
        LOG.warning("Raw contract anomaly: %s; retained for preprocessing", message)
    expected = pd.to_datetime(
        dict(year=frame["YEAR"], month=frame["MONTH"], day=frame["DAY_OF_MONTH"]),
        errors="raise",
    )
    if not (
        frame["FL_DATE"].reset_index(drop=True) == expected.reset_index(drop=True)
    ).all():
        raise ValueError("FL_DATE does not match calendar components")


# ----------------------------------------------------------- request parsing
_DATE_FORMATS = ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y")  # TranStats CSV export formats


def _parse_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"expected an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            number = math.nan
        if number.is_integer():
            return int(number)
    raise ValueError(f"expected an integer, got {value!r}")


def _parse_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"expected a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if math.isfinite(number):
        return number
    raise ValueError(f"expected a finite number, got {value!r}")


def _parse_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            pass
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                pass
    raise ValueError(f"expected a date (YYYY-MM-DD or M/D/YYYY), got {value!r}")


def _parse_string(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"expected a string, got {value!r}")
    return value


_PARSERS = {
    "int": _parse_int,
    "float": _parse_float,
    "date": _parse_date,
    "code": _parse_string,
    "text": _parse_string,
}
_PYTHON_TYPES = {"int": int, "float": float, "date": date, "code": str, "text": str}


def _field_parser(spec: ColumnSpec):
    required = spec.request == "required"

    def parse(value: Any):
        if isinstance(value, str):
            value = value.strip()
            if spec.kind == "code":
                value = value.upper()
            if value == "":
                value = None
        if value is None:
            if required:
                raise ValueError("required field must not be null or empty")
            return None
        value = _PARSERS[spec.kind](value)  # wrong type -> ValueError (rejected)
        if not _value_ok(spec, value):
            if required:
                raise ValueError(f"{value!r} violates the contract ({spec.describe()})")
            LOG.warning(
                "%s=%r violates the contract (%s); treated as missing",
                spec.name,
                value,
                spec.describe(),
            )
            return None
        return value

    return parse


class _FlightRecordBase(BaseModel):
    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _carrier_alias(cls, data: Any) -> Any:
        """OP_CARRIER (IATA code) stands in for OP_UNIQUE_CARRIER when absent."""
        if isinstance(data, dict) and data.get("OP_UNIQUE_CARRIER") in (None, ""):
            if data.get("OP_CARRIER") not in (None, ""):
                data = {**data, "OP_UNIQUE_CARRIER": data["OP_CARRIER"]}
        return data


def _request_field(spec: ColumnSpec):
    annotation = Annotated[
        Optional[_PYTHON_TYPES[spec.kind]], BeforeValidator(_field_parser(spec))
    ]
    default = ... if spec.request == "required" else None
    return annotation, Field(default=default, description=spec.describe())


FlightRecord = create_model(
    "FlightRecord",
    __base__=_FlightRecordBase,
    __doc__="One flight in a /classify request, validated against the data contract.",
    **{
        spec.name: _request_field(spec) for spec in COLUMNS if spec.request != "ignored"
    },
)


def pandas_dtype(spec: ColumnSpec) -> str:
    """pandas dtype of a column, shared by API requests and preprocessing."""
    return {
        "int": "Int64",
        "float": "Float64",
        "code": "string",
        "text": "string",
        "date": "datetime64[ns]",
    }[spec.kind]


def records_to_frame(records: Sequence[BaseModel]) -> pd.DataFrame:
    """Validated records -> DataFrame in input order, typed like the raw Parquet data."""
    frame = pd.DataFrame(
        [record.model_dump() for record in records], columns=REQUEST_COLUMNS
    )
    for name in REQUEST_COLUMNS:
        spec = SPECS[name]
        if spec.kind == "date":
            frame[name] = pd.to_datetime(frame[name])
        else:
            frame[name] = frame[name].astype(pandas_dtype(spec))
    return frame
