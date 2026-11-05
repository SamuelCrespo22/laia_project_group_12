# Data decisions

## Issue #1: complete BTS ingestion

Run `python scripts/ingest_bts.py` from the repository root. All ingestion
parameters are in `configs/config.yaml`. Process every month of 2022 and 2023;
do not sample or remove flights during ingestion. Complete data preserves
small airline/airport subgroups. Training-time sampling belongs to the
training configuration and is outside this issue.

### Source and names

Download one monthly Reporting Carrier On-Time Performance PREZIP archive
from `https://transtats.bts.gov/PREZIP/`. Read its single CSV in bounded
chunks, map PREZIP headings to uppercase TranStats form names, and derive
`FL_DATE` from `YEAR`, `MONTH` and `DAY_OF_MONTH`. Reject missing required
columns and invalid calendar values. Additional source columns are ignored.
`COLUMN_MAP` in `src/data/contract.py` is the executable list of columns.

Kept columns and their purpose:

| Purpose | Columns |
| --- | --- |
| Calendar and temporal analysis | `FL_DATE`, `YEAR`, `QUARTER`, `MONTH`, `DAY_OF_MONTH`, `DAY_OF_WEEK` |
| Carrier identifiers and subgroup monitoring | `OP_UNIQUE_CARRIER`, `OP_CARRIER_AIRLINE_ID`, `OP_CARRIER_FL_NUM` |
| Origin identifiers and geography | `ORIGIN_AIRPORT_ID`, `ORIGIN_AIRPORT_SEQ_ID`, `ORIGIN_CITY_MARKET_ID`, `ORIGIN`, `ORIGIN_CITY_NAME`, `ORIGIN_STATE_ABR`, `ORIGIN_STATE_NM`, `ORIGIN_WAC` |
| Destination identifiers and geography | `DEST_AIRPORT_ID`, `DEST_AIRPORT_SEQ_ID`, `DEST_CITY_MARKET_ID`, `DEST`, `DEST_CITY_NAME`, `DEST_STATE_ABR`, `DEST_STATE_NM`, `DEST_WAC` |
| Scheduled candidate features | `CRS_DEP_TIME`, `CRS_ARR_TIME`, `CRS_ELAPSED_TIME`, `DISTANCE`, `DISTANCE_GROUP` |
| Target and flight eligibility | `ARR_DELAY`, `CANCELLED`, `DIVERTED` |
| Raw operational information for later analysis | `TAIL_NUM`, `DEP_TIME`, `DEP_DELAY`, `TAXI_OUT`, `TAXI_IN`, `ARR_TIME`, `FLIGHTS`, `FIRST_DEP_TIME`, `TOTAL_ADD_GTIME`, `LONGEST_ADD_GTIME` |

Retaining a raw column does not approve it as a model feature. Actual
operational values, arrival delay and eligibility flags must not leak into
prediction inputs. Feature selection and flight filtering belong to
preprocessing, not ingestion.

### Storage, integrity and recovery

Store one Zstandard-compressed Parquet per month. Compared with CSV, this
preserves explicit nullable types and supports column-selective reads.
Use int64 for integer columns, float64 for numeric measures, strings for
categorical/text columns and a timestamp for `FL_DATE`.

Validate names, order, types, nonempty output, row count, year/month and
nonmissing calendar dates. Log schema success, SHA-256 and row counts even
when skipping an existing file. `manifest.json` records output checksums and
counts. The raw schema and value checks are defined in `src/data/contract.py`
for reuse by Issue #12 (1.6). Reject invalid binary flags, negative
distance, invalid scheduled HHMM times (allowing 2400), and dates
inconsistent with their calendar components. Nullable operational values
are retained. Six negative scheduled durations were found in the downloaded
2022/2023 source data: log these anomalies and preserve them for downstream
preprocessing rather than silently deleting flights or repeatedly downloading
the same source. API field requirements and integration remain under Issue #12.

Reuse only files matching their manifest checksum and passing validation.
Rebuild missing or corrupt months. Convert in a temporary directory and
replace the output only after validation. Update the manifest atomically.
Temporary archives are removed after each month, including on failure.
The download size limit and minimum free space are configurable. These
guards and chunked conversion reduce disk/memory pressure; an exhausted
disk still raises an error and leaves previously published files intact.

### Git LFS for the current delivery

Track `data/raw/*.parquet` with Git LFS: Git contains small pointers and
GitHub LFS stores the binary contents. Keep the manifest in ordinary Git.
Each member installs Git LFS, runs `git lfs install`, and obtains datasets
with `git lfs pull` after cloning/pulling the repository. Authors use normal
`git add`, `git commit` and `git push`; the LFS hook uploads the objects.

Record the Git commit with each training run to identify its data version.
VM-based dataset storage/versioning is deferred. Historical binary files
in pre-LFS commits remain; this delivery does not rewrite shared history.

## Issue #12: data contract and schema validation

`COLUMNS` in `src/data/contract.py` is the single definition of every
column: TranStats name, PREZIP heading, type, valid range, whether it is
known before the scheduled departure, and its role in `/classify` requests.
Ingestion (`COLUMN_MAP`, raw dtypes, `validate_values`), the API request
schema (`FlightRecord`) and the leakage checks (`POST_DEPARTURE_PATTERNS`)
are all derived from it, so training and serving cannot disagree on what a
valid flight looks like (no train/serve skew). It is a Python module rather
than a YAML file because ingestion already used it and Pydantic models can
be generated from it directly.

### Required and optional request fields

Required: `FL_DATE`, `OP_UNIQUE_CARRIER`, `ORIGIN`, `DEST`, `CRS_DEP_TIME`.
Without them a record does not identify a scheduled flight, and carrier and
origin are also the subgroups that monitoring must report on. Requiring more
fields would increase the risk of rejecting evaluation records that omit
them; every other pre-departure column is optional. Calendar fields
(`YEAR`, `MONTH`, `DAY_OF_WEEK`, ...) are derivable from `FL_DATE`, and
`DEP_TIME_BLK`/`ARR_TIME_BLK` from the scheduled times; these columns are not
stored in the raw files, so features should derive them.

`OP_CARRIER` (the IATA carrier code, also a TranStats form field) is used as
`OP_UNIQUE_CARRIER` when the latter is absent. Codes are stripped and
upper-cased. `FL_DATE` accepts ISO dates and the TranStats CSV format
(`5/15/2025 12:00:00 AM`); times accept `900` or `"0900"`.

### Error policy

| Situation | Result |
| --- | --- |
| Required field missing, null, empty, wrong type or out of range | Record rejected; the error names the instance index and the field |
| Optional field missing | Null |
| Optional field of the wrong type (e.g. `"DISTANCE": "far"`) | Record rejected |
| Optional field out of range (e.g. `DAY_OF_WEEK: 9`) | Null and a warning in the log |
| Unknown or post-departure field | Ignored and never passed on |
| Unseen category (new airline or airport) | Accepted; encoders handle it (Issue 1.4) |

A wrong type means the client sends malformed data and should be told. An
out-of-range optional value only loses that value, so it should not discard
the whole batch. The HTTP status code for rejected requests is decided in
Issue 2.3 (FastAPI currently returns 422).

At ingestion the same value rules apply, but a violation fails the month
(`error`) or is logged and kept (`warn`, used for negative
`CRS_ELAPSED_TIME`). Rules without an ingestion policy (e.g. `MONTH` in
1-12) are checked on requests only: raw data is preserved and filtering
belongs to preprocessing. Against the 2022/2023 data, the only values that violate any
rule are the six known negative scheduled durations.

### Leakage

Each column is marked as known or not known before the scheduled departure.
`POST_DEPARTURE_PATTERNS` lists the post-departure columns, including
TranStats fields that are not ingested (`WHEELS_OFF`, `*_DELAY`, `DIV_*`,
...), and is the source for the leakage checks in Issue #3. `TAIL_NUM` is
treated as post-departure because the recorded tail is the aircraft that
actually flew, and aircraft swaps happen up to departure. `CANCELLED` and
`DIVERTED` are outcomes, used only to filter flights.
