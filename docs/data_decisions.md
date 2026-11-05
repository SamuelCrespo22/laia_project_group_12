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

## Issue #3: leakage prevention and preprocessing

Run `python -m src.data.make_dataset` from the repository root. It reads the
monthly raw files in file-name order, keeps eligible flights, builds the
features and the target, and writes `data/processed/dataset.parquet` plus
`data/processed/dataset_manifest.json` (input files, row counts per step,
class counts, output SHA-256). Parameters are in the `target` and
`preprocessing` sections of `configs/config.yaml`. The output is written to a
temporary file and moved into place when complete; nothing time-dependent is
stored, so re-runs on the same raw data give byte-identical files.

### Eligible flights and target

Cancelled (`CANCELLED==1`) and diverted (`DIVERTED==1`) flights have no
arrival delay to predict, and flights without `ARR_DELAY` have no label: all
are dropped. A null cancellation/diversion flag is not proof of a valid
flight, so those rows are dropped too (none exist in 2022/2023).

| Step (2022-2023) | Rows |
| --- | ---: |
| Raw flights | 13,577,024 |
| Cancelled | -269,229 |
| Diverted | -32,378 |
| Missing `ARR_DELAY` | -2 |
| Model dataset | 13,275,415 |

`delay_severity` comes from `ARR_DELAY` with the thresholds in
`target.thresholds`: `< 15` on_time (10,511,918; 79.2%), `15..59`
moderate_delay (1,843,805; 13.9%), `>= 60` severe_delay (919,692; 6.9%). The
imbalance is handled at training time (Issue 2.1), not by resampling here.

### Allowlist

Only the columns in `preprocessing.feature_allowlist` reach the feature step;
every other column is dropped, so a new raw column can never leak by default.
The allowlist is checked when the configuration is loaded: each column must
be in the data contract and be pre-departure there, otherwise loading fails.
Which columns are post-departure is decided only by the contract
(`POST_DEPARTURE_PATTERNS`); this issue adds the numbered diversion fields
(`DIV1_AIRPORT`, ...), which `DIV_*` did not match. A test checks that every
column listed in Issue #3 is flagged and that none of them reaches the output.

| Column | Why it is kept |
| --- | --- |
| `FL_DATE` | Scheduled date; used by the temporal splits (1.5) and for date features |
| `MONTH`, `DAY_OF_WEEK` | Seasonality and weekly pattern |
| `OP_UNIQUE_CARRIER` | Airline; also a monitoring subgroup |
| `ORIGIN`, `DEST` | Airports (congestion, weather exposure); origin is a monitoring subgroup |
| `CRS_DEP_TIME`, `DEP_TIME_BLK` | Scheduled departure time; delays accumulate during the day |
| `CRS_ARR_TIME`, `ARR_TIME_BLK` | Scheduled arrival time |
| `CRS_ELAPSED_TIME`, `DISTANCE`, `DISTANCE_GROUP` | Scheduled duration and route length |

Not kept: `YEAR` (the model must predict years not seen in training),
`QUARTER` and `DAY_OF_MONTH` (redundant with `FL_DATE`/`MONTH`), numeric
airport/market/WAC identifiers and city/state names (redundant with the
airport codes), `OP_CARRIER_FL_NUM` (very high-cardinality identifier),
`FLIGHTS` (always 1), `TAIL_NUM` (post-departure, see Issue #12). This is the
candidate list of the issue; the EDA notebook (Issue 1.2) must confirm it or
change `feature_allowlist`.

### Features from a single record

`build_features` in `src/data/preprocessor.py` is the only feature function.
`make_dataset` uses it for training and `/classify` uses it for requests, so
both produce the same columns, values and dtypes (a test compares a request
with the equivalent raw row). Every feature is computed from its own record;
nothing uses other flights (e.g. the previous flight of the same aircraft).

Derived columns are always computed from their source when it is present, so
training rows (where time blocks are not stored) and requests get identical
values; a value sent in a request is used only when its source is null.

| Column | Derived from | Rule |
| --- | --- | --- |
| `MONTH`, `DAY_OF_WEEK` | `FL_DATE` | Calendar month; ISO weekday (Monday = 1, as BTS) |
| `DEP_TIME_BLK`, `ARR_TIME_BLK` | `CRS_DEP_TIME`, `CRS_ARR_TIME` | BTS blocks: `0001-0559`, then one per hour (`0600-0659` ... `2300-2359`); 0000 and 2400 fall in `0001-0559` |
| `DISTANCE_GROUP` | `DISTANCE` | `floor(DISTANCE / 250) + 1`, capped at 11 (BTS 250-mile intervals) |

On all 13.6 million raw rows the derived `MONTH`, `DAY_OF_WEEK` and
`DISTANCE_GROUP` equal the values supplied by BTS. Only six scheduled times
are 0000 or 2400.

Requests: post-departure fields are ignored, not rejected, following the
request policy of Issue #12. Allowlisted fields missing from a request are
null and are handled by the encoders (Issue 1.4). Row order is never changed,
so `/classify` returns predictions in input order.
