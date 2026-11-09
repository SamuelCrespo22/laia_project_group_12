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

## Issue #4: feature encoders

Encoders live in `src/features/encoders.py` and their parameters in the
`features` block of `configs/config.yaml`. `build_feature_encoder` returns an
unfitted transformer. It must be fitted on the training split only (2022-01 to
2023-06), and is saved with the classifier as one pipeline, so serving applies
the same encoding. Only standard scikit-learn classes are used, so the saved
object does not depend on project code.

| Model type | Categorical columns | Numeric columns |
| :--- | :--- | :--- |
| `linear` | One-hot, sparse | Median imputation, then scaling |
| `tree` | Ordinal codes | Median imputation |

Categorical columns come first in the output. For `tree` models they are
positions `0..4`, which LightGBM receives through
`fit(categorical_feature=...)`.

Target encoding is not used: without cross-fitting it leaks the target into the
training features, which is unjustified complexity for a baseline.

### Columns

Categorical: `OP_UNIQUE_CARRIER`, `ORIGIN`, `DEST`, `DEP_TIME_BLK`,
`ARR_TIME_BLK`. Numeric: `CRS_ELAPSED_TIME`, `DISTANCE`, `MONTH`, `DAY_OF_WEEK`,
`DISTANCE_GROUP`.

Excluded: `FL_DATE`, because the model predicts years it never saw, and
`CRS_DEP_TIME` and `CRS_ARR_TIME`, because the time blocks already carry that
information and HHMM values are not linear in time.

### Rare categories

`min_frequency` is a fraction of the rows used for fitting, so it keeps its
meaning when training uses a sample. The value is `0.0001`, about 981 flights on
the full training split. Measured on that split:

| Column | Categories | Grouped at `0.0001` | Flights affected | Grouped at `0.001` | Flights affected |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `OP_UNIQUE_CARRIER` | 17 | 0 | 0% | 0 | 0% |
| `ORIGIN` | 372 | 79 | 0.44% | 256 | 6.66% |
| `DEST` | 373 | 80 | 0.44% | 256 | 6.57% |

`0.001` is the share in the EDA rarity rule. It is not used as the default
because it would merge 256 airports, including some with thousands of flights,
and the EDA itself notes that the rule flags such airports. Issue #5 can compare
both values on the validation split; `0.0001` has not yet been validated against
model performance.

### Unseen and missing values

| Input | `linear` | `tree` |
| :--- | :--- | :--- |
| Rare category | Grouped column | Shared code for the rare group |
| Unseen category or null, column has a rare group | Grouped column | Code `-1` |
| Unseen category or null, column has no rare group | All zeros | Code `-1` |
| Null number | Training median | Training median |

No carrier or airport first appears in the validation or test periods, so
unseen categories are a safeguard for the centralized evaluation data.

On the full training split (9,811,971 flights), fitting took about 36 seconds
(`linear`, 648 output columns) and 31 seconds (`tree`, 10 columns) on an 8 GB
laptop, with a peak of 3.64 GB including the loaded data. The fitted encoder is
16 KB when serialized.

## Issue 1.5: Temporal train, validation and test splits

### Objective

The dataset is divided chronologically to simulate a realistic
future-flight prediction scenario.

Random splitting is not used because it would mix historical and
future observations, producing an evaluation less representative
of deployment on subsequent months.

### Temporal boundaries

All boundaries are defined in `configs/config.yaml`.

| Split | Start (inclusive) | End (exclusive) | Months |
| --- | --- | --- | --- |
| Train | 2022-01-01 | 2023-07-01 | 18 |
| Validation | 2023-07-01 | 2023-10-01 | 3 |
| Test | 2023-10-01 | 2024-01-01 | 3 |

The split uses the original BTS `FL_DATE` column.

Intervals are contiguous and non-overlapping. Records outside the
configured range are rejected.

### Seasonality and distribution shift

Exploratory analysis of the eligible flights identified substantial
monthly variation in delay severity.

For example:

- June 2023: 10.13% severe delays.
- July 2023: 11.39% severe delays.
- September 2023: 6.22% severe delays.
- November 2023: 3.55% severe delays.
- December 2022: 9.85% severe delays.

These observations motivate evaluating the model on future months
rather than using a random split.

The validation period includes the high-delay summer months of
July and August 2023, while the test period covers the final
quarter of 2023.

Only two calendar years are available, so these observations do
not establish a stable long-term seasonal pattern.

### Dataset preservation and eligibility

The temporal split preserves the original BTS columns and records.

Flight eligibility and target generation belong to preprocessing.

For the class-distribution report, eligible flights satisfy:

- `CANCELLED == 0`
- `DIVERTED == 0`
- `ARR_DELAY` is not missing

Delay severity is defined as:

- `on_time`: arrival delay below 15 minutes
- `moderate_delay`: arrival delay from 15 to below 60 minutes
- `severe_delay`: arrival delay of 60 minutes or more

The arrival delay is used to construct the target and must never
be included among prediction features.

### Reproducibility and storage

The pipeline is implemented in:

- `src/data/temporal_split.py`
- `scripts/create_temporal_splits.py`

Run from the repository root:

`python -m scripts.create_temporal_splits --config configs/config.yaml`

PyArrow processes Parquet files incrementally using the configured
batch size and Zstandard compression.

Each successful execution creates a versioned directory under
`data/processed/runs/`.

The active version is identified by:

`data/processed/temporal_split_manifest.json`

Downstream training code must read dataset paths from this
manifest rather than assume fixed Parquet filenames.

Generated processed datasets are excluded from Git.

### Observed dataset sizes

The complete BTS dataset contains 13,577,024 flight records.

| Split | Total records | Eligible records |
| --- | ---: | ---: |
| Train | 10,069,694 | 9,811,971 |
| Validation | 1,774,191 | 1,738,087 |
| Test | 1,733,139 | 1,725,357 |
| Total | 13,577,024 | 13,275,415 |

The persisted Parquet files were checked against the manifest.
Row counts, date boundaries and class totals passed validation.

### Model development policy

The training split is used for model fitting.

The validation split is used for model comparison, hyperparameter
selection and decision-making during development.

The test split is reserved for final evaluation after model
selection. It must not be used for tuning or choosing models.

If cross-validation is introduced, it must preserve temporal
ordering and use training-period data only.

### Automated verification

Synthetic tests verify temporal boundaries, determinism,
missing dates, out-of-range records, YAML configuration,
Parquet persistence, class counts and preservation of the
previously published manifest after an input-reading failure.

The real-data execution additionally verified the expected
13,577,024 records and temporal boundaries.

The tests do not yet cover every possible interruption or
filesystem failure.

---

# LAIA ÔÇö Data Understanding and Decisions (Issue 1.2)

## Dataset and target population

BTS TranStats, 2022ÔÇô2023: **13,577,024** reported flights. Arrival-delay analysis includes **13,275,415 (97.78%)** flights with `CANCELLED == 0`, `DIVERTED == 0` and observed `ARR_DELAY`. It excludes **269,229 (1.98%)** cancelled flights, **32,378 (0.24%)** diverted flights and **2** otherwise eligible records missing `ARR_DELAY`. Do not assign an arrival-delay class to excluded flights.

## Target thresholds

Configured class definitions: `on_time` (`ARR_DELAY < 15`), `moderate_delay` (`15 <= ARR_DELAY < 60`) and `severe_delay` (`ARR_DELAY >= 60`). Among eligible flights: **79.18%**, **13.89%** and **6.93%** (919,692 severe), respectively. These cutoffs are configurable in `configs/config.yaml`; **team approval and rationale remain required**. Retain all classes and report classwise proportions.

## Missingness and quality

`ARR_DELAY` is absent for 100% of cancelled and diverted flights and for only two flights outside those groups. `FIRST_DEP_TIME`, `TOTAL_ADD_GTIME` and `LONGEST_ADD_GTIME` are each over **99.37%** missing; these are specialised operational fields, not general flight measurements. `TAIL_NUM` is missing in **50,046 (0.37%)** of all reported records but not in eligible completed flights. The overall `ARR_DELAY` missing share is **2.93% (2022)** and **1.53% (2023)**, largely tracking operational statuses rather than proving a change in data quality. Inspect flagged non-positive scheduled durations; do not delete them automatically.

## Temporal and spatial observations

The eligible-flight arrival-delay rate (at least 15 minutes) is **20.82%**. Scheduled departure time, weekday and month are associated with different delay rates and severity compositions. Retain scheduled calendar/time variables. `CRS_DEP_TIME` is an HHMM local time, not a continuous count of minutes; any derived hour/period representation must preserve local-time semantics. `DEP_TIME_BLK` was reconstructed from `CRS_DEP_TIME` because it is absent from the local Parquet schema.

The dataset contains **17 airlines**, **373 airports**, and **7,163 directed routes**. No airport or airline code is first observed in 2023, but route composition changes between years. The rarity rule (`min_count=100` **or** `min_share=0.001`) flags many airport codes, including some with thousands of flights. Report absolute and relative rarity separately and base any category handling on the training period, not combined 2022ÔÇô2023 data. Preserve `OP_UNIQUE_CARRIER`, `ORIGIN`, and `DEST` separately while evaluating the handling of unseen route pairs.

## Differences between years

Median `DISTANCE` changes from **666 to 679 miles**; median `CRS_ELAPSED_TIME` changes from **126 to 129 minutes**. The carrier mix changes, including `WN` increasing from **19.43% to 21.01%** of reported flights. These are evidence of distribution changes, not of predictive performance decline. A chronological comparison is preferable to a random year-mixed evaluation when assessing changes over time.

## Arrival traffic

The unadjusted delay rate is **20.26%** in destination airport-hour windows below **0.75├ù** usual scheduled arrival volume and **23.19%** in windows at least **1.5├ù** usual volume (difference **2.93 percentage points**). In the 06:00ÔÇô23:00 sensitivity subset, the rates are **20.13%** and **22.25%**. Consider scheduled arrival intensity and relative pressure as exploratory derived variables; do **not** infer congestion causality. The existing exploratory calculation uses `FL_DATE` as an approximate arrival date and a both-years reference denominator. Correct calendar/fusos and construct a time-valid, training-only reference before operational use.

## Scheduled duration

Across **6,388** comparable routeÔÇôcarrier combinations, the mean year-over-year change in observed duration buffer is **ÔêÆ0.05 minutes** (median **ÔêÆ0.14**). There is no evidence of an overall increase. Keep `CRS_ELAPSED_TIME` and `DISTANCE`. A route-relative scheduled-duration measure is an optional hypothesis, not an established improvement, and would need historical reference values computed without future information.

## Configuration and references

The current EDA configuration uses **2022** (`2022-01-01` inclusive to `2023-01-01` exclusive) as a **provisional** reference period and compares it with **2023**. Confirm the final training/validation/test cutoffs before reusing reference statistics downstream. The notebook's export reports `data/reference/train_feature_statistics.json` for **6,729,125** training-period records. Check that its feature inventory covers the distributions required for later monitoring; the configured `reference_features` list is broader than the numeric and categorical lists used by the current export.

## Remaining steps to close Issue 1.2

- [ ] Approve the three-class cutoffs, prediction-time definition and chronological split dates; synchronise `configs/config.yaml` and this document.
- [ ] Investigate flagged quality anomalies and validate date handling for arrival traffic.
- [ ] Confirm the notebook runs from beginning to end in the target environment and the reference JSON is written to the configured path.
- [ ] Ensure the exported reference statistics cover the agreed feature set.
- [ ] Add and run unit tests with small synthetic fixtures; confirm CI passes.
- [ ] Commit `notebooks/01_eda_and_target_analysis.ipynb` and `docs/data_decisions.md` in a dedicated branch; open a PR referencing Issue 1.2, obtain teammate review, and merge.
