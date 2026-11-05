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
