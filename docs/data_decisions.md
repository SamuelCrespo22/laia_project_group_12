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

In Issue 2.1, model and hyperparameter selection use expanding-window
cross-validation inside the training period. The fixed validation split
confirms performance on later months; its scores do not enter the ranking.

The test split is reserved for final evaluation after model
selection. It must not be used for tuning or choosing models.

The implemented cross-validation preserves temporal ordering, groups
boundaries by complete days, and uses training-period data only.

### Automated verification

Synthetic tests verify temporal boundaries, determinism,
missing dates, out-of-range records, YAML configuration,
Parquet persistence, class counts and preservation of the
previously published manifest after an input-reading failure.

The real-data execution additionally verified the expected
13,577,024 records and temporal boundaries.

The tests do not yet cover every possible interruption or
filesystem failure.

## Issue #5 (2.1): baseline training decisions and evidence

### Configuration and scope

`python -m src.models.train --config configs/config.yaml` runs the five
configured baselines end-to-end from the published Issue 1.5 manifest.
Paths, years, target thresholds, temporal intervals, feature columns, model
types/hyperparameters, sample size, seed, threads and batch sizes are explicit
in that YAML. Native fitted defaults and overrides are also exported in each
model's `config.json`; `random_state` and `n_jobs` come from the single global
seed/thread settings rather than conflicting per-model copies.

The default is the validated **controlled** profile: 100,000 training flights,
seed 42, one thread, 10,000-row read/evaluation batches and three CV folds.
Previously `sample_size: null` attempted to retain all 9,811,971 eligible
training flights and exceeded the 1,024 MiB retention guard. The controlled
default makes the ordinary one-command baseline usable with that budget.
Full-data fitting remains an explicit `sample_size: null` configuration with
an independently adequate memory budget; the controlled run is not evidence
of full-data training or final-test performance.

Explicit Logistic `multi_class: multinomial` and LightGBM `objective: multiclass`
name the backends that the previous library defaults already resolved for
three classes. Synthetic equivalence tests compare all five candidates with
their original implicit parameter sets. No target, split boundary or estimator
family has changed. Optional Modular/ablations/A-B-C settings in `configs/modular.yaml`
are separate from this default baseline; their real-data performance has not
been established and they are not used to justify the selection below.

### Inputs, eligibility and missing values

The reader resolves `data/processed/temporal_split_manifest.json` through
`load_published_splits()`, never a guessed train filename. It projects only
required columns and scans in bounded PyArrow batches. Issue 1.5's persisted
splits contain raw rows; training delegates filtering/features/target to
Issue 1.3 rather than assuming the split files are already model-ready.

Keep only `CANCELLED == 0`, `DIVERTED == 0` and observed `ARR_DELAY`; unknown
flags are excluded as well. Such rows lack an eligible completed-flight
label, so filtering them defines the prediction population. This excludes
no flight merely for a large finite arrival delay. Impossible/fractional
scheduled HHMM, negative distance and nonfinite schedule/outcome values fail
validation. Null optional schedule fields remain missing; nonpositive
scheduled duration becomes missing at the training adapter, preserving the
row and its label. This adds no change to the ingestion or split contracts.

Median imputation is learned only on each fold's training rows. It is a simple
reference robust to skewed duration/distance distributions and valid extremes;
it is not claimed to be an empirically optimal imputer. Entirely missing
columns keep a stable dimension. StandardScaler follows imputation for Logistic
Regression because HHMM, miles and minutes have different numerical scales
under L2 regularization. Dummy and LightGBM use passthrough scaling: the former
ignores feature values, and tree split ordering does not require standard units.

### Effective feature columns and encoders

Issue 1.3's 13-column allowlist remains the safety boundary. The baseline's
ColumnTransformer selects the following seven numeric columns, in this order:

| Feature | Origin | Reason for inclusion |
| --- | --- | --- |
| `MONTH` | Recomputed from `FL_DATE` | Calendar seasonality without a year identifier |
| `DAY_OF_WEEK` | Recomputed from `FL_DATE`, Monday=1 | Weekly operating patterns |
| `CRS_DEP_TIME` | Scheduled HHMM | Planned time of day is available before departure |
| `CRS_ARR_TIME` | Scheduled HHMM | Planned arrival time complements route/schedule information |
| `CRS_ELAPSED_TIME` | Scheduled minutes | Planned route duration; never actual elapsed time |
| `DISTANCE` | Scheduled route miles | Route length known before departure |
| `DISTANCE_GROUP` | `floor(DISTANCE/250)+1`, capped at 11 by Issue 1.3 | BTS distance bands as a coarse numeric reference |

`FL_DATE` is retained for ordering, calendar derivation and fold assignment,
but excluded from the estimator. `OP_UNIQUE_CARRIER` and `ORIGIN` remain aligned
subgroup metadata. `DEST`, carrier/origin codes and derived time-block strings
are excluded from `minimal_numeric` because it is the reproducible numeric
reference while official Issue #4 encoding remains a separate dependency.
This exclusion is not evidence that airport/carrier information is unhelpful.
Year, identifiers and post-departure variables remain excluded as justified
in Issue #3 above. `ARR_DELAY` creates y and never appears in X.

The optional `modular` mode uses native OneHotEncoder with `handle_unknown='ignore'`,
after a categorical SimpleImputer constant `__MISSING__`. This distinguishes
missing input and avoids failures on new codes without fitting categories on
validation. Both operations are inside the serialized train-fitted object.
No target encoding, frequency encoding or learned rare-category grouping is
claimed. There is **no rare-category threshold** in the controlled baseline;
official Issue #4 grouping and its EDA justification remain pending. These
optional encoders were tested synthetically, not used in the real numeric run.

Optional binning, feature selection, z-score treatment and tsfresh are disabled
in the controlled baseline. Optional Modular modes are documented below.
BTS contains no real per-record history for causal tsfresh extraction, so no
historical features were fabricated. No future-flight aggregate is used.
The inactive legacy IQR factor 1.5/minimum route size 20 and selector `k=all`
are exploratory compatibility defaults, not thresholds selected by this run
or an assertion of completed EDA justification. Optional Modular z-score threshold
3 is likewise a candidate convention, not an empirically selected BTS cutoff.

### Target thresholds

Use `target.thresholds.on_time_max=15` and `moderate_delay_max=60`: on_time
is `<15`, moderate is `15 <= delay <60`, severe is `>=60` minutes. These are
the fixed operational class definitions required by Issue 1.3, not cutoffs
optimized on validation/test. The 60-minute boundary explicitly isolates
hour-long delays; retaining the specified 15-minute tolerance and common
thresholds makes labels comparable across ingestion analysis, training and
serving. Tests cover both boundaries. There is no additional optimized
decision threshold: classifiers return their ordinary predicted class.

### Temporal sampling, cross-validation and fixed validation

Keep the existing intervals: train `[2022-01-01, 2023-07-01)`, fixed validation
`[2023-07-01, 2023-10-01)`, reserved test `[2023-10-01, 2024-01-01)`.
The seasonality rationale is documented in Issue 1.5 above. Input dates must
respect these intervals; training cannot silently mix published splits.

The 100,000-flight sample allocates approximately proportional quotas to every
eligible training day, with at least one row per day and seeded random-priority
selection within each day. This bounds retained training data while preserving
coverage of all 546 days/18 months. It is not class balancing and not a random
train/validation split. The sample contains 78,551 on_time, 14,221 moderate and
7,228 severe flights. Every candidate receives the same sample and folds.
The 100,000 size is a resource-controlled reference, supported by the completed
run's approximately 396 MiB process peak and minority examples in every fold;
it was not optimized against validation. The 1,024 MiB guard with conservative
8x DataFrame headroom protects retained data but is not an OS-enforced RAM cap.

| Fold | Train dates, inclusive | Train rows | CV dates, inclusive | CV rows |
| --- | --- | ---: | --- | ---: |
| 1 | 2022-01-01 to 2022-05-17 | 24,318 | 2022-05-18 to 2022-10-01 | 25,795 |
| 2 | 2022-01-01 to 2022-10-01 | 50,113 | 2022-10-02 to 2023-02-14 | 24,321 |
| 3 | 2022-01-01 to 2023-02-14 | 74,434 | 2023-02-15 to 2023-06-30 | 25,566 |

Sort by FL_DATE, divide unique days into four consecutive blocks, and expand
training over previous blocks. Never split a day across train/CV validation.
Each fit constructs a fresh complete pipeline, including imputation/scaling
and any enabled encoder/selector/sampler; no fitted mask or statistic crosses
fold boundaries. CV is the common comparison budget, not three months sampled
from the fixed validation period.

Rank by mean CV Macro F1, then lowest mean CV log loss, then experiment name.
Each candidate also receives a new fit on all 100,000 sampled training rows
and integral evaluation on the fixed validation's 1,738,087 eligible flights.
These fixed-validation results confirm generalization and do not enter ranking.
The test was neither evaluated nor hashed in these baseline runs. No model
was refitted on train plus validation, and no final-test performance is claimed.

### Class imbalance choice and observed results

The sample preserves the natural imbalance: 78.551% on_time, 14.221% moderate,
7.228% severe. Compare Dummy prior and both unweighted/balanced Logistic and
LightGBM. `class_weight='balanced'` uses `w_c = N / (3*n_c)` from that fit's y,
including within each CV training window. For the final sample the weights are
approximately 0.42435, 2.34395 and 4.61170 respectively. Preprocessing statistics
are unweighted; validation metrics use the observed validation distribution.
No over/under-sampling is used in the controlled baseline.

Observed completed run: `models/runs/20261109T232325685039Z`; official split
version: `40e5ff79a3824d949e1e78e9cf34e410`. The earlier instrumented run under
`models/experiments/issue5_controlled_20261109T210352Z` has the same CV and fixed
validation scores and records day coverage and an exact joblib round-trip check.
The table below is versioned documentation; generated model/data files remain
local artifacts, not required by synthetic tests or CI.

| Candidate | Mean CV Macro F1 | Mean CV log loss | Fixed-val Macro F1 | Fixed-val log loss | Severe F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dummy prior | 0.293052 | 0.659053 | 0.289963 | 0.692257 | 0 |
| Logistic | 0.293052 | 0.645517 | 0.289967 | 0.669883 | 0.000014 |
| Logistic balanced | 0.339880 | 1.075136 | 0.366728 | 1.055070 | 0.212883 |
| LightGBM | 0.295262 | 0.655351 | 0.290105 | 0.662470 | 0.000014 |
| LightGBM balanced | **0.363473** | 0.954210 | **0.382187** | 0.999798 | **0.219839** |

Select LightGBM balanced because it wins the specified CV Macro F1 objective,
not because fixed validation selected it. Unweighted classifiers almost always
predict on_time; class weighting makes the minority classes detectable. The
trade-off is explicit: balanced LightGBM has worse fixed-val log loss than
unweighted LightGBM. Severe recall/precision are 0.449518/0.145498; Logistic
balanced has higher severe recall 0.514798 but precision 0.134186 and lower
global Macro F1. This supports balanced LightGBM as this numeric baseline,
without claiming calibrated probabilities or an optimal operational cost rule.

### Metrics, efficiency and inference artifact

Every fixed-validation report includes Macro F1, log loss, accuracy, balanced
accuracy, precision/recall/F1/support per class, confusion matrix, and the same
metrics for each OP_UNIQUE_CARRIER and ORIGIN (15 carriers, 342 origins in the
observed run). Metadata is aligned after eligibility filtering. Batch evaluation
accumulates counts and log-loss sums before calculating final metrics; it does
not average batch F1. Classes are explicitly ordered as moderate_delay, on_time,
severe_delay in probability matrices and manifests.

| Candidate | Final fit seconds | Fixed-val predict + predict_proba seconds | Pipeline bytes |
| --- | ---: | ---: | ---: |
| Dummy prior | 0.749 | 46.023 | 3,404 |
| Logistic | 1.836 | 50.435 | 4,444 |
| Logistic balanced | 1.692 | 51.133 | 4,460 |
| LightGBM | 3.583 | 144.822 | 1,030,357 |
| LightGBM balanced | 4.433 | 144.074 | 1,030,594 |

Fit times include train-fitted preprocessing; inference times include both
prediction calls and their transformations, excluding I/O/metric assembly.
They are measurements, not reproducibility assertions about elapsed time or
serialized byte size. Seeds, input hashes, library versions, Git commit/dirty
state and source fingerprints identify each run. JSON and the single joblib
object are written atomically with explicit run states.

`pipeline.joblib` contains the native sklearn (or imbalanced-learn) inference pipeline and frozen preprocessing config. Filtering/target creation
remain outside inference. It accepts DataFrames from the existing FlightRecord /
records_to_frame request contract, including missing optional values, and
returns ordered class labels; optional encoders are inside the same object.
The actual FastAPI endpoint still returns a placeholder. Loading this object in
that endpoint belongs to Issue 2.3 and is not claimed complete by Issue 2.1.
Official Issue #4 rare-category handling remains separate as noted above.

### Native pipeline composition

`factory.create_estimator()` translates the named YAML model and resolves seed
and thread overrides. `factory.build_pipeline(estimator, config, preprocessing,
seed)` receives an already-created estimator and does not inspect its type or
name. It returns a native pipeline, with no strategy wrapper or model/feature/
strategy registries:

```text
FlightFeatures -> outliers -> engineering -> tsfresh -> ColumnTransformer
              -> selection -> sampling -> estimator
```

Disabled stages are `passthrough`. With resampling enabled the object is a native
imbalanced-learn Pipeline; otherwise it is a sklearn Pipeline. Sampling runs only
during fit. ColumnTransformer routes selected numeric, binned and categorical
columns to their applicable transformations. Native nested parameters such as
`model__C` and `columns__numeric__imputer__n_neighbors` work directly.

`create_model()` only connects YAML settings to those two functions. Every CV fit
creates fresh objects. Baseline, ablations and sequential A/B/C continue to use
the same temporal CV and scoring function. Artifact diagnostics and effective
configuration inspection live in `artifacts.py`; the Runner calls the model's
`fit`, `predict` and `predict_proba` contract.

Previously generated joblib files containing the removed `PipelineStrategy`
class require the code version that created them to load. Historical files and
metrics have not been rewritten; newly exported artifacts contain native
pipelines. This refactor does not rerun the historical BTS experiment.

### Optional Modular modes and execution

All modes reuse the same Factory, day-based CV, Evaluation and Artifact Export.
The optional `configs/modular.yaml` overlay changes only training settings and uses
a smaller 5,000-row sample for exploratory KNN/encoding configurations.

```powershell
python -m src.models.train --mode baseline
python -m src.models.train --mode ablations --experiment-config configs/modular.yaml
python -m src.models.train --mode sequential --experiment-config configs/modular.yaml --stages A B C
```

Use `--stages A`, `B` or `C` for an isolated stage. Ablations change one family
relative to an explicit reference. Sequential search uses sklearn ParameterSampler,
without Optuna: A selects preprocessing/engineering, B selection, and C model
parameters/imbalance. Only configurations propagate between stages; every fold
fits new transformers and selectors. CV selects candidates; fixed validation
evaluates the final winners. Automatic modes reject final-test evaluation.

The Modular composition supports train-fitted z-score cell treatment, Simple/KNN
imputation, Standard/MinMax/Robust scaling, column-specific KBinsDiscretizer,
OneHotEncoder with unknown categories ignored, existing OperationalFeatures,
SelectKBest/SelectPercentile, and random over/under-sampling during fit only.
Legacy RouteIQR remains for compatibility. tsfresh requires explicit per-record
history, finite observations strictly before scheduled departure, and the same
naive local clock as the schedule. No such history is present in ordinary BTS
inputs, so extraction is disabled by default and no series is fabricated.

Each run records configuration, input hashes, source fingerprint, metrics,
fold results, times, serialized size and running/completed/failed status.
`search.json` records stage candidates and failures; `summary.json` records
selection and ablation differences. The 186-test local suite and Black/flake8
checks passed at the acceptance review. Local verification does not establish
hosted CI, an issue-linked PR, another member's review or merge; those remain
pending/not verified, as does actual endpoint loading in Issue 2.3.


### Sklearn training refactor (10 November 2026)

The current implementation uses `TimeSeriesSplit` over sorted unique flight days,
then maps its indices back to flight rows. `cv_splits` must be at least two.
`sklearn.model_selection.cross_validate` clones and fits the pipeline for baseline
and ablation CV. The historical fold boundaries and scores above were produced
by the earlier `numpy.array_split` implementation and have not been rerun; new
fold boundaries can differ when the day count is not divisible by splits + 1.

Sequential A/B/C keeps only stage orchestration and YAML translation in project
code. `ParameterSampler` draws each stage's budget of configurations;
`GridSearchCV` evaluates those candidates together with the incoming reference,
using the same day-level folds, Macro F1 and negative log loss. Including the
reference preserves it if alternatives fail or lose. Only configurations pass
between stages. A fresh `cross_validate` call collects the selected candidate's
full fold/subgroup reports. Search timings are mean fit times from `cv_results_`,
not individual fold timings. Fixed validation and test never enter the search.

Classification metrics use `sklearn.metrics`. The project retains only input
checks, subgroup iteration and a bounded-memory accumulator: confusion matrices
and unnormalized log losses are summed across batches. A weighted class-pair
table lets sklearn compute precision/recall/F1 from the accumulated confusion
matrix without storing all predictions. This is not an average of batch F1s.

`factory.py` separates column resolution, numeric/binned routing, the existing
categorical branch and optional resampling into small builders. It still returns
native sklearn/imbalanced-learn pipelines. Existing encoder behavior is unchanged.
`training_frame` remains outside the inference pipeline because it filters rows
and builds y; inference transformations remain inside the serialized pipeline.
