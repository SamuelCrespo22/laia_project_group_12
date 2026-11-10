# Data decisions

The reasoning behind the data pipeline. Parameters are in `configs/config.yaml`
and column rules are in `src/data/contract.py`.

## Ingestion (#1)

- **Source:** monthly BTS PREZIP archives for 2022 and 2023. Headings are mapped
  to the TranStats names that the evaluation endpoint receives, and `FL_DATE` is
  derived from the calendar columns.
- **No sampling or filtering:** small airline and airport subgroups must survive.
  Sampling is a training parameter, and filtering belongs to preprocessing.
- **Storage:** one Zstandard Parquet per month, because it keeps types and allows
  reading selected columns. Files are in Git LFS.
- **Integrity:** `data/raw/manifest.json` stores each month's SHA-256 and row
  count. Valid months are reused, and missing or corrupt ones are rebuilt.
- **Kept columns:** `COLUMN_MAP` in the contract is the list. Keeping a raw column
  does not approve it as a feature.
- **Anomalies:** invalid flags, negative distances, invalid HHMM times and
  inconsistent dates fail the month. The six negative `CRS_ELAPSED_TIME` values
  in the source are logged and kept.

## Data contract (#12)

`COLUMNS` is the single definition of every column. Ingestion, the API schema and
the leakage checks derive from it, so training and serving agree on what a valid
flight is. It is a Python module because Pydantic models are generated from it.

Required request fields are `FL_DATE`, `OP_UNIQUE_CARRIER`, `ORIGIN`, `DEST` and
`CRS_DEP_TIME`. They identify a scheduled flight and include the monitoring
subgroups. Requiring more would risk rejecting evaluation records.

| Situation | Result |
| :--- | :--- |
| Required field missing, null, wrong type or out of range | Rejected, naming the instance and field |
| Optional field missing | Null |
| Optional field of the wrong type | Rejected |
| Optional field out of range | Null and a logged warning |
| Unknown or post-departure field | Ignored |
| Unseen airline or airport | Accepted, handled by the encoders |

A wrong type signals a malformed client. An out-of-range optional value loses only
that value, so it should not discard the batch. Non-finite numbers count as missing.
Every rejected request returns 400, so clients handle one code for invalid input.

`TAIL_NUM` counts as post-departure because aircraft swaps happen up to departure.
`CANCELLED` and `DIVERTED` are outcomes, used only for filtering.

## Preprocessing (#3)

Re-runs on the same raw data produce byte-identical output.

| Step (2022-2023) | Rows |
| :--- | ---: |
| Raw flights | 13,577,024 |
| Cancelled | -269,229 |
| Diverted | -32,378 |
| Missing `ARR_DELAY` | -2 |
| Model dataset | 13,275,415 |

Cancelled and diverted flights have no arrival delay to predict. Class imbalance
is handled at training time, not by resampling.

An allowlist decides which columns reach the features, so a new raw column cannot
leak by default. Configuration loading fails if a listed column is not
pre-departure in the contract.

| Kept column | Reason |
| :--- | :--- |
| `FL_DATE` | Used for splits and calendar features |
| `MONTH`, `DAY_OF_WEEK` | Seasonal and weekly patterns |
| `OP_UNIQUE_CARRIER`, `ORIGIN`, `DEST` | Airline and airport effects; monitoring subgroups |
| `CRS_DEP_TIME`, `DEP_TIME_BLK` | Delays accumulate during the day |
| `CRS_ARR_TIME`, `ARR_TIME_BLK` | Scheduled arrival period |
| `CRS_ELAPSED_TIME`, `DISTANCE`, `DISTANCE_GROUP` | Duration and route length |

Not kept: `YEAR` (future years are unseen), `QUARTER` and `DAY_OF_MONTH`
(redundant), numeric airport identifiers and city or state names (redundant with
airport codes), `OP_CARRIER_FL_NUM` (high cardinality), `FLIGHTS` (always 1).

`build_features` is the only feature function and uses one record at a time.
`MONTH` and `DAY_OF_WEEK` come from `FL_DATE`, time blocks from the scheduled
times, and `DISTANCE_GROUP` from `DISTANCE`. On the 2022-2023 data, the derived
calendar and distance values match the BTS values.

## EDA findings (#2)

- **Target:** 79.18% `on_time`, 13.89% `moderate_delay`, 6.93% `severe_delay`. The
  thresholds come from the essay.
- **Missing values:** `ARR_DELAY` is absent for all cancelled and diverted flights
  and for only two others, which supports excluding rather than imputing.
- **Time:** delay rates vary by hour, weekday and month, so calendar and time
  features are kept. The share of severe delays was 11.39% in July 2023 and 3.55%
  in November 2023. `CRS_DEP_TIME` is local HHMM, not a continuous quantity.
- **Categories:** 17 airlines, 373 airports and 7,163 routes. No airline or airport
  first appears in 2023, but the route mix changes.
- **Drift 2022 to 2023:** median distance rises from 666 to 679 miles, and median
  scheduled time from 126 to 129 minutes. `WN` grows from 19.43% to 21.01% of
  flights. This supports chronological evaluation.
- **Arrival traffic:** the delay rate is 20.26% below 0.75x usual destination volume
  and 23.19% at 1.5x or more. It is not a feature: the calculation approximates
  arrival dates and uses both years as its reference, and it does not show causality.
- **Scheduled duration:** no overall year-on-year change in the observed duration
  buffer (mean -0.05 minutes), so `CRS_ELAPSED_TIME` and `DISTANCE` are kept as they are.

## Temporal splits (#11)

| Split | Period | All flights | Eligible | `severe_delay` |
| :--- | :--- | ---: | ---: | ---: |
| Train | 2022-01 to 2023-06 | 10,069,694 | 9,811,971 | 7.15% |
| Validation | 2023-07 to 2023-09 | 1,774,191 | 1,738,087 | 8.47% |
| Test | 2023-10 to 2023-12 | 1,733,139 | 1,725,357 | 4.12% |

Random splits would mix past and future, unlike the evaluation on later months.
Periods are contiguous and flights outside them are rejected. Validation covers the
high-delay summer and test the final quarter, so their metrics are not directly
comparable. Two years cannot establish a stable seasonal pattern.

Train is for fitting and validation for model selection. Test is only for the final
evaluation. Any cross-validation must be time-ordered within the training period.

`data/processed/temporal_split_manifest.json` identifies the published files. They
keep the raw columns, including cancelled flights and post-departure columns, so
they must not go directly to a model. `temporal_split` applied to `dataset.parquet`
gives the same periods on eligible flights and allowlisted columns only.

## Encoders (#4)

`build_feature_encoder` returns an unfitted transformer, to be fitted on the
training split only and saved with the classifier. It uses only scikit-learn
classes, so the saved object does not depend on project code.

| Model type | Categorical | Numeric |
| :--- | :--- | :--- |
| `linear` | One-hot, sparse | Median imputation and scaling |
| `tree` | Ordinal codes | Median imputation |

Categorical columns are `OP_UNIQUE_CARRIER`, `ORIGIN`, `DEST`, `DEP_TIME_BLK` and
`ARR_TIME_BLK`. Numeric columns are `CRS_ELAPSED_TIME`, `DISTANCE`, `MONTH`,
`DAY_OF_WEEK` and `DISTANCE_GROUP`. `FL_DATE` is excluded because evaluation uses
future years, and HHMM times are excluded because the blocks carry that information.
Target encoding is not used because it needs cross-fitting to avoid leakage.

`min_frequency` is a fraction of the fitted rows, so it keeps its meaning under
sampling. On the full training split:

| Threshold | Airports grouped (origin) | Flights affected |
| :--- | ---: | ---: |
| `0.0001` (used) | 79 of 372 | 0.44% |
| `0.001` (EDA rarity share) | 256 of 372 | 6.66% |

`0.001` would merge airports with thousands of flights. No airline is grouped at
either value.

| Input | `linear` | `tree` |
| :--- | :--- | :--- |
| Rare category | Grouped column | Shared code |
| Unseen or null category | Grouped column, or all zeros if the column has no rare group | `-1` |
| Null number | Training median | Training median |

## Known gaps

- The reference statistics in `data/reference/` cover 2022 only (6,532,012 eligible
  flights), while training runs to June 2023. The `eda` block in the config sets
  that period.
- The notebook's export cell writes a different format and includes cancelled
  flights, so re-running it would replace the current reference file.
- The reference file covers fewer columns than `eda.reference_features`.
- The `0.0001` threshold has not been compared with `0.001` on model performance.

## Training

See [training decisions and validation](training.md) for the controlled baseline,
temporal CV, experiment search, historical metrics and sklearn refactor.
