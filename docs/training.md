# Training decisions and validation

Historical baseline evidence and current training architecture. The official
categorical encoders are now available in `src/features/encoders.py`; references
below to Issue #4 being pending describe the historical baseline run. The
controlled numeric baseline and existing modular encoder behavior are unchanged.

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
