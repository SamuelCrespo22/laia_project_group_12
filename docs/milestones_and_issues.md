# Project Roadmap: Milestones & GitHub Issues

This document defines the milestone schedule and actionable GitHub issues for the **Flight Delay Severity Prediction MLOps Pipeline** (LAIA 2026/2027 — Group 12).

---

## Overview of Milestones

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ MILESTONE 1: Data Pipeline, Leakage Prevention & Baseline EDA (Due: Oct 25)      │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ MILESTONE 2: Checkpoint Release — MLflow, CI/CD & Serving (Due: Nov 10) [5%]    │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ MILESTONE 3: DEI VMs Deployment & Production Serving (Due: Nov 24)               │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ MILESTONE 4: Production Monitoring, Subgroup Drift & Continuous Training (Dec 5) │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ MILESTONE 5: Final Delivery, Technical Report & Individual Defenses (Dec 11)[95%]│
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Milestone 1: Data Pipeline, Leakage Prevention & Baseline EDA
**Target Completion:** Late October (approx. Oct 25)  
**Goal:** Ingest raw BTS 2022–2023 flight data, enforce temporal integrity (no data leakage), perform Exploratory Data Analysis (EDA), and establish clean dataset splits.

> **Column naming convention (applies to all Milestone 1 issues):** the evaluation endpoint receives flights "using the same feature names as in the original BTS dataset", i.e. the uppercase names used by the TranStats download form (`FL_DATE`, `OP_UNIQUE_CARRIER`, `ORIGIN`, `CRS_DEP_TIME`, `ARR_DELAY`, ...). All code, configs and docs use these names only. If data is obtained from any other BTS source with different column names, ingestion must map them to the form names.

#### Order of work and dependencies

| Step | Issue | GitHub | Depends on |
| :---: | :--- | :---: | :--- |
| 1 | 1.1 Ingestion | #1 | — |
| 2 | 1.6 Data contract | #12 | 1.1 (column list) |
| 2 | 1.2 EDA | #2 | 1.1 |
| 3 | 1.3 Leakage & preprocessing | #3 | 1.6, 1.2 (allowlist justification) |
| 3 | 1.5 Temporal splits | #11 | 1.2 (seasonality) |
| 4 | 1.4 Categorical encoders | #4 | 1.3, 1.5 (fit on train split) |
| any | 1.7 (Bonus) DVC | #13 | 1.1; stages added as 1.3/1.5 land |

#### Definition of done (every issue)
- Work done in a dedicated branch (e.g. `feature/1.3-preprocessing`) and merged through a pull request that references the issue (`Closes #N`), reviewed by at least one other team member.
- CI passes; new code has unit tests that run on small synthetic fixtures (real data is never needed in CI).
- Parameters (paths, years, thresholds, split dates) live in `configs/config.yaml`, not hard-coded.
- Every data decision (columns kept, filters, split boundaries, encoders, thresholds) is recorded with its justification in `docs/data_decisions.md`, which feeds the "Dataset and preprocessing steps" section of the technical report and the individual defenses.

### Issue 1.1: Ingestion Script & Raw Data Storage
- **Labels:** `data`, `setup`
- **Description:** Implement an automated, idempotent script that downloads all BTS flight records for 2022 and 2023 (one file per month) into `data/raw/`, keeping only the columns the project needs. Sampling is **not** done at ingestion: full data is needed for subgroup analysis (small airlines/airports), and sampling becomes a training-time parameter in `configs/config.yaml`.
- **Acceptance Criteria:**
  - A script in `scripts/` downloads the 24 monthly files into `data/raw/`, reading the years from `configs/config.yaml` and skipping months already present.
  - Column names match the TranStats form names (see convention above); the list of kept columns (features, target, filtering and monitoring) is documented.
  - Storage format is chosen with efficiency in mind (Parquet recommended over CSV: smaller files, preserved dtypes, column-selective reads) and the choice is documented.
  - SHA-256 checksum and row count per month are logged, and each file is validated against the data contract (Issue 1.6).
  - Large data files are excluded from git.

### Issue 1.2: Exploratory Data Analysis (EDA) & Feature Distribution Analysis
- **Labels:** `data`, `analysis`
- **Description:** Create a Jupyter notebook that explores the data and **justifies** the feature, split and encoding decisions taken in Issues 1.3–1.5 (decisions must be defensible individually; "an LLM suggested it" is not an acceptable justification).
- **Acceptance Criteria:**
  - Notebook committed under `notebooks/01_eda_and_target_analysis.ipynb`.
  - Target distribution (`on_time`, `moderate_delay`, `severe_delay`) and class imbalance, overall and per month.
  - Missing values per column, and share of cancelled/diverted flights removed by filtering.
  - Temporal patterns: delay severity by month, day of week and scheduled departure hour (`DEP_TIME_BLK`).
  - Cardinality and frequency of `OP_UNIQUE_CARRIER`, `ORIGIN`, `DEST`, including rare categories and categories present in 2023 but not in 2022.
  - Comparison of 2022 vs 2023 distributions (first evidence of drift).
  - Reference statistics of the training data saved for later drift monitoring (Milestone 4).
  - Short written summary of findings and the decisions they support, copied to `docs/data_decisions.md`.

### Issue 1.3: Data Leakage Prevention & Preprocessing Pipeline
- **Labels:** `data`, `pipeline`
- **Description:** Build a reproducible preprocessing pipeline enforcing the temporal cutoff: only attributes known before the scheduled departure may reach the model. Use an **allowlist** of permitted input columns (e.g. `FL_DATE`, `MONTH`, `DAY_OF_WEEK`, `OP_UNIQUE_CARRIER`, `ORIGIN`, `DEST`, `CRS_DEP_TIME`, `DEP_TIME_BLK`, `CRS_ARR_TIME`, `ARR_TIME_BLK`, `CRS_ELAPSED_TIME`, `DISTANCE`, `DISTANCE_GROUP`; final list justified by the EDA) rather than removing columns one by one.
- **Acceptance Criteria:**
  - Discard cancelled (`CANCELLED==1`) and diverted (`DIVERTED==1`) flights, and flights without a valid `ARR_DELAY`.
  - Generate target column based on `ARR_DELAY`: `<15` (on_time), `15..59` (moderate_delay), `>=60` (severe_delay).
  - Only allowlisted columns reach the feature step. A test asserts that none of the post-departure columns can: `DEP_TIME`, `DEP_DELAY*`, `DEP_DEL15`, `TAXI_OUT`, `WHEELS_OFF`, `WHEELS_ON`, `TAXI_IN`, `ARR_TIME`, `ARR_DELAY*`, `ARR_DEL15`, `ACTUAL_ELAPSED_TIME`, `AIR_TIME`, `CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY`, `CANCELLATION_CODE`, `DIV_*`, `FIRST_DEP_TIME`, `TOTAL_ADD_GTIME`, `LONGEST_ADD_GTIME`.
  - Features are computable from a single request record (no features based on other flights, e.g. the previous flight of the same aircraft).
  - The same feature function/pipeline object is used for training and by the API (no train/serve skew).
  - The whole dataset build (raw → filtered → features) runs with a single command (e.g. `python -m src.data.make_dataset`) and produces identical output on re-runs.
  - Unit tests in `tests/test_data_pipeline.py` pass.

### Issue 1.4: Robust Feature Transformers for Unseen Categorical Values
- **Labels:** `data`, `features`
- **Description:** Develop feature encoders (e.g., TargetEncoder, OneHot with `handle_unknown='ignore'`, frequency encoding) for `OP_UNIQUE_CARRIER`, `ORIGIN`, and `DEST` that handle unseen categories during centralized evaluation.
- **Acceptance Criteria:**
  - Encoders are fitted on the training split only (target/frequency statistics never use validation or test data).
  - Rare categories are grouped (threshold justified in the EDA).
  - Transformer handles unseen airport or airline codes and missing values without runtime errors (covered by unit tests).
  - Transformers are serializable via `joblib`/`sklearn.pipeline.Pipeline`.

### Issue 1.5: Temporal Train / Validation / Test Splits
- **Labels:** `data`, `pipeline`
- **Description:** Define a fixed, reproducible temporal split that mimics the evaluation setting (models are tested on future months with drift). No random splits.
- **Acceptance Criteria:**
  - Split boundaries defined in `configs/config.yaml` (e.g. train Jan 2022–Jun 2023, validation Jul–Sep 2023, test Oct–Dec 2023), with the choice justified by the EDA (seasonality).
  - If cross-validation is used, it is time-ordered (e.g. `TimeSeriesSplit`) within the training period.
  - Splits are deterministic (no randomness, or a fixed seed) and saved to `data/processed/`.
  - Class distribution per split is reported.
  - The test split is not used for model selection or tuning.

### Issue 1.6: Data Contract & Schema Validation
- **Labels:** `data`, `pipeline`
- **Description:** Define, in one place, the schema of the raw input columns (names, dtypes, allowed ranges, required vs optional). The same contract is used to validate ingested data (Issue 1.1) and API requests (Issue 2.3).
- **Acceptance Criteria:**
  - Single schema definition in `src/` (e.g. a module or a YAML file in `configs/`) using the TranStats form column names.
  - Ingestion fails or logs clearly when a monthly file violates the contract.
  - Required vs optional request fields are explicit; inference with a missing optional field does not fail.
  - Unit tests cover valid and invalid records.

### Issue 1.7: (Bonus) DVC Initialization & Data Pipeline Stages
- **Labels:** `bonus`, `dvc`
- **Description:** Initialize DVC early so data lineage covers the whole project. Remote storage on VM 1 and CI integration are handled in Issue 5.1.
- **Acceptance Criteria:**
  - `dvc init` committed; `data/raw` tracked by DVC.
  - `dvc.yaml` defines `download → preprocess → split` stages, reproducible with `dvc repro`.

---

## Milestone 2: Checkpoint Release — MLflow, CI/CD & Container Serving
**Deadline:** 10 November (Graded: 5% of final mark)  
**Goal:** Deliver baseline predictive model, MLflow experiment tracking and model registry, automated CI/CD workflows, containerized model training and serving, a reachable deployment on DEI VM 2, and endpoint registration.

> The essay's checkpoint deliverable is: *"Baseline model, code and experiment versioning (Git, MLflow or equivalent), CI/CD automation, containerized model training and serving, and endpoint registration."* Every member presents and must show their own contribution. The CI/CD dimension is graded on automation "extending beyond the functionality provided in the baseline project", so check what the course's baseline project already provides and go beyond it. The Milestone 1 definition of done applies to every issue below.

#### Order of work and dependencies

| Step | Issue | GitHub | Depends on |
| :---: | :--- | :---: | :--- |
| now | 2.7 VM provisioning & first deployment (request the VMs immediately) | #14 | — |
| 1 | 2.1 Baseline model & training pipeline | #5 | Milestone 1 |
| 1 | 2.2 MLflow tracking & model registry | #6 | — (integrated with 2.1) |
| 2 | 2.3 FastAPI `/classify` with real model | #7 | 2.1, 2.2, 1.6 |
| 2 | 2.4 Docker images (training + serving) | #8 | 2.1, 2.3 |
| 2 | 2.5 CI/CD workflows | #9 | 2.4 |
| 3 | 2.6 Endpoint registration | #10 | 2.3, 2.4, 2.7 |
| 4 | 2.8 Checkpoint presentation & demo | #15 | all |

### Issue 2.1: Baseline Model Implementation & Training Pipeline
- **Labels:** `model`, `training`
- **Description:** Implement a reproducible training pipeline and a first set of computationally efficient baseline classifiers (e.g. LogisticRegression and LightGBM), compared against a trivial reference (majority class / `DummyClassifier`). Model selection is based on Macro F1-score, with multiclass log loss as a secondary metric.
- **Acceptance Criteria:**
  - `src/models/train.py` trains end-to-end with one command, reading all parameters (model type, hyperparameters, sample size, seed) from `configs/config.yaml`.
  - Uses the temporal splits from Issue 1.5 (no random splits); validation results (and time-ordered cross-validation, if used) documented.
  - Class imbalance handling (e.g. `class_weight`) is evaluated and the choice is justified.
  - Metrics reported overall, per class and per subgroup (`OP_UNIQUE_CARRIER`, `ORIGIN`) on the validation split.
  - Preprocessing, encoders and model are saved as **one** serializable pipeline object, the same one the API loads.
  - Training time and model size are recorded (efficiency is an explicit requirement).

### Issue 2.2: MLflow Experiment Tracking & Model Registry
- **Labels:** `mlops`, `mlflow`
- **Description:** Integrate MLflow into the training pipeline so every run is fully traceable, and register candidate models in the MLflow Model Registry from the start (a local/compose MLflow server is enough until the VM 1 server exists in Milestone 3).
- **Acceptance Criteria:**
  - Each run logs parameters, metrics (Macro F1, per-class F1, log loss, subgroup metrics), the full pipeline artifact with an input signature, the git commit hash and the data version (checksums or DVC hash).
  - Runs are grouped in a named experiment; confusion matrix and classification report logged as artifacts.
  - The best model is registered and promoted between stages using registry aliases (e.g. `staging` → `production`; aliases replace the deprecated stage API in recent MLflow versions).
  - The promotion rule (e.g. "beats current `production` on validation Macro F1") is documented.

### Issue 2.3: FastAPI Inference Serving with Batch `/classify` Endpoint
- **Labels:** `api`, `serving`
- **Description:** Replace the placeholder prediction in `src/api/app.py` with the trained pipeline. The `/classify` endpoint must accept batch `instances` and return ordered `classifications`.
- **Acceptance Criteria:**
  - Model is loaded once at startup from a configurable source (MLflow registry alias or a model artifact path); the service fails fast with a clear error if no model is available.
  - `/health` reports the loaded model name and version.
  - Pydantic schema validation, based on the data contract from Issue 1.6, returns a meaningful 4xx error on malformed input (status code chosen consistently and documented, e.g. 400 vs FastAPI's default 422).
  - Extra unknown fields in an instance are ignored; missing optional fields and unseen categories do not cause failures.
  - Inference is vectorized (one batch call, no per-row loop) and output order matches input order.
  - Integration tests run `/classify` with a small model trained on synthetic data, covering order preservation, unseen categories, missing optional fields and malformed payloads.

### Issue 2.4: Docker Packaging — Training & Serving Images
- **Labels:** `devops`, `docker`
- **Description:** Containerize both model training and model serving (the essay asks for "containerized model training and serving"), and ensure the local multi-container stack works.
- **Acceptance Criteria:**
  - Serving image (`docker/Dockerfile`) installs only runtime dependencies (separate `requirements` for serving vs development/notebooks), runs as a non-root user and passes the `/health` check.
  - Training image (e.g. `docker/Dockerfile.train`) runs the training pipeline and logs to MLflow given a data volume and `MLFLOW_TRACKING_URI`.
  - Image sizes are recorded and kept small.
  - `docker compose -f docker/docker-compose.yml up` starts API and MLflow with persistent MLflow storage; third-party images pinned to explicit versions.

### Issue 2.5: GitHub Actions CI/CD Pipeline Automation
- **Labels:** `devops`, `ci/cd`
- **Description:** Automate quality checks, tests, image builds and delivery, beyond what the course baseline project provides. Deployment to the VMs through self-hosted runners is completed in Milestone 3.
- **Acceptance Criteria:**
  - CI on every push and pull request: `flake8`, `black --check`, `isort --check`, `pytest` with coverage report, and Docker build of both images.
  - Pull requests to `main` require passing CI and one approving review (branch protection).
  - CD on merge to `main`: serving image pushed to a container registry (e.g. GHCR), tagged with the commit SHA.
  - Training workflow that can be triggered manually (`workflow_dispatch`), ready to run on a self-hosted runner in Milestone 3.

### Issue 2.6: Registration of Inference Endpoint on Centralized Evaluation Platform
- **Labels:** `deployment`, `checkpoint`
- **Description:** Register the team's endpoint (running on DEI VM 2, Issue 2.7) on the centralized evaluation platform before the checkpoint.
- **Acceptance Criteria:**
  - Endpoint reachable from the evaluation platform.
  - A smoke-test script sends a sample batch in the essay format to the deployed endpoint and validates the response contract.
  - Registered URL and model version documented.

### Issue 2.7: DEI VMs Provisioning & First Deployment
- **Labels:** `infra`, `dei-vm`
- **Description:** Request and configure VM 1 (MLflow / artifacts) and VM 2 (FastAPI / monitoring) via the DEI Helpdesk. Endpoint registration at the checkpoint needs a reachable deployment, and the VMs sit inside DEI's network, so this cannot wait for Milestone 3. The first deployment to VM 2 may be manual; it is automated in Milestone 3.
- **Acceptance Criteria:**
  - VMs requested through the DEI Helpdesk as early as possible.
  - SSH access configured for all group members.
  - Docker, Docker Compose, and Python runtimes installed on both VMs.
  - Serving container running on VM 2 and reachable on the URL used for registration; deployment steps documented.

### Issue 2.8: Checkpoint Presentation & Demo
- **Labels:** `management`, `checkpoint`
- **Description:** Prepare the checkpoint presentation. All members must participate, answer questions and show their individual contributions (a member without demonstrable contribution gets zero).
- **Acceptance Criteria:**
  - Short slide deck covering data pipeline, baseline model, MLflow runs/registry, CI/CD runs and the live endpoint.
  - Live (or recorded fallback) demo: training run logged in MLflow → image built by CI → `/classify` answering on VM 2.
  - Each member presents the issues they owned, backed by their commits and pull requests.

---

## Milestone 3: DEI Virtual Machines Deployment & Production Serving
**Target Completion:** Late November (approx. Nov 24)  
**Goal:** Set up self-hosted GitHub Actions runners, configure persistent MLflow, and automate deployment of the live inference API.

> Issue 3.1 (VM provisioning) moved to Issue 2.7: endpoint registration at the checkpoint requires a reachable deployment.

### Issue 3.2: Self-Hosted GitHub Actions Runners Setup
- **Labels:** `devops`, `ci/cd`
- **Description:** Configure self-hosted GitHub Actions runners on the DEI VMs to allow automated CI/CD within DEI's private network.
- **Acceptance Criteria:**
  - Runner registered in the GitHub repository settings.
  - CI/CD workflow deploys new Docker images to VM 2 upon merge to `main`.

### Issue 3.3: Production MLflow Server & Model Registry Promotion
- **Labels:** `mlops`, `mlflow`
- **Description:** Configure a persistent MLflow tracking server on VM 1 with SQLite/PostgreSQL backend and model registry.
- **Acceptance Criteria:**
  - MLflow server runs as a managed service on VM 1.
  - Model promotion (`staging` → `production` aliases, Issue 2.2) runs against the VM 1 registry.
  - VM 2 deploys the model currently aliased `production`, either at container start or during the image build.

---

## Milestone 4: Monitoring, Drift Detection & Production Rounds
**Target Completion:** Early December (approx. Dec 5)  
**Goal:** Implement real-time system metrics, statistical drift detection, subgroup analysis, and automated alerting under delayed-label conditions.

### Issue 4.1: Prometheus & Grafana System Monitoring
- **Labels:** `monitoring`, `infra`
- **Description:** Instrument the FastAPI application with `prometheus-fastapi-instrumentator` and configure Grafana dashboards on VM 2.
- **Acceptance Criteria:**
  - Requests/sec, latency quantiles (p50, p95, p99), and error rates scraped by Prometheus.
  - Dashboard visualizes API activity in Grafana.

### Issue 4.2: Input Feature Drift Detection
- **Labels:** `monitoring`, `drift`
- **Description:** Implement statistical drift detection (e.g., using Evidently or SciPy tests like Kolmogorov-Smirnov and Chi-Square) to detect shifts in flight features over sequential evaluation rounds.
- **Acceptance Criteria:**
  - Drift detector flags significant distribution changes in scheduled departure times, distances, or top airline frequencies.
  - Generates periodic drift reports.

### Issue 4.3: Delayed-Label Evaluation & Subgroup Performance Degradation Tracking
- **Labels:** `monitoring`, `evaluation`
- **Description:** Build evaluation routines for when delayed ground-truth labels arrive. Compute performance breakdowns by operating airline and origin airport.
- **Acceptance Criteria:**
  - Calculates Macro F1-score and confusion matrices per evaluation round.
  - Detects performance degradation isolated to specific airlines or hubs even when global metrics remain steady.

### Issue 4.4: Alerting Strategies Definition & Demonstration
- **Labels:** `monitoring`, `alerts`
- **Description:** Define alert thresholds for latency spikes, high error rates, significant feature drift, and sharp performance drops.
- **Acceptance Criteria:**
  - Alert rules configured in Prometheus/Alertmanager or logged as alert events.
  - Simulation demonstrated in the technical report.

---

## Milestone 5: Final Submission, Bonus Features & Defense Preparation
**Deadline:** 11 December (Graded: 95% of final mark)  
**Goal:** Deliver complete end-to-end operational pipeline, bonus implementations, comprehensive technical report, and peer assessments.

### Issue 5.1: (Bonus) Data Versioning with DVC Integration
- **Labels:** `bonus`, `dvc`
- **Description:** Building on Issue 1.7, configure the DVC remote cache on VM 1 and integrate DVC into the CI/CD workflow.
- **Acceptance Criteria:**
  - DVC remote on VM 1 configured; `dvc pull` works for all group members and in CI.
  - Dataset changes traceable across commits, including the evaluation rounds' data.

### Issue 5.2: (Bonus) Automated Continuous Training (CT) Pipeline
- **Labels:** `bonus`, `continuous-training`
- **Description:** Implement an automated retraining pipeline triggered when significant drift or performance degradation is detected.
- **Acceptance Criteria:**
  - Automated script retrains model on newly labeled data rounds.
  - Candidate model evaluated against current production model before automated promotion.

### Issue 5.3: Technical Report & MLOps Documentation
- **Labels:** `documentation`, `report`
- **Description:** Author the comprehensive technical report covering dataset preprocessing, modeling strategy, MLOps pipeline architecture, monitoring results, and lessons learned.
- **Acceptance Criteria:**
  - Report submitted in PDF format covering all required essay sections.
  - Architecture diagrams and experiment tracking graphs embedded.

### Issue 5.4: Self- and Peer-Assessment Reports & Individual Defense Preparation
- **Labels:** `management`, `assessment`
- **Description:** Complete team collaboration assessment and prepare individual technical defense arguments.
- **Acceptance Criteria:**
  - Individual peer-assessment completed.
  - All group members demonstrate mastery of the codebase and technical decisions.

