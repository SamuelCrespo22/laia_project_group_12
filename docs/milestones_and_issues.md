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

### Issue 1.1: Ingestion Script & Raw Data Sampling
- **Labels:** `data`, `setup`
- **Description:** Implement an automated script to download and sample representative BTS flight records for 2022 and 2023 without exceeding local/VM disk limits.
- **Acceptance Criteria:**
  - Script `scripts/download_sample_data.py` downloads data files into `data/raw/`.
  - Checksums and schema verification are logged.
  - Large binary data files are excluded from git.

### Issue 1.2: Exploratory Data Analysis (EDA) & Feature Distribution Analysis
- **Labels:** `data`, `analysis`
- **Description:** Create Jupyter notebook analyzing target distribution (`on_time`, `moderate_delay`, `severe_delay`), class imbalance, missing values, and flight volume across carriers/airports.
- **Acceptance Criteria:**
  - Notebook committed under `notebooks/01_eda_and_target_analysis.ipynb`.
  - Visualizations of class proportions and carrier/origin distributions documented.

### Issue 1.3: Data Leakage Prevention & Preprocessing Pipeline
- **Labels:** `data`, `pipeline`
- **Description:** Build a reproducible preprocessing pipeline enforcing the temporal cutoff: only attributes strictly available before `CRS_DEP_TIME` may be retained.
- **Acceptance Criteria:**
  - Discard cancelled (`CANCELLED==1`) and diverted (`DIVERTED==1`) flights.
  - Filter out post-departure attributes (`DEP_DELAY`, `TAXI_OUT`, `AIR_TIME`, `ARR_TIME`).
  - Generate target column based on `ARR_DELAY`: `<15` (on_time), `15..59` (moderate_delay), `>=60` (severe_delay).
  - Unit tests in `tests/test_data_pipeline.py` pass.

### Issue 1.4: Robust Feature Transformers for Unseen Categorical Values
- **Labels:** `data`, `features`
- **Description:** Develop feature encoders (e.g., TargetEncoder, OneHot with `handle_unknown='ignore'`, frequency encoding) for `OP_CARRIER`, `ORIGIN`, and `DEST` that handle unseen categories during centralized evaluation.
- **Acceptance Criteria:**
  - Transformer handles unseen airport or airline codes without throwing runtime errors.
  - Transformers are serializable via `joblib`/`sklearn.pipeline.Pipeline`.

---

## Milestone 2: Checkpoint Release — MLflow, CI/CD & Container Serving
**Deadline:** 10 November (Graded: 5% of final mark)  
**Goal:** Deliver baseline predictive model, MLflow experiment tracking, automated CI/CD workflows, containerized FastAPI endpoint, and endpoint registration.

### Issue 2.1: Baseline Model Implementation & Training Pipeline
- **Labels:** `model`, `training`
- **Description:** Implement initial baseline classifiers (e.g., LogisticRegression/RandomForest and LightGBM) optimizing Macro F1-score and multiclass loss.
- **Acceptance Criteria:**
  - `src/models/train.py` executes training and logs performance.
  - Proper temporal train/validation split (e.g., 2022 train, 2023 validate).
  - Cross-validation results documented.

### Issue 2.2: MLflow Experiment Tracking & Artifact Logging
- **Labels:** `mlops`, `mlflow`
- **Description:** Integrate MLflow tracking in training scripts to record parameters, validation metrics (Macro F1, per-class F1, log loss), and serialize model artifacts with input signatures.
- **Acceptance Criteria:**
  - Parameters, metrics, and models logged to local/remote MLflow server.
  - Pipeline artifacts include the preprocessor + model ensemble.

### Issue 2.3: FastAPI Inference Serving with Batch `/classify` Endpoint
- **Labels:** `api`, `serving`
- **Description:** Connect the trained baseline model to `src/api/app.py`. The `/classify` endpoint must accept batch `instances` and return ordered `classifications`.
- **Acceptance Criteria:**
  - Handles batch payloads matching the essay format.
  - Pydantic schema validation returns HTTP 400 on malformed input.
  - Resilient to unseen categories.

### Issue 2.4: Docker Packaging & Local Composition
- **Labels:** `devops`, `docker`
- **Description:** Finalize the Docker image for the inference service and ensure local multi-container orchestration works seamlessly.
- **Acceptance Criteria:**
  - `docker/Dockerfile` builds cleanly with a lightweight image footprint.
  - `docker compose -f docker/docker-compose.yml up` starts API and MLflow.
  - Container passes health check (`/health`).

### Issue 2.5: GitHub Actions CI Pipeline Automation
- **Labels:** `devops`, `ci/cd`
- **Description:** Automate code linting (`flake8`, `black`), automated unit tests (`pytest`), and Docker image build on every push and pull request.
- **Acceptance Criteria:**
  - `.github/workflows/ci.yml` runs successfully on GitHub.
  - Pull requests require passing CI checks before merging.

### Issue 2.6: Registration of Inference Endpoint on Centralized Evaluation Platform
- **Labels:** `deployment`, `checkpoint`
- **Description:** Register team's endpoint on the centralized university evaluation system for initial smoke tests.
- **Acceptance Criteria:**
  - Endpoint reachable from evaluation harness.
  - Correct response contract confirmed.

---

## Milestone 3: DEI Virtual Machines Deployment & Production Serving
**Target Completion:** Late November (approx. Nov 24)  
**Goal:** Provision DEI virtual machines, set up self-hosted GitHub Actions runners, configure persistent MLflow, and deploy the live inference API.

### Issue 3.1: DEI VMs Provisioning & Network Environment Configuration
- **Labels:** `infra`, `dei-vm`
- **Description:** Request and configure VM 1 (MLflow / Artifacts) and VM 2 (FastAPI / Monitoring) via the DEI Helpdesk.
- **Acceptance Criteria:**
  - SSH access configured for all group members.
  - Docker, Docker Compose, and Python runtimes installed on both VMs.

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
  - Models promoted across stages: `None` -> `Staging` -> `Production`.
  - VM 2 fetches latest `Production` tagged model dynamically or during build.

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
- **Description:** Implement data lineage and version control using DVC, storing data snapshots and remote cache on VM 1.
- **Acceptance Criteria:**
  - DVC tracking configuration committed to the repository (`.dvc` files).
  - Dataset changes traceable across commits.

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

