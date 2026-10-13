# System Architecture & Technical Design

This document details the architectural decisions, pipeline workflows, and infrastructure setup for the **Flight Delay Severity Prediction MLOps Platform** (Group 12).

---

## 1. High-Level Architecture

```
                  ┌───────────────────────────────────────────────┐
                  │                 GitHub Repo                   │
                  │   - Main codebase & configuration             │
                  │   - CI/CD Workflows (.github/workflows)       │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │      GitHub Actions (CI/CD)     │
                         │   1. Linting & Formatting       │
                         │   2. Automated Pytest Suites    │
                         │   3. Docker Image Build & Push  │
                         └────────────────┬────────────────┘
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                             ▼
     ┌───────────────────────────┐                 ┌───────────────────────────┐
     │       DEI VM 1            │                 │       DEI VM 2            │
     │  (Experiment & Registry)  │                 │   (Serving & Monitoring)  │
     │                           │                 │                           │
     │  ┌─────────────────────┐  │                 │  ┌─────────────────────┐  │
     │  │   MLflow Server     │  │◄────────────────┼──┤ FastAPI Container   │  │
     │  │  - Experiment Logs  │  │ (fetch models)  │  │  - /classify        │  │
     │  │  - Model Registry   │  │                 │  │  - /health          │  │
     │  └─────────────────────┘  │                 │  └──────────┬──────────┘  │
     │                           │                 │             │ metrics     │
     │  ┌─────────────────────┐  │                 │  ┌──────────▼──────────┐  │
     │  │    DVC Storage      │  │                 │  │ Prometheus Scraper  │  │
     │  │  - Versioned Data   │  │                 │  └──────────┬──────────┘  │
     │  └─────────────────────┘  │                 │             │             │
     │                           │                 │  ┌──────────▼──────────┐  │
     │                           │                 │  │  Grafana Dashboard  │  │
     │                           │                 │  └─────────────────────┘  │
     └───────────────────────────┘                 └───────────────────────────┘
```

---

## 2. Key Components

### 2.1. Data Ingestion & Leakage Prevention
- **Historical Data:** BTS Airline On-Time Performance Data for years 2022 and 2023.
- **Leakage Prevention Strategy:**
  - Strict temporal cut-off: All input features must correspond strictly to operational attributes known prior to scheduled departure time (`CRS_DEP_TIME`).
  - Prohibited training/inference attributes include: `DEP_DELAY`, `TAXI_OUT`, `AIR_TIME`, `ARR_TIME`, `ACTUAL_ELAPSED_TIME`.
- **Target Derivation:**
  - `ARR_DELAY < 15` $\rightarrow$ `on_time`
  - `15 <= ARR_DELAY < 60` $\rightarrow$ `moderate_delay`
  - `ARR_DELAY >= 60` $\rightarrow$ `severe_delay`
- **Filtering:** Flights where `CANCELLED == 1` or `DIVERTED == 1` or missing `ARR_DELAY` are excluded.

### 2.2. Model Training & MLflow Tracking
- Models are trained reproducibly with fixed random seeds.
- Experiment tracking logs:
  - Dataset hash/version.
  - Hyperparameters (e.g., max depth, learning rate, estimator count).
  - Validation metrics (Macro F1, Multiclass Log Loss, per-subgroup F1 scores).
  - Model artifacts packaged in MLflow format with input signatures.
- Registered models follow stage transitions:
  `Candidate` $\rightarrow$ `Staging` (validated against benchmark) $\rightarrow$ `Production` (deployed to VM2).

### 2.3. Inference Service (FastAPI)
- Exposes `POST /classify` for batch prediction:
  - Input: `{ "instances": [ { ...flight attributes... } ] }`
  - Output: `{ "classifications": [ "moderate_delay", "on_time", ... ] }`
- **Robustness:** Handles unseen categorical values (unfamiliar carrier codes or airports) through unknown-category imputation or fallback encodings.

### 2.4. Production Monitoring & Drift Detection
- **System Metrics:** Request rate, latency distribution (p50, p95, p99), HTTP status codes collected via Prometheus.
- **Data & Concept Drift:**
  - Evaluated under delayed-label conditions across evaluation rounds.
  - Subgroup performance monitoring isolates degradation affecting specific operating carriers or hub airports.
  - Statistical testing using SciPy / Evidently (Kolmogorov-Smirnov for numerical features, Chi-Square for categorical distributions).

