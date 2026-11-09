# End-to-End MLOps Pipeline for Flight Delay Severity Prediction

**Advanced Artificial Intelligence Laboratory (LAIA) - 2026/2027**  
*Department of Informatics Engineering (DEI) — University of Coimbra*  
**Team:** Group 12

---

## 1. Project Overview

This project implements an end-to-end **Machine Learning Operations (MLOps)** pipeline designed to predict the severity of domestic flight arrival delays across the United States. It utilizes historical flight records from the **US Bureau of Transportation Statistics (BTS)** Airline On-Time Performance Data from **2022** and **2023**.

The primary objective is to build a reliable, reproducible, and fully automated production machine learning system following industry MLOps best practices:
- **Reproducible data pipelines** and feature engineering.
- **Strict prevention of data leakage** (only attributes available before departure).
- **Experiment tracking and model registry** with stage promotion via MLflow.
- **Containerized inference serving** via FastAPI and Docker.
- **Automated CI/CD workflows** via GitHub Actions.
- **Production monitoring and drift detection** across operational subgroups (airlines, origin airports) under delayed-label conditions.
- **Bonus components:** Data versioning with DVC and automated Continuous Training (CT).

---

## 2. Problem Specification & Target Classes

The model performs multi-class classification, categorizing each flight into one of three arrival delay severity categories based on its recorded arrival delay (`ARR_DELAY`):

| Category | Definition | Threshold |
| :--- | :--- | :--- |
| **`on_time`** | Arrival delay strictly below 15 minutes | `ARR_DELAY < 15` |
| **`moderate_delay`** | Arrival delay of at least 15 but less than 60 minutes | `15 <= ARR_DELAY < 60` |
| **`severe_delay`** | Arrival delay of at least 60 minutes | `ARR_DELAY >= 60` |

### Critical Constraints & Engineering Rules
1. **No Data Leakage:** Only features known **prior to the scheduled departure time** (`CRS_DEP_TIME`) may be used as model inputs (e.g., scheduled times, planned elapsed time, distance, carrier, origin, destination). Real-time attributes generated post-departure (such as actual departure delay, taxi-out time, airborne time, or actual arrival time) are strictly prohibited during inference.
2. **Filtering Clean Flights:** Only domestic flights that were **neither cancelled nor diverted** and possess a valid recorded arrival delay must be included in training and validation datasets.
3. **Handling Unseen Categorical Values:** The inference pipeline must gracefully handle previously unseen categorical values (such as new carrier codes or regional airport identifiers) without raising unhandled runtime errors.
4. **Order Preservation:** The inference endpoint must return predictions matching the exact sequential order of the input instances.

---

## 3. System Architecture & Infrastructure

To simulate real-world production environments, the system leverages GitHub for version control and CI/CD, alongside two dedicated virtual machines (VMs) hosted within DEI's internal infrastructure:

```
[Local Development] 
         │ (git push / PR)
         ▼
[GitHub Repository] ─────────► [GitHub Actions CI/CD]
                                         │
                        ┌────────────────┴────────────────┐
                        ▼                                 ▼
             [DEI Virtual Machine 1]           [DEI Virtual Machine 2]
             ├── MLflow Tracking Server        ├── Docker Engine Host
             ├── MLflow Model Registry         ├── FastAPI Inference Service (:8000)
             └── DVC Remote Storage            ├── Prometheus Monitoring (:9090)
                                               └── Grafana Dashboards (:3000)
```

- **DEI VM 1 (Experiment & Artifact Store):** Hosts the centralized MLflow server, artifact store, and optional DVC remote cache.
- **DEI VM 2 (Production Serving & Monitoring):** Runs the containerized FastAPI prediction endpoint and the monitoring stack (Prometheus / Grafana).
- **Self-Hosted GitHub Actions Runners:** Configured on the DEI VMs to allow automated deployments and training pipelines behind the institutional firewall.

---

## 4. Repository Structure

```text
laia_project_group_12/
├── .github/
│   └── workflows/
│       ├── ci.yml               # CI: flake8 + pytest on pushes and pull requests
│       └── docker-ci.yml        # Builds both images and tests the Compose stack
├── configs/
│   └── config.yaml              # Project parameters (target, ingestion, preprocessing, MLflow, API)
├── data/
│   ├── raw/                     # Monthly BTS Parquet files (Git LFS) + manifest.json
│   ├── interim/                 # Intermediate datasets (git-ignored)
│   └── processed/               # Model-ready datasets and splits (git-ignored)
├── docker/
│   ├── Dockerfile               # Serving image (FastAPI inference service)
│   ├── Dockerfile.train         # Training image (logs runs to MLflow)
│   ├── requirements-serving.txt # Runtime dependencies of the serving image
│   ├── requirements-training.txt # Dependencies of the training image
│   └── docker-compose.yml       # Local stack (API, MLflow, optional training profile)
├── docs/
│   ├── architecture.md          # Architecture documentation
│   ├── data_decisions.md        # Data decisions and their justifications
│   ├── docker_packaging.md      # Docker images, Compose stack and validation
│   └── milestones_and_issues.md # Milestones and GitHub issues roadmap
├── models/                      # Local model artifacts (git-ignored)
├── scripts/
│   └── ingest_bts.py            # Incremental BTS download, validation and Parquet storage
├── src/
│   ├── api/                     # Inference service
│   │   ├── app.py               # FastAPI application (/health and /classify)
│   │   └── schemas.py           # Pydantic request and response schemas
│   ├── data/
│   │   ├── contract.py          # Data contract: columns, types and value rules
│   │   ├── preprocessor.py      # Leakage-safe filtering, target and features (train + serve)
│   │   └── make_dataset.py      # raw -> filtered -> features: python -m src.data.make_dataset
│   ├── features/                # Encoders for rare, unseen and missing values
│   ├── models/                  # Training and evaluation (planned, Issue 2.1)
│   ├── monitoring/              # Drift and subgroup monitoring (planned, Milestone 4)
│   └── utils/                   # Shared helpers
├── tests/                       # pytest suite (synthetic fixtures, no real data needed)
├── .gitattributes               # Git LFS tracking rules for data/raw/*.parquet
├── .gitignore
├── essay.pdf                    # Official project specification
├── README.md
└── requirements.txt             # Project dependencies
```

---

## 5. Inference API Specification

The inference service provides a batch classification endpoint adhering to the centralized evaluation requirements.

### Endpoint: `POST /classify`

#### Request Body
Accepts a JSON payload containing an `instances` array of flight records with raw attributes available prior to departure:

```json
{
  "instances": [
    {
      "FL_DATE": "2025-05-15",
      "OP_UNIQUE_CARRIER": "AA",
      "ORIGIN": "JFK",
      "DEST": "LAX",
      "CRS_DEP_TIME": 900,
      "CRS_ARR_TIME": 1215,
      "CRS_ELAPSED_TIME": 375,
      "DISTANCE": 2475
    },
    {
      "FL_DATE": "2025-05-16",
      "OP_UNIQUE_CARRIER": "DL",
      "ORIGIN": "ATL",
      "DEST": "ORD",
      "CRS_DEP_TIME": 1400,
      "CRS_ARR_TIME": 1530,
      "CRS_ELAPSED_TIME": 150,
      "DISTANCE": 606
    }
  ]
}
```

#### Response (`200 OK`)
Returns a JSON object with a `classifications` array containing the predicted category for each flight, preserving the original ordering:

```json
{
  "classifications": [
    "moderate_delay",
    "on_time"
  ]
}
```

#### Status Codes:
- `200 OK`: Successful inference.
- `400 Bad Request`: Empty instances array, malformed JSON, or missing critical fields.
- `500 Internal Server Error`: Server-side inference pipeline error.

---

## 6. Setup and Development Guide

### 6.1. Prerequisites
- **Python 3.12** installed on your system.
- **Git** configured.
- **Docker** & **Docker Compose** installed (for containerized execution).

### 6.2. Activate the Virtual Environment
The virtual environment is already provisioned in the repository root at `venv/`.

**On macOS / Linux:**
```bash
source venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

### 6.3. Install Dependencies
Ensure `pip` is up to date and install all required libraries:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 6.4. Run the Test Suite
Execute automated unit tests using `pytest`:
```bash
pytest tests/ -v
```

To build the model dataset from the raw files (after `git lfs pull`), run
`python -m src.data.make_dataset`. It writes `data/processed/dataset.parquet`
and a manifest; re-runs produce identical files.

### 6.5. Launch the Local Development Server
Start the FastAPI server with hot-reloading:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be available at:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

### 6.6. Run the Full Stack with Docker Compose
To launch the local stack (API container and MLflow server):
```bash
docker compose -f docker/docker-compose.yml up --build
```
Prometheus and Grafana are not part of this stack yet. They return in Issue 4.1 (Milestone 4).

---

## 7. MLOps Lifecycle & Engineering Practices

### 7.1. Experiment Tracking & Model Registry (MLflow)
- Every training run logs:
  - Hyperparameters and preprocessing settings.
  - Evaluation metrics: Macro F1-score, multiclass log-loss, subgroup performance (breakdowns by airline and origin airport).
  - Serialized model artifacts (pipeline including encoders and classifiers).
- Candidate models are registered in the MLflow Model Registry and transitioned from `None` -> `Staging` -> `Production`.

### 7.2. Continuous Integration & Continuous Deployment (CI/CD)
- GitHub Actions (`.github/workflows/ci.yml`) runs on pushes and pull requests to `main`, `master` and `develop`:
  - `flake8` fails the build on syntax errors and undefined names (`E9,F63,F7,F82`); other style issues are reported as warnings only (`--exit-zero`).
  - `pytest tests/` runs the unit tests (ingestion, data contract, preprocessing and API) on synthetic fixtures.
- GitHub Actions (`.github/workflows/docker-ci.yml`) builds both images and tests the Compose stack when Docker files, `src/` or `configs/` change.
- Publishing images and deployment are not automated yet (Issue 2.5).

### 7.3. Monitoring & Drift Detection (planned, Milestone 4)
- Real-time prediction latency, throughput, and error rates exported to Prometheus.
- Delayed-label performance tracking: When evaluation ground-truth becomes available in sequential rounds, performance drift and statistical data drift (using Evidently / SciPy Kolmogorov-Smirnov & Chi-Square tests) are computed.
- Subgroup performance monitoring isolates performance dips targeting specific airlines or airports.

---

## 8. Milestones and Evaluation

For the complete list of GitHub issues and task assignments across milestones, refer to the Milestones and Issues Roadmap in `docs/milestones_and_issues.md`.

| Milestone | Due Date | Core Deliverables |
| :--- | :---: | :--- |
| **Checkpoint** | **10 November** | Baseline model, versioning (Git & MLflow), CI/CD, containerized serving, endpoint registration. |
| **Final Submission** | **11 December** | Complete end-to-end pipeline, VM deployment, production monitoring, drift detection, and technical report. |

### Grading Dimensions
- **Model Training Process (20%):** Feature engineering, leakage prevention, and validation strategy.
- **Code Versioning & Reproducibility (20%):** Disciplined Git workflows, clear commit history, and clean code.
- **Experiment & Model Versioning (20%):** MLflow traceability, artifact logging, and model registry lifecycle.
- **CI/CD Automation (20%):** Automated testing, linting, packaging, and deployment workflows.
- **Deployment & Monitoring (20%):** Containerized live API, Prometheus/Grafana dashboards, and drift detection.
- **Bonus Credit (+10%):** Continuous Training (CT) triggered by drift and DVC data versioning.
