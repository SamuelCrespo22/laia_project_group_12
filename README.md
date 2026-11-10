# MLOps Pipeline for Flight Delay Severity Prediction

LAIA 2026/2027, DEI, University of Coimbra. Group 12.

Predicts the arrival delay severity of US domestic flights from BTS On-Time
Performance data (2022 and 2023), using only information known before the
scheduled departure. The official specification is [essay.pdf](essay.pdf).

| Class | Definition |
| :--- | :--- |
| `on_time` | `ARR_DELAY < 15` |
| `moderate_delay` | `15 <= ARR_DELAY < 60` |
| `severe_delay` | `ARR_DELAY >= 60` |

## Status

Working: ingestion, data contract, preprocessing, temporal splits, encoders,
baseline training, MLflow tracking and registry, Docker images, CI and image
publishing to GHCR.

Not implemented yet: real inference in `/classify` (#7, it currently always
returns `on_time`), VM deployment (#14) and endpoint registration (#10).
Monitoring is planned for Milestone 4.

## Documentation

| File | Content |
| :--- | :--- |
| [docs/architecture.md](docs/architecture.md) | Components and their state |
| [docs/data_decisions.md](docs/data_decisions.md) | Data decisions and justifications |
| [docs/docker_packaging.md](docs/docker_packaging.md) | Images and Compose stack |
| [docs/milestones_and_issues.md](docs/milestones_and_issues.md) | Roadmap |

## Repository

```text
.github/workflows/  CI, Docker CI, CD and manual training
configs/config.yaml All project parameters
data/               raw (Git LFS), processed (ignored), reference statistics
docker/             Serving and training images, Compose stack
notebooks/          EDA
scripts/            BTS ingestion, temporal splits
src/api/            FastAPI service
src/data/           Data contract, preprocessing, dataset build, splits
src/features/       Encoders for rare, unseen and missing values
src/analysis/       EDA helpers
src/models/         Training (#5, empty)
src/monitoring/     Monitoring (Milestone 4, empty)
tests/              pytest suite on synthetic fixtures
```

## Setup

Requires Python 3.12, Git LFS and Docker. Run every command from the repository
root.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
git lfs install && git lfs pull
```

## Commands

```bash
python -m scripts.ingest_bts                # Download BTS data into data/raw
python -m src.data.make_dataset             # Build data/processed/dataset.parquet
python -m scripts.create_temporal_splits    # Publish the train/validation/test files
pytest tests/                               # Run the tests
flake8 . && black --check . && isort --check-only .
uvicorn src.api.app:app --reload            # API on http://localhost:8000/docs
docker compose -f docker/docker-compose.yml up --build   # API and MLflow
```

## API

`POST /classify` receives flights with the BTS column names and returns one
class per flight, in input order. Required fields are `FL_DATE`,
`OP_UNIQUE_CARRIER`, `ORIGIN`, `DEST` and `CRS_DEP_TIME`. Unknown fields are
ignored.

```json
{"instances": [{"FL_DATE": "2025-05-15", "OP_UNIQUE_CARRIER": "AA", "ORIGIN": "JFK", "DEST": "LAX", "CRS_DEP_TIME": 900}]}
```

```json
{"classifications": ["on_time"]}
```

| Status | Meaning |
| :--- | :--- |
| `200` | Predictions returned |
| `400` | Empty `instances` |
| `422` | Malformed JSON or a record that violates the data contract |
| `500` | Inference failure |

`GET /health` reports the service status.

## CI/CD

| Workflow | Trigger | What it does |
| :--- | :--- | :--- |
| `ci.yml` | Every push and pull request | `flake8`, `black`, `isort`, `pytest` with coverage |
| `docker-ci.yml` | Docker, `src/` or `configs/` changes on `main` and pull requests | Builds both images and tests the Compose stack |
| `cd.yml` | Push to `main` | Publishes the serving image to GHCR, tagged with the commit SHA |
| `training.yml` | Manual | Runs training on a self-hosted runner. No runner exists yet (Milestone 3) |

Merging into `main` requires the `lint-and-test` check to pass.
