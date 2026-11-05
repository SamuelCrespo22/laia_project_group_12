
# Docker Packaging — Training & Serving Images

## 1. Overview

This document describes the Docker infrastructure implemented for
Issue 2.4 — Docker Packaging: Training & Serving Images.

The objective is to provide reproducible and isolated environments
for model training, model serving, and experiment tracking.

The solution consists of:

- A Serving image for the FastAPI application.
- A Training image for machine learning experiments.
- An MLflow tracking server.
- Docker Compose orchestration.
- Persistent storage for MLflow experiments and artifacts.
- Automated configuration and integration tests.

The infrastructure can be built and tested without access to
the final processed flight dataset.

## 2. Architecture

The system contains three main services.

### 2.1. API Serving

The Serving container runs the FastAPI application using Uvicorn.

- Dockerfile: `docker/Dockerfile`
- Dependencies: `docker/requirements-serving.txt`
- Internal port: `8000`
- Health endpoint: `/health`
- Runtime user: `appuser` (UID 10001)

The image contains the API source code, configuration files,
and runtime dependencies required for the inference environment.

The container runs as a non-root user.

The `/health` endpoint was successfully tested and returned
HTTP 200 with the expected service status.

Actual model inference remains dependent on the integration
of the trained model artifact.

### 2.2. Model Training

The Training image provides an isolated Python environment
for machine learning experiments.

- Dockerfile: `docker/Dockerfile.train`
- Dependencies: `docker/requirements-training.txt`
- Runtime user: `trainer` (UID 10001)
- Training entrypoint: `python -m src.models.train`
- MLflow configuration: `MLFLOW_TRACKING_URI`

The image includes data-processing, machine-learning,
serialization, and experiment-tracking dependencies.

The training container is configured to access:

- `/app/data` for input datasets.
- `/app/models` for generated models.
- `/app/reports` for experimental reports.

Input data is mounted read-only through Docker Compose.

The image was successfully built and its non-root user
was verified.

Execution of the complete training pipeline remains pending
integration with the final training implementation.

### 2.3. MLflow Tracking Server

The MLflow server manages experiment metadata, metrics,
parameters, and artifacts.

- Container image: `ghcr.io/mlflow/mlflow:v2.10.0`
- Internal port: `5000`
- External port: `5000`
- Backend store: SQLite
- Artifact storage: MLflow-managed filesystem storage
- Persistent Docker volume: `mlflow-data`

The server uses:

`sqlite:////mlflow/mlflow.db`

for experiment metadata, and:

`/mlflow/artifacts`

for artifact storage.

Artifact serving is enabled using:

- `--serve-artifacts`
- `--artifacts-destination /mlflow/artifacts`

This configuration allows clients to upload artifacts through
the MLflow tracking server instead of accessing its filesystem
directly.

## 3. Docker Compose

The orchestration configuration is located at:

`docker/docker-compose.yml`

The default stack starts:

- `api`
- `mlflow`

The Training service belongs to the optional `training` profile.

This separation prevents the training pipeline from running
automatically whenever the API infrastructure is started.

Docker Compose provides an internal network through which
the Training container accesses MLflow using:

`http://mlflow:5000`

The name `mlflow` is resolved through Docker's internal service
discovery.

## 4. Building the Images

Run the following commands from the repository root.

### 4.1. Serving Image

```bash
docker build -f docker/Dockerfile \
  -t flight-delay-serving:dev .
```

### 4.2. Training Image

```bash
docker build -f docker/Dockerfile.train \
  -t flight-delay-training:dev .
```

### 4.3. Verify Images

```bash
docker images --filter "reference=flight-delay-*"
```

## 5. Running the Infrastructure

Start the default stack:

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

Check container status:

```bash
docker compose -f docker/docker-compose.yml ps
```

Access the services:

- API: http://localhost:8000
- API health: http://localhost:8000/health
- MLflow UI: http://localhost:5000

Stop the stack:

```bash
docker compose -f docker/docker-compose.yml down
```

The named MLflow volume is preserved by this command.

Do not use `down -v` unless deleting persistent MLflow data
is explicitly intended.

### 5.1. Training Profile

The Training service can be invoked explicitly through:

```bash
docker compose -f docker/docker-compose.yml \
  run --rm training
```

This command invokes the configured training entrypoint.

Successful end-to-end training requires the completed
training pipeline and compatible processed datasets.

## 6. Persistent Storage

The Docker volume `mlflow-data` stores the MLflow SQLite
database and experimental artifacts.

Persistence was verified in two independent tests.

### 6.1. Experiment Metadata

A test experiment named `docker_persistence_test`
was created through the MLflow REST API.

After removing and recreating the containers using
Docker Compose, the experiment remained accessible
with its original experiment ID.

Result: PASSED.

### 6.2. Artifact Storage

A text artifact named `docker_artifact_test.txt`
was uploaded from a temporary Training container
to the MLflow server.

The artifact was listed through the MLflow client API.

After removing and recreating the containers, the same
artifact was successfully downloaded and its contents
were verified.

Result: PASSED.

These tests confirm persistence across container
recreation with the named volume preserved.

## 7. Security

Both application images use dedicated non-root users.

| Image | User | UID |
|---|---|---|
| Serving | appuser | 10001 |
| Training | trainer | 10001 |

The Serving image exposes only the API port required
for the service.

The Training service mounts source datasets read-only.

A `.dockerignore` file excludes unnecessary local files
from the Docker build context, including datasets,
generated models, reports, local environments,
Git metadata, and common secret files.

The Docker infrastructure uses separate dependency
files for serving and training.

## 8. Image Sizes

Image sizes were measured using `docker image inspect`.

| Image | Size (bytes) | Approximate size |
|---|---:|---:|
| Serving | 701443107 | 701.4 MB |
| Training | 1280436619 | 1280.4 MB |

The Serving image uses `python:3.12-slim`.

The largest contributors to its installed dependencies
include SciPy, Pandas, Scikit-learn, and NumPy.

The Training image is larger because it additionally
includes packages required for dataset processing
and experiment tracking.

These sizes represent the tested local builds.
Future dependency changes may alter them.

Further optimization should be evaluated after
the final model-serving requirements are confirmed.

## 9. Testing

### 9.1. Automated Python Tests

Docker configuration tests are implemented in:

`tests/test_docker_packaging.py`

The tests verify:

- Serving Dockerfile configuration.
- Training Dockerfile configuration.
- Separation of dependencies.
- Required Docker Compose services.
- Persistent MLflow storage configuration.
- Training-to-MLflow connectivity configuration.
- Read-only dataset mounting.
- Explicit versions for third-party images.

Local results:

- Docker-specific tests: 8 passed.
- Complete project test suite: 111 passed.
- Test failures: 0.

### 9.2. Manual Integration Tests

The following operations were tested successfully:

- Building both Docker images.
- Running the API inside Docker.
- Checking `/health` through HTTP.
- Verifying the Docker health status.
- Verifying non-root container users.
- Starting API and MLflow with Docker Compose.
- Creating MLflow experiments.
- Logging parameters and metrics from Training.
- Uploading MLflow artifacts from Training.
- Recreating containers while retaining metadata.
- Downloading persisted artifacts after recreation.

### 9.3. CI/CD

A Docker-specific GitHub Actions workflow was added at:

`.github/workflows/docker-ci.yml`

The workflow is designed to validate:

- Docker Compose syntax.
- Serving and Training image builds.
- Non-root user configuration.
- API health.
- MLflow availability.
- Training-to-MLflow communication.
- Docker image sizes.

The workflow also collects container logs when
a step fails and removes test containers afterward.

Local Docker tests passed. Execution of this workflow
on GitHub-hosted runners must be confirmed separately
before reporting CI success.

## 10. Known Limitations

The following limitations remain:

1. The complete training pipeline has not yet been
   executed inside Docker because the final training
   implementation and processed data are pending.

2. The inference endpoint has not yet been validated
   against a final trained model artifact.

3. Docker image sizes may be reduced further after
   the runtime dependencies are finalized.

4. Git metadata is not automatically available inside
   the Training image because the Git executable is
   not installed. MLflow logging still works.

5. End-to-end GitHub Actions execution must be
   confirmed after pushing the branch.

6. The original Compose configuration included
   monitoring services. Their integration with
   the updated stack should be coordinated with
   the team if those services are still required.

## 11. Conclusion

Issue 2.4 provides a containerized infrastructure
for API serving, machine learning training,
and experiment tracking.

The Docker Serving environment, Docker Compose stack,
MLflow communication, and persistent storage were
validated successfully without relying on the
final processed flight dataset.

The remaining integration work concerns executing
the completed training pipeline, loading the final
model for inference, and confirming GitHub Actions
execution in the remote CI environment.
```
