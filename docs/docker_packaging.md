# Docker packaging

| Service | Definition | Runs as | Purpose |
| :--- | :--- | :--- | :--- |
| `api` | `docker/Dockerfile` | `appuser` (UID 10001) | FastAPI on port 8000 |
| `mlflow` | `ghcr.io/mlflow/mlflow:v2.10.0` | Image default | Tracking server on port 5000 |
| `training` | `docker/Dockerfile.train` | `trainer` (UID 10001) | Runs `python -m src.models.train` |

Serving and training have separate dependency files, so the serving image has
no MLflow client or data tooling. `.dockerignore` keeps data, models, secrets
and Git metadata out of the images.

Image sizes in CI: serving 497 MB, training 924 MB.

## Usage

```bash
docker compose -f docker/docker-compose.yml up -d --build   # API and MLflow
docker compose -f docker/docker-compose.yml run --rm training
docker compose -f docker/docker-compose.yml down
```

`down -v` also deletes the `mlflow-data` volume, which holds the SQLite database
and the artifacts. Training is in a separate Compose profile, so it never starts
with the API. It reaches MLflow at `http://mlflow:5000` and mounts `data/`
read-only.

## Model updates on VM 2

`scripts/deploy_model.sh` installs the MLflow `champion`. Keep it in
`~/flight-delay/` with `compose.vm2.yml` and a `.env` file:

```bash
IMAGE_TAG=<commit SHA on main>
MLFLOW_URL=http://<VM 1 address>:5000
```

The VM needs `jq` and write access to `/opt/flight-delay/models`.

```bash
bash deploy_model.sh                 # Check once
bash deploy_model.sh --install-cron  # Check every 5 minutes
```

The script does nothing when the version is already installed. A new version
must pass the `/health` check, otherwise the previous model is restored and that
version is not retried. Output is in `deploy_model.log`.

To update the code, change `IMAGE_TAG` in `.env` and run
`docker compose -f compose.vm2.yml pull` and `up -d`.

## Verification

`docker-ci.yml` validates the Compose file, builds both images, checks the
non-root users, starts the stack, checks `/health` and MLflow, and logs a test
metric from the training container. Metadata and artifact persistence across
container recreation was verified manually.

## Current limits

- The serving image contains no model, and the API does not start without one.
  Place it at `models/model.joblib`, or `/opt/flight-delay/models/` on VM 2.
- The training image has no Git, so pass the commit in through `GIT_COMMIT`.
- On Linux hosts, the mounted `models/` and `reports/` folders must be writable
  by UID 10001.
