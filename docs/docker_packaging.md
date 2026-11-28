# Docker packaging

| Service | Definition | Runs as | Purpose |
| :--- | :--- | :--- | :--- |
| `api` | `docker/Dockerfile` | `appuser` (UID 10001) | FastAPI on port 8000 |
| `mlflow` | `ghcr.io/mlflow/mlflow:v3.18.0` | Image default | Tracking server on port 5000 |
| `training` | `docker/Dockerfile.train` | `trainer` (UID 10001) | Runs `python -m src.models.train` |
| `prometheus` | `prom/prometheus:v3.1.0` | Image default | Scrapes `api:8000/metrics` every 15 s, port 9090 |
| `grafana` | `grafana/grafana:11.4.0` | Image default | Dashboards, port 3000 |

Serving and training have separate dependency files, so the serving image has
no MLflow client or data tooling. `.dockerignore` keeps data, models, secrets
and Git metadata out of the images.

Image sizes in CI: serving 497 MB, training 924 MB.

## Usage

```bash
docker compose -f docker/docker-compose.yml up -d --build   # API, MLflow and monitoring
docker compose -f docker/docker-compose.yml run --rm training
docker compose -f docker/docker-compose.yml down
```

`down -v` also deletes the `mlflow-data` volume, which holds the SQLite database
and the artifacts. Training is in a separate Compose profile, so it never starts
with the API. It reaches MLflow at `http://mlflow:5000` and mounts `data/`
read-only.

## Monitoring

Prometheus reads `/metrics` from the API over the Compose network, so no port
between machines has to be opened for it. It keeps 90 days of data, capped at
2 GB, in the `prometheus-data` volume. Grafana gets its Prometheus data source
and the *Flight delay API* dashboard from `docker/monitoring/grafana/`, and
keeps its own state in `grafana-data`. The dashboard is read-only in the UI:
change the JSON file and restart Grafana.

| Dashboard row | Panels |
| :--- | :--- |
| Service | Loaded model, `/classify` requests per second, 5xx rate, p95 latency |
| Traffic and errors | Requests per second by endpoint, 4xx and 5xx rate |
| Latency | `/classify` p50, p95, p99; inference time by mode |
| Predictions | Class share (filter by mode), instances per second, fallback share, discarded values by field, batch size, label deliveries |

Locally, Grafana is at `http://localhost:3000` (user `admin`, password `admin`
unless `GRAFANA_ADMIN_PASSWORD` is set) and Prometheus at
`http://localhost:9090`.

On VM 2, `compose.vm2.yml` refuses to start without `GRAFANA_ADMIN_PASSWORD`.
Add it to `.env` **before** copying a new `compose.vm2.yml`, otherwise the next
automatic model deployment fails:

```bash
cd ~/flight-delay
openssl rand -base64 18                     # copy the generated password
echo 'GRAFANA_ADMIN_PASSWORD=<generated password>' >> .env
chmod 600 .env
```

Then copy `docker/compose.vm2.yml` and the `docker/monitoring/` folder to
`~/flight-delay/`, set `IMAGE_TAG` to the merge commit, and run
`docker compose -f compose.vm2.yml pull` and `up -d`. Over the VPN, the API is
at `http://<VM 2 address>:8080`, Grafana at `http://<VM 2 address>:3000` and
Prometheus at `http://<VM 2 address>:9090`.

Port 8000 is blocked on the DEI VMs, so VM 2 publishes the API on 8080, and the
leaderboard registration is `http://<VM 2 address>:8080/classify`. Inside the
container, and for Prometheus over the Compose network, the API stays on 8000.

## Model updates on VM 2

`scripts/deploy_model.sh` installs the MLflow `champion`. Keep it in
`~/flight-delay/` with `compose.vm2.yml` and a `.env` file:

```bash
IMAGE_TAG=<commit SHA on main>
MLFLOW_URL=http://<VM 1 address>:5000
```

The VM needs `jq` and write access to `/opt/flight-delay/models`. The API
stores `/classify` and `/labels` traffic in `/opt/flight-delay/traffic`, which
must be writable by the container user:

```bash
sudo mkdir -p /opt/flight-delay/traffic
sudo chown 10001:10001 /opt/flight-delay/traffic
```

```bash
bash deploy_model.sh                 # Check once
bash deploy_model.sh --install-cron  # Check every 5 minutes
```

The script does nothing when the version is already installed. It recreates
only the `api` container, so Prometheus and Grafana keep running. A new version
must pass the `/health` check, otherwise the previous model is restored and that
version is not retried. Output is in `deploy_model.log`.

To update the code, change `IMAGE_TAG` in `.env` and run
`docker compose -f compose.vm2.yml pull` and `up -d`.

## Delayed-label evaluation on VM 2

`python -m src.monitoring.performance` joins the stored `/classify` batches
with the `/labels` deliveries by `request_id`. For each round (one leaderboard
request), each day and the whole history, it reports Macro F1, one-vs-rest AUC
and the confusion matrix. Over the last 7 days, airlines and
airports with at least 100 labelled flights are flagged when their Macro F1 is
more than 0.10 below the overall score (`monitoring.performance` in
`configs/config.yaml`).

- AUC uses the class probabilities that the API writes to the traffic log
  (never to the response), so batches scored by an older image have none.
- Subgroups need the inputs, which only live traffic keeps.

It runs in the serving image, so VM 2 needs nothing else installed:

```bash
sudo mkdir -p /opt/flight-delay/reports
sudo chown 10001:10001 /opt/flight-delay/reports
```

```cron
15 * * * * cd $HOME/flight-delay && for mode in live sandbox; do docker compose -f compose.vm2.yml run --rm --no-deps -v /opt/flight-delay/reports:/app/reports api python -m src.monitoring.performance --mode $mode; done >> performance.log 2>&1
```

Reports: `/opt/flight-delay/reports/performance/<mode>/performance.{json,md}`.

## MLflow upgrades on VM 1

MLflow migrates its database schema in one direction only, so upgrade a
**copy** first. The nightly backups in `/opt/mlflow-backups/` hold
`artifacts/` and the database as `mlflow-backup.db`; rename it to `mlflow.db`
when restoring.

1. Extract a backup to `~/mlflow-test` (database renamed to `mlflow.db`), run
   the new image on port 5001 with `mlflow db upgrade` before `mlflow server`,
   and from VM 2 fetch the `champion` alias and its `pipeline.joblib` from
   port 5001. The file must match `/opt/flight-delay/models/model.joblib`.
2. On VM 1, with the new `compose.vm1.yml`:

   ```bash
   bash ~/flight-delay/backup_mlflow.sh            # fresh backup first
   docker compose -f compose.vm1.yml down          # never -v
   docker compose -f compose.vm1.yml pull
   docker compose -f compose.vm1.yml run --rm mlflow mlflow db upgrade sqlite:////mlflow/mlflow.db
   docker compose -f compose.vm1.yml up -d
   ```

3. On VM 2, run `bash deploy_model.sh` and check `deploy_model.log`.

The API on VM 2 serves its local model, so it keeps answering while MLflow is
down. Clients must use the same MLflow major version as the server.

## Verification

`docker-ci.yml` validates both Compose files and the Prometheus configuration,
builds both images, checks the non-root users, starts the stack, checks
`/health`, that Prometheus has scraped the API metrics, that Grafana has the
data source and the dashboard, and MLflow, and logs a test metric from the
training container. Metadata and artifact persistence across
container recreation was verified manually.

## Current limits

- The serving image contains no model, and the API does not start without one.
  Place it at `models/model.joblib`, or `/opt/flight-delay/models/` on VM 2.
- The training image has no Git, so pass the commit in through `GIT_COMMIT`.
- On Linux hosts, the mounted `models/` and `reports/` folders must be writable
  by UID 10001.
