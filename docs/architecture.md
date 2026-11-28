# Architecture

```text
Developer -> GitHub -> GitHub Actions (tests, image builds, GHCR)
                             |
             +---------------+---------------+
             v                               v
         DEI VM 1                        DEI VM 2
  MLflow tracking and registry -model-> FastAPI (/classify, /labels, /health)
  DVC remote (bonus)                    Prometheus and Grafana
```

The VMs are inside DEI's network and are not reachable from outside it, so
automated deployment needs self-hosted GitHub Actions runners on them.

## Components

| Component | Implementation | State |
| :--- | :--- | :--- |
| Ingestion | `scripts/ingest_bts.py` | Done |
| Data contract | `src/data/contract.py` | Done |
| Preprocessing | `src/data/preprocessor.py`, `make_dataset.py` | Done |
| Temporal splits | `src/data/temporal_split.py`, `scripts/create_temporal_splits.py` | Done |
| Encoders | `src/features/encoders.py` | Done |
| CI and image publishing | `.github/workflows/` | Done |
| Containers | `docker/` | Done, without a trained model |
| Training | `src/models/train.py` | Done: numeric baseline on a sample |
| Experiment tracking | `src/models/tracking.py` | Done: runs, registry and aliases |
| Inference | `src/api/app.py`, `src/api/inference.py` | Done: leaderboard `/classify` and `/labels`, one class per instance |
| Traffic log | `src/monitoring/traffic.py` | Done: predictions (with class probabilities) and labels by mode, input for monitoring |
| Delayed-label evaluation | `src/monitoring/performance.py` | Done: Macro F1, AUC and confusion matrix per round and day; degraded airlines and airports flagged (4.3) |
| VM deployment and registration | VM 1 and VM 2 | Planned (#14, #10) |
| System monitoring | `src/monitoring/metrics.py`, `docker/monitoring/` | Done: `/metrics`, Prometheus and Grafana dashboard on VM 2 (#22) |
| Drift, label evaluation, alerts | Prometheus rules, traffic log | Planned (Milestone 4) |

## Design rules

- **No leakage:** only columns allowlisted in `configs/config.yaml` and marked
  pre-departure in the data contract reach the features.
- **No train/serve skew:** training and `/classify` use the same
  `build_features`, and encoders are saved with the classifier as one pipeline.
- **Evaluation:** chronological splits. Models are selected by validation Macro
  F1, with multiclass log loss as the secondary metric.
- **Model promotion:** MLflow registry aliases `staging` and `champion`, which
  replace the deprecated stages. The rule is in `docs/training.md`.
- **Monitoring:** the API exposes request, latency, error and prediction
  metrics on `/metrics`, scraped by Prometheus on VM 2 over the Compose network
  and shown in Grafana. Metric labels take only fixed values (mode, class,
  contract field), so requests cannot create new series. Delayed labels are
  joined with the stored predictions by `request_id` in periodic reports (Macro
  F1, AUC, confusion matrix, and airlines or airports far below the overall
  score). Input drift is planned.
