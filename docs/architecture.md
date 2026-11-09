# Architecture

```text
Developer -> GitHub -> GitHub Actions (tests, image builds, GHCR)
                             |
             +---------------+---------------+
             v                               v
         DEI VM 1                        DEI VM 2
  MLflow tracking and registry -model-> FastAPI (/classify, /health)
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
| Training | `src/models/train.py` | Planned (#5) |
| Experiment tracking | MLflow | Server runs in Compose, integration planned (#6) |
| Inference | `src/api/app.py` | Validation done, returns a placeholder (#7) |
| VM deployment and registration | VM 1 and VM 2 | Planned (#14, #10) |
| Monitoring | Prometheus, Grafana, drift checks | Planned (Milestone 4) |

## Design rules

- **No leakage:** only columns allowlisted in `configs/config.yaml` and marked
  pre-departure in the data contract reach the features.
- **No train/serve skew:** training and `/classify` use the same
  `build_features`, and encoders are saved with the classifier as one pipeline.
- **Evaluation:** chronological splits. Models are selected by validation Macro
  F1, with multiclass log loss as the secondary metric.
- **Model promotion:** planned through MLflow registry aliases, `staging` then
  `production`. The alias API replaces the deprecated stages.
- **Monitoring:** planned for system metrics, input drift, and performance by
  airline and origin airport once delayed labels arrive.
