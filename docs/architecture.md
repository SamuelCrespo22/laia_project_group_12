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
| Training | `src/models/train.py` | Done: numeric baseline on a sample |
| Experiment tracking | `src/models/tracking.py` | Done: runs, registry and aliases |
| Inference | `src/api/app.py` | Done: loads the pipeline from `MODEL_PATH` |
| VM deployment and registration | VM 1 and VM 2 | Planned (#14, #10) |
| Monitoring | Prometheus, Grafana, drift checks | Planned (Milestone 4) |

## Design rules

- **No leakage:** only columns allowlisted in `configs/config.yaml` and marked
  pre-departure in the data contract reach the features.
- **No train/serve skew:** training and `/classify` use the same
  `build_features`, and encoders are saved with the classifier as one pipeline.
- **Evaluation:** chronological splits. Models are selected by validation Macro
  F1, with multiclass log loss as the secondary metric.
- **Model promotion:** MLflow registry aliases `staging` and `champion`, which
  replace the deprecated stages. The rule is in `docs/training.md`.
- **Monitoring:** planned for system metrics, input drift, and performance by
  airline and origin airport once delayed labels arrive.
