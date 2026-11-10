# Roadmap

Status and full acceptance criteria are in the GitHub issues. Work not yet on
GitHub is described here.

| Milestone | Due | Goal |
| :--- | :--- | :--- |
| 1. Data pipeline | 25 Oct | Ingestion, EDA, leakage prevention, splits |
| 2. Checkpoint (5% of the grade) | 10 Nov | Baseline model, code and experiment versioning, CI/CD, containerized training and serving, endpoint registration |
| 3. Production serving | 24 Nov | Self-hosted runners, persistent MLflow, automated deployment |
| 4. Monitoring | 5 Dec | System metrics, drift, subgroup performance, alerts |
| 5. Final submission (95% of the grade) | 11 Dec | Complete system, report, assessments, individual defenses |

**Definition of done:** a dedicated branch and a reviewed pull request with
`Closes #N`, passing CI, unit tests on synthetic fixtures, parameters in
`configs/config.yaml`, and data decisions recorded in `docs/data_decisions.md`.

Every member must demonstrate their own contribution at the checkpoint and at the
defenses. Column names follow the TranStats form everywhere.

## Milestone 1

| Id | Work | GitHub |
| :--- | :--- | :--- |
| 1.1 | Ingestion and raw storage | #1 |
| 1.2 | EDA | #2 |
| 1.3 | Leakage prevention and preprocessing | #3 |
| 1.4 | Encoders for unseen categories | #4 |
| 1.5 | Temporal splits | #11 |
| 1.6 | Data contract | #12 |
| 1.7 | DVC stages (bonus) | #13 |

## Milestone 2

| Order | Id | Work | GitHub | Depends on |
| :---: | :--- | :--- | :--- | :--- |
| 1 | 2.7 | VM provisioning and first deployment | #14 | None |
| 1 | 2.1 | Baseline model and training pipeline | #5 | Milestone 1 |
| 1 | 2.2 | MLflow tracking and registry | #6 | Integrated with 2.1 |
| 2 | 2.3 | `/classify` with the real model | #7 | 2.1, 2.2 |
| 2 | 2.4 | Docker images | #8 | 2.1, 2.3 |
| 2 | 2.5 | CI/CD workflows | #9 | 2.4 |
| 3 | 2.6 | Endpoint registration | #10 | 2.3, 2.4, 2.7 |

## Milestone 3

| Id | Work | Done when |
| :--- | :--- | :--- |
| 3.2 | Self-hosted runners | A merge to `main` deploys the new image to VM 2 |
| 3.3 | Production MLflow on VM 1 | The server is a managed service, aliases run against it, and VM 2 deploys the `production` model |

## Milestone 4

| Id | Work | Done when |
| :--- | :--- | :--- |
| 4.1 | Prometheus and Grafana (#22) | See the issue |
| 4.2 | Input drift detection | Significant shifts in schedule, distance or airline mix are flagged in periodic reports |
| 4.3 | Delayed-label evaluation | Macro F1 and confusion matrices per round; degradation isolated to airlines or airports is detected |
| 4.4 | Alerting | Rules for latency, errors, drift and performance drops, demonstrated in the report |

## Milestone 5

| Id | Work | Done when |
| :--- | :--- | :--- |
| 5.1 | DVC remote and CI (bonus) | `dvc pull` works for all members and in CI |
| 5.2 | Continuous training (bonus) | Retraining on new labels, promoted only if it beats production |
| 5.3 | Technical report | PDF covering all essay sections |
| 5.4 | Assessments and defenses | Peer assessment done; every member masters the whole project |
