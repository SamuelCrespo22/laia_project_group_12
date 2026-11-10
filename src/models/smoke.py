"""Small model on synthetic flights, to test the serving path in CI and in tests.

It says nothing about predictive quality. The class depends only on ``DISTANCE``,
so tests can check that each prediction belongs to its own flight.

Usage: python -m src.models.smoke --output models/model.joblib
"""

import argparse
from pathlib import Path

import joblib

from src.data.contract import FlightRecord, records_to_frame
from src.data.preprocessor import CLASSES, default_config
from src.models.factory import create_model

DISTANCE_BY_CLASS = {"on_time": 300, "moderate_delay": 1200, "severe_delay": 2800}
EXPERIMENT = {
    "name": "smoke",
    "model": "logistic",
    "params": {"max_iter": 500},
    "scaling": "standard",
    "features": {"mode": "minimal_numeric"},
}


def build_smoke_model():
    """A fitted pipeline with the same structure as a trained candidate."""
    records, labels = [], []
    for label, distance in DISTANCE_BY_CLASS.items():
        for offset in range(30):
            records.append(
                FlightRecord(
                    FL_DATE="2023-01-02",
                    OP_UNIQUE_CARRIER="AA",
                    ORIGIN="JFK",
                    DEST="LAX",
                    CRS_DEP_TIME=900,
                    CRS_ARR_TIME=1200,
                    CRS_ELAPSED_TIME=180,
                    DISTANCE=distance + offset,
                )
            )
            labels.append(label)
    model, _ = create_model(EXPERIMENT, default_config(), 42, 1, CLASSES)
    return model.fit(records_to_frame(records), labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(build_smoke_model(), args.output)


if __name__ == "__main__":
    main()
