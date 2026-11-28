"""Batch inference that always returns one class per instance.

The evaluation service scores every batch as a whole, so one bad record must not cost the other predictions:

* a value that violates the data contract is discarded (treated as missing)
  and the record is validated again without it;
* a record that still fails (a required field is missing, null or invalid,
  or the instance is not a JSON object) gets the fallback class;
* if the model fails on the batch, it predicts flight by flight, and only the
  flights it still fails on get the fallback class.

Class probabilities are computed after ``predict``, for the traffic log only,
where the delayed-label evaluation uses them for AUC. They never change the
classifications, and a failure to compute them never fails the batch.

Unknown and post-departure fields are ignored by ``FlightRecord``.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from pydantic import BaseModel, ValidationError

from src.data.contract import FlightRecord, records_to_frame

LOG = logging.getLogger("flight-delay-api")


@dataclass
class BatchResult:
    classifications: list[str]
    fallback: int = 0
    discarded_fields: Counter = field(default_factory=Counter)
    # Model class order and, per instance, its probabilities (None without them).
    class_order: Optional[list[str]] = None
    probabilities: Optional[list[Optional[list[float]]]] = None


def _invalid_fields(error: ValidationError) -> set[str]:
    return {str(item["loc"][0]) for item in error.errors() if item["loc"]}


def parse_instance(instance: Any) -> tuple[Optional[BaseModel], set[str]]:
    """The validated record (or None) and the fields whose values were discarded."""
    if not isinstance(instance, dict):
        return None, set()
    try:
        return FlightRecord.model_validate(instance), set()
    except ValidationError as error:
        discarded = _invalid_fields(error)
    usable = {key: value for key, value in instance.items() if key not in discarded}
    try:
        return FlightRecord.model_validate(usable), discarded
    except ValidationError:
        return None, discarded


def _predict_one(model, record: BaseModel) -> Optional[str]:
    try:
        return str(model.predict(records_to_frame([record]))[0])
    except Exception:
        return None


def _probabilities(model, frame) -> Optional[tuple[list[str], list[list[float]]]]:
    try:
        values = model.predict_proba(frame)
        classes = [str(name) for name in model.classes_]
        rows = [[round(float(value), 4) for value in row] for row in values]
    except Exception:
        LOG.warning("Could not compute class probabilities", exc_info=True)
        return None
    if len(rows) != len(frame) or any(len(row) != len(classes) for row in rows):
        return None
    return classes, rows


def classify(model, instances: Sequence[Any], fallback: str) -> BatchResult:
    """One class per instance, in input order, with one predict call per batch."""
    result = BatchResult(classifications=[fallback] * len(instances))
    records: dict[int, BaseModel] = {}
    for position, instance in enumerate(instances):
        record, discarded = parse_instance(instance)
        result.discarded_fields.update(discarded)
        if record is not None:
            records[position] = record

    labels: list[Optional[str]] = []
    if records:
        try:
            frame = records_to_frame(list(records.values()))
            labels = [str(label) for label in model.predict(frame)]
        except Exception:
            LOG.exception("Batch inference failed; predicting flight by flight")
            labels = [_predict_one(model, record) for record in records.values()]
        else:
            probabilities = _probabilities(model, frame)
            if probabilities is not None:
                result.class_order = probabilities[0]
                result.probabilities = [None] * len(instances)
                for position, row in zip(records, probabilities[1]):
                    result.probabilities[position] = row

    for position, label in zip(records, labels):
        if label is not None:
            result.classifications[position] = label
    result.fallback = len(instances) - sum(label is not None for label in labels)
    return result
