"""Prometheus metrics of the inference API, exposed on ``GET /metrics``.

HTTP metrics come from ``prometheus-fastapi-instrumentator``:

* ``http_requests_total{handler, method, status}``: requests, with the status
  grouped as ``2xx``/``4xx``/``5xx``, for request rate and error rate;
* ``http_request_duration_seconds{handler, method}``: latency histogram for
  the p50/p95/p99 quantiles, with buckets up to the 30 s evaluation limit.

The model metrics below say what the API answered. Every label has a small,
fixed set of values (``mode`` is normalized, classes and contract fields are
fixed), so no request can create new time series.
"""

from __future__ import annotations

from typing import Iterable

from fastapi import FastAPI
from prometheus_client import Counter, Histogram, Info
from prometheus_fastapi_instrumentator import Instrumentator

# Seconds; the evaluation service gives up after 30 s.
LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30)
BATCH_BUCKETS = (1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000)

MODEL = Info("flight_delay_model", "Model loaded by the API")
BATCH_SIZE = Histogram(
    "flight_delay_batch_instances",
    "Instances per /classify request",
    ["mode"],
    buckets=BATCH_BUCKETS,
)
INFERENCE_SECONDS = Histogram(
    "flight_delay_inference_seconds",
    "Time to validate and classify one batch, without HTTP overhead",
    ["mode"],
    buckets=LATENCY_BUCKETS,
)
PREDICTIONS = Counter(
    "flight_delay_predictions",
    "Classes returned by /classify, fallback answers included",
    ["mode", "label"],
)
FALLBACKS = Counter(
    "flight_delay_fallback_predictions",
    "Instances answered with the fallback class",
    ["mode"],
)
DISCARDED = Counter(
    "flight_delay_discarded_values",
    "Instance values that violated the data contract and were treated as missing",
    ["field"],
)
LABEL_DELIVERIES = Counter(
    "flight_delay_label_deliveries",
    "Deliveries of true outcomes stored by /labels",
    ["mode"],
)


def instrument(app: FastAPI) -> None:
    """Add the HTTP metrics middleware and the /metrics endpoint."""
    Instrumentator(excluded_handlers=["/metrics"]).instrument(
        app, latency_lowr_buckets=LATENCY_BUCKETS
    ).expose(app, include_in_schema=False)


def set_model(name: str, version: str) -> None:
    MODEL.info({"name": str(name), "version": str(version)})


def observe_classification(
    mode: str,
    classifications: Iterable[str],
    fallback: int,
    discarded_fields: dict[str, int],
    seconds: float,
) -> None:
    classifications = list(classifications)
    BATCH_SIZE.labels(mode).observe(len(classifications))
    INFERENCE_SECONDS.labels(mode).observe(seconds)
    for label in set(classifications):
        PREDICTIONS.labels(mode, label).inc(classifications.count(label))
    FALLBACKS.labels(mode).inc(fallback)
    for field, count in discarded_fields.items():
        DISCARDED.labels(field).inc(count)


def observe_labels(mode: str) -> None:
    LABEL_DELIVERIES.labels(mode).inc()
