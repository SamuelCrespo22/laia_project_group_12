"""Prometheus metrics exposed by the API on /metrics."""

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from src.api.app import app
from src.monitoring import metrics
from tests.test_api import FLIGHT, flight

client = TestClient(app)


def sample(metric, **labels):
    return REGISTRY.get_sample_value(metric, labels) or 0.0


def test_metrics_endpoint_serves_prometheus_text():
    client.post("/classify", json={"mode": "sandbox", "instances": [FLIGHT]})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert 'http_requests_total{handler="/classify",method="POST",status="2xx"}' in body
    assert "http_request_duration_seconds_bucket" in body
    assert "flight_delay_predictions_total" in body


def test_metrics_endpoint_is_neither_documented_nor_counted():
    client.get("/metrics")
    client.get("/metrics")
    assert "/metrics" not in client.get("/openapi.json").json()["paths"]
    assert 'handler="/metrics"' not in client.get("/metrics").text


def test_latency_buckets_reach_the_evaluation_timeout():
    client.post("/classify", json={"instances": [FLIGHT]})
    bucket = sample(
        "http_request_duration_seconds_bucket",
        handler="/classify",
        method="POST",
        le="30.0",
    )
    assert bucket >= 1


def test_classify_counts_every_returned_class_by_mode():
    labels = ["on_time", "severe_delay", "severe_delay"]
    before = {
        label: sample("flight_delay_predictions_total", mode="live", label=label)
        for label in set(labels)
    }
    batches = sample("flight_delay_batch_instances_count", mode="live")
    instances = sample("flight_delay_batch_instances_sum", mode="live")

    response = client.post(
        "/classify",
        json={"mode": "live", "instances": [flight(label) for label in labels]},
    )

    for label, count in {"on_time": 1, "severe_delay": 2}.items():
        assert response.json()["classifications"].count(label) == count
        after = sample("flight_delay_predictions_total", mode="live", label=label)
        assert after - before[label] == count
    assert sample("flight_delay_batch_instances_count", mode="live") - batches == 1
    assert sample("flight_delay_batch_instances_sum", mode="live") - instances == 3
    assert sample("flight_delay_inference_seconds_count", mode="live") >= 1


def test_fallbacks_and_discarded_values_are_counted():
    fallbacks = sample("flight_delay_fallback_predictions_total", mode="unknown")
    discarded = sample("flight_delay_discarded_values_total", field="CRS_ARR_TIME")

    response = client.post(
        "/classify",
        json={"instances": [{**FLIGHT, "CRS_ARR_TIME": "noon"}, "not a flight", None]},
    )

    assert response.status_code == 200
    assert (
        sample("flight_delay_fallback_predictions_total", mode="unknown") - fallbacks
        == 2
    )
    assert (
        sample("flight_delay_discarded_values_total", field="CRS_ARR_TIME") - discarded
        == 1
    )


def test_unexpected_modes_share_one_series():
    before = sample("flight_delay_batch_instances_count", mode="unknown")
    client.post("/classify", json={"mode": "x" * 50, "instances": [FLIGHT]})
    assert 'mode="' + "x" * 50 not in client.get("/metrics").text
    assert sample("flight_delay_batch_instances_count", mode="unknown") - before == 1


def test_label_deliveries_are_counted_once_stored():
    before = sample("flight_delay_label_deliveries_total", mode="sandbox")
    response = client.post(
        "/labels",
        json={"request_id": "r1", "mode": "sandbox", "labels": [], "outcomes": []},
    )
    assert response.status_code == 200
    assert sample("flight_delay_label_deliveries_total", mode="sandbox") - before == 1


def test_model_info_reports_the_loaded_version():
    metrics.set_model("flight-delay-severity", 3)
    assert (
        sample("flight_delay_model_info", name="flight-delay-severity", version="3")
        == 1
    )
