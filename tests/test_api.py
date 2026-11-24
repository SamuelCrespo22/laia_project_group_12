import json

import joblib
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.models.smoke import DISTANCE_BY_CLASS

client = TestClient(app)
FLIGHT = {
    "FL_DATE": "2025-05-15",
    "OP_UNIQUE_CARRIER": "AA",
    "ORIGIN": "JFK",
    "DEST": "LAX",
    "CRS_DEP_TIME": 900,
    "CRS_ARR_TIME": 1200,
    "CRS_ELAPSED_TIME": 180,
}


def flight(label):
    return {**FLIGHT, "DISTANCE": DISTANCE_BY_CLASS[label]}


def classify(*instances):
    return client.post("/classify", json={"instances": list(instances)})


def post_raw(body, path="/classify"):
    return client.post(path, content=body, headers={"content-type": "application/json"})


def stored(traffic_dir, mode, kind):
    files = list((traffic_dir / mode).glob(f"{kind}-*.jsonl"))
    return [
        json.loads(line) for file in files for line in file.read_text().splitlines()
    ]


def test_health_reports_the_loaded_model():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "flight-delay-api",
        "model": {"name": "smoke", "version": "test"},
    }


def test_classify_valid_batch():
    response = classify(FLIGHT, {**FLIGHT, "ORIGIN": "ATL"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["classifications"]) == 2
    for pred in data["classifications"]:
        assert pred in ["on_time", "moderate_delay", "severe_delay"]


def test_output_order_matches_input_order():
    labels = ["severe_delay", "on_time", "moderate_delay", "on_time", "severe_delay"]

    response = classify(*[flight(label) for label in labels])

    assert response.status_code == 200
    assert response.json() == {"classifications": labels}


def test_inference_is_one_call_per_batch(smoke_model, monkeypatch):
    calls = []

    class Spy:
        def predict(self, frame):
            calls.append(len(frame))
            return smoke_model.predict(frame)

    monkeypatch.setattr(app.state, "model", Spy())

    assert classify(FLIGHT, FLIGHT, FLIGHT).status_code == 200
    assert calls == [3]


def test_unseen_categories_are_accepted():
    unseen = {**FLIGHT, "OP_UNIQUE_CARRIER": "Q9", "ORIGIN": "ZZZ", "DEST": "YYY"}
    assert classify(unseen).status_code == 200


def test_missing_optional_fields_are_accepted():
    required = ("FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "CRS_DEP_TIME")
    assert classify({key: FLIGHT[key] for key in required}).status_code == 200


def test_extra_and_post_departure_fields_are_ignored():
    response = classify({**FLIGHT, "ARR_DELAY": 500, "NOT_A_COLUMN": "x"})
    assert response.status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        b'{"instances": [',
        b"",
        b'{"other": 1}',
        b'{"instances": "x"}',
        b'{"instances": []}',
    ],
    ids=["broken-json", "no-body", "no-instances", "not-list", "empty"],
)
def test_malformed_requests_return_400(body):
    response = post_raw(body)
    assert response.status_code == 400
    assert response.json()["detail"]


def test_errors_do_not_echo_the_request():
    response = post_raw(b'{"instances": "secret-value"}')
    assert response.status_code == 400
    assert "secret-value" not in response.text


def test_leaderboard_request_is_accepted():
    response = client.post(
        "/classify",
        json={"request_id": 17, "mode": "live", "instances": [flight("on_time")]},
    )
    assert response.status_code == 200
    assert response.json() == {"classifications": ["on_time"]}


SEVERE = flight("severe_delay")


@pytest.mark.parametrize(
    "broken",
    [
        {**SEVERE, "CRS_DEP_TIME": 2500},
        {**SEVERE, "CRS_DEP_TIME": None},
        {key: value for key, value in SEVERE.items() if key != "ORIGIN"},
        {**SEVERE, "FL_DATE": "not a date"},
        None,
        1,
    ],
    ids=["out-of-range", "null", "missing", "wrong-type", "null-instance", "number"],
)
def test_unclassifiable_instance_gets_the_fallback_class(broken):
    response = classify(SEVERE, broken, SEVERE)
    assert response.status_code == 200
    assert response.json() == {
        "classifications": ["severe_delay", "on_time", "severe_delay"]
    }


def test_invalid_optional_value_is_discarded_and_the_flight_classified():
    response = classify({**flight("severe_delay"), "CRS_ARR_TIME": "noon"})
    assert response.json() == {"classifications": ["severe_delay"]}


def test_invalid_carrier_falls_back_to_op_carrier():
    record = {**flight("severe_delay"), "OP_UNIQUE_CARRIER": 5, "OP_CARRIER": "AA"}
    assert classify(record).json() == {"classifications": ["severe_delay"]}


def test_non_finite_required_value_gets_the_fallback_class():
    body = (
        b'{"instances":[{"FL_DATE":"2025-05-15","OP_UNIQUE_CARRIER":"AA",'
        b'"ORIGIN":"JFK","DEST":"LAX","CRS_DEP_TIME":NaN,"DISTANCE":2800}]}'
    )
    response = post_raw(body)
    assert response.status_code == 200
    assert response.json() == {"classifications": ["on_time"]}


def test_model_is_not_called_without_a_valid_instance(monkeypatch):
    class Unused:
        def predict(self, frame):
            raise AssertionError("called")

    monkeypatch.setattr(app.state, "model", Unused())
    assert classify(None, None).json() == {"classifications": ["on_time", "on_time"]}


def test_model_failure_on_the_batch_is_retried_flight_by_flight(
    smoke_model, monkeypatch
):
    class FailsOnBatchesAndOnModerate:
        def predict(self, frame):
            if len(frame) > 1 or frame["DISTANCE"].iloc[0] == 1200:
                raise ValueError("secret detail")
            return smoke_model.predict(frame)

    monkeypatch.setattr(app.state, "model", FailsOnBatchesAndOnModerate())
    response = classify(
        flight("severe_delay"), flight("moderate_delay"), flight("severe_delay")
    )
    assert response.status_code == 200
    assert response.json() == {
        "classifications": ["severe_delay", "on_time", "severe_delay"]
    }
    assert "secret" not in response.text


def test_live_traffic_is_stored_with_its_inputs(traffic_dir):
    instances = [flight("severe_delay"), None]
    client.post(
        "/classify", json={"request_id": "r1", "mode": "live", "instances": instances}
    )

    [entry] = stored(traffic_dir, "live", "predictions")
    assert entry["request_id"] == "r1"
    assert entry["classifications"] == ["severe_delay", "on_time"]
    assert entry["fallback"] == 1
    assert entry["instances"] == instances
    assert entry["model"] == {"name": "smoke", "version": "test"}


@pytest.mark.parametrize(
    "mode, folder", [("sandbox", "sandbox"), (None, "unknown"), ("../x", "unknown")]
)
def test_other_traffic_is_stored_apart_without_inputs(traffic_dir, mode, folder):
    client.post("/classify", json={"mode": mode, "instances": [FLIGHT]})

    [entry] = stored(traffic_dir, folder, "predictions")
    assert entry["mode"] == folder
    assert "instances" not in entry
    assert not (traffic_dir / "live").exists()


def test_storage_failure_does_not_fail_classify(monkeypatch):
    class BrokenDisk:
        def append(self, kind, entry):
            raise OSError("disk full")

    monkeypatch.setattr(app.state, "traffic", BrokenDisk())
    response = client.post("/classify", json={"mode": "live", "instances": [FLIGHT]})
    assert response.status_code == 200


def test_labels_are_stored_as_received(traffic_dir):
    delivery = {
        "request_id": "r1",
        "mode": "sandbox",
        "labels": ["on_time", "severe_delay"],
        "outcomes": [{"ARR_DELAY": 3.0}, {"ARR_DELAY": 95.0, "WEATHER_DELAY": 80}],
        "scored_at": "2026-11-28T00:30:00Z",
    }
    response = client.post("/labels", json=delivery)

    assert response.status_code == 200
    assert response.json() == {"status": "received", "request_id": "r1"}
    [entry] = stored(traffic_dir, "sandbox", "labels")
    assert {key: entry[key] for key in delivery} == delivery
    assert "received_at" in entry


def test_labels_with_an_unexpected_shape_are_accepted(traffic_dir):
    assert client.post("/labels", json={"labels": {"0": "on_time"}}).status_code == 200
    assert len(stored(traffic_dir, "unknown", "labels")) == 1


def test_labels_that_cannot_be_stored_return_500(monkeypatch):
    class BrokenDisk:
        def append(self, kind, entry):
            raise OSError("disk full")

    monkeypatch.setattr(app.state, "traffic", BrokenDisk())
    response = TestClient(app, raise_server_exceptions=False).post(
        "/labels", json={"request_id": "r1", "mode": "live", "labels": []}
    )
    assert response.status_code == 500


def test_labels_must_be_a_json_object():
    assert post_raw(b"[1, 2]", path="/labels").status_code == 400


def test_startup_loads_the_configured_model(smoke_model, tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    joblib.dump(smoke_model, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    monkeypatch.setenv("MODEL_NAME", "flight-delay-severity")
    monkeypatch.setenv("MODEL_VERSION", "7")
    monkeypatch.setenv("TRAFFIC_DIR", str(tmp_path / "started"))

    with TestClient(app) as started:
        model = started.get("/health").json()["model"]
        response = started.post(
            "/classify", json={"mode": "live", "instances": [FLIGHT]}
        )

    assert model == {"name": "flight-delay-severity", "version": "7"}
    assert response.status_code == 200
    assert stored(tmp_path / "started", "live", "predictions")


def test_startup_fails_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with pytest.raises(RuntimeError, match="Model file not found"):
        with TestClient(app):
            pass


def test_startup_fails_with_an_unreadable_model(tmp_path, monkeypatch):
    path = tmp_path / "broken.joblib"
    path.write_bytes(b"not a model")
    monkeypatch.setenv("MODEL_PATH", str(path))
    with pytest.raises(RuntimeError, match="Could not load model"):
        with TestClient(app):
            pass
