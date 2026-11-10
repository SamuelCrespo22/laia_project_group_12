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


def post_raw(body):
    return client.post(
        "/classify", content=body, headers={"content-type": "application/json"}
    )


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
        b'{"instances": [1]}',
    ],
    ids=["broken-json", "no-body", "no-instances", "not-list", "empty", "not-object"],
)
def test_malformed_requests_return_400(body):
    response = post_raw(body)
    assert response.status_code == 400
    assert response.json()["detail"]


def test_invalid_record_returns_400_naming_the_field():
    response = classify(FLIGHT, {**FLIGHT, "CRS_DEP_TIME": 2500})
    assert response.status_code == 400
    locations = [item["loc"] for item in response.json()["detail"]]
    assert ["body", "instances", 1, "CRS_DEP_TIME"] in locations


def test_errors_do_not_echo_the_request():
    response = classify({**FLIGHT, "CRS_DEP_TIME": 2500})
    assert "input" not in response.json()["detail"][0]


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_optional_value_is_treated_as_missing(value):
    body = (
        '{"instances":[{"FL_DATE":"2025-05-15","OP_UNIQUE_CARRIER":"AA",'
        f'"ORIGIN":"JFK","DEST":"LAX","CRS_DEP_TIME":900,"DISTANCE":{value}}}]}}'
    )
    assert post_raw(body.encode()).status_code == 200


def test_non_finite_required_value_returns_400():
    body = (
        b'{"instances":[{"FL_DATE":"2025-05-15","OP_UNIQUE_CARRIER":"AA",'
        b'"ORIGIN":"JFK","DEST":"LAX","CRS_DEP_TIME":NaN}]}'
    )
    assert post_raw(body).status_code == 400


def test_inference_failure_returns_500_without_details(monkeypatch):
    class Broken:
        def predict(self, frame):
            raise ValueError("secret detail")

    monkeypatch.setattr(app.state, "model", Broken())
    response = TestClient(app, raise_server_exceptions=False).post(
        "/classify", json={"instances": [FLIGHT]}
    )
    assert response.status_code == 500
    assert "secret" not in response.text


def test_startup_loads_the_configured_model(smoke_model, tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    joblib.dump(smoke_model, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    monkeypatch.setenv("MODEL_NAME", "flight-delay-severity")
    monkeypatch.setenv("MODEL_VERSION", "7")

    with TestClient(app) as started:
        model = started.get("/health").json()["model"]
        response = started.post("/classify", json={"instances": [FLIGHT]})

    assert model == {"name": "flight-delay-severity", "version": "7"}
    assert response.status_code == 200


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
