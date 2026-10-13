from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "flight-delay-api"}


def test_classify_valid_batch():
    payload = {
        "instances": [
            {
                "FL_DATE": "2025-05-15",
                "OP_CARRIER": "AA",
                "ORIGIN": "JFK",
                "DEST": "LAX",
                "CRS_DEP_TIME": 900
            },
            {
                "FL_DATE": "2025-05-16",
                "OP_CARRIER": "DL",
                "ORIGIN": "ATL",
                "DEST": "ORD",
                "CRS_DEP_TIME": 1400
            }
        ]
    }
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "classifications" in data
    assert len(data["classifications"]) == 2
    for pred in data["classifications"]:
        assert pred in ["on_time", "moderate_delay", "severe_delay"]


def test_classify_empty_instances():
    payload = {"instances": []}
    response = client.post("/classify", json=payload)
    assert response.status_code == 400

