import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import f1_score

from src.api.app import app
from src.models.smoke import DISTANCE_BY_CLASS
from src.monitoring.performance import build_report, main, render_markdown
from src.monitoring.traffic import TrafficLog

NOW = datetime(2026, 12, 3, 12, tzinfo=timezone.utc)
ON, MODERATE, SEVERE = "on_time", "moderate_delay", "severe_delay"


def report(traffic_dir, mode="live", min_support=100):
    return build_report(traffic_dir, mode, min_support, 0.10, 7, now=NOW)


def log(
    traffic_dir, request_id, predicted, labels, mode="live", day="2026-12-01", **extra
):
    traffic = TrafficLog(traffic_dir)
    common = {"request_id": request_id, "mode": mode}
    if predicted is not None:
        traffic.append(
            "predictions",
            {
                **common,
                "received_at": f"{day}T10:00:00.000+00:00",
                "model": {"version": "3"},
                "classifications": predicted,
                **extra,
            },
        )
    if labels is not None:
        traffic.append(
            "labels",
            {**common, "received_at": f"{day}T18:00:00.000+00:00", "labels": labels},
        )


def test_rounds_join_predictions_and_labels_by_request_id(traffic_dir):
    log(traffic_dir, "a", [ON, SEVERE, ON], [ON, MODERATE, ON])
    log(traffic_dir, "b", [MODERATE, ON], [MODERATE, SEVERE])
    log(traffic_dir, "manual", [ON], None)
    log(traffic_dir, "teste-1000", None, [ON])

    result = report(traffic_dir)

    assert [row["request_id"] for row in result["rounds"]] == ["a", "b"]
    assert result["rounds"][0]["model_version"] == "3"
    assert result["skipped"] == {"awaiting_labels": 1, "labels_without_predictions": 1}
    truth = [ON, MODERATE, ON, MODERATE, SEVERE]
    guess = [ON, SEVERE, ON, MODERATE, ON]
    assert result["overall"]["macro_f1"] == pytest.approx(
        f1_score(truth, guess, average="macro")
    )
    assert result["overall"]["confusion_matrix"] == [[2, 0, 0], [0, 1, 1], [1, 0, 0]]


def test_macro_f1_ignores_classes_a_round_does_not_contain(traffic_dir):
    log(traffic_dir, "a", [ON, MODERATE], [ON, MODERATE])

    assert report(traffic_dir)["rounds"][0]["macro_f1"] == 1.0


def test_auc_reads_probabilities_in_the_stored_class_order(traffic_dir):
    log(
        traffic_dir,
        "a",
        [ON, SEVERE, MODERATE, ON],
        [ON, SEVERE, MODERATE, ON],
        class_order=[SEVERE, ON, MODERATE],
        probabilities=[[0.1, 0.8, 0.1], [0.7, 0.1, 0.2], [0.1, 0.2, 0.7], None],
    )
    log(traffic_dir, "old", [ON, SEVERE], [ON, SEVERE])

    new, old = report(traffic_dir)["rounds"]

    assert (new["auc_ovr_macro"], new["auc_support"]) == (1.0, 3)
    assert (old["auc_ovr_macro"], old["auc_support"]) == (None, 0)


def test_bad_rounds_flights_and_lines_are_skipped(traffic_dir):
    log(traffic_dir, "short", [ON], [ON, ON])
    log(traffic_dir, "odd", [ON, ON], [ON, "cancelled"])
    with (traffic_dir / "live" / "labels-2026-12-01.jsonl").open("a") as file:
        file.write("{not json\n[1, 2]\n")

    result = report(traffic_dir)

    assert result["overall"]["support"] == 1
    assert result["skipped"] == {
        "flights_with_unknown_classes": 1,
        "misaligned_rounds": 1,
        "unreadable_lines": 2,
    }


def test_days_follow_the_prediction_date(traffic_dir):
    log(traffic_dir, "a", [ON], [ON], day="2026-12-01")
    log(traffic_dir, "b", [ON, ON], [ON, SEVERE], day="2026-12-02")

    days = report(traffic_dir)["days"]

    assert {day: row["support"] for day, row in days.items()} == {
        "2026-12-01": 1,
        "2026-12-02": 2,
    }


def carrier_round(traffic_dir, carrier, correct, size, day="2026-12-01"):
    truth = [ON, SEVERE] * (size // 2)
    guess = truth if correct else [SEVERE, ON] * (size // 2)
    instances = [{"OP_UNIQUE_CARRIER": carrier, "ORIGIN": "JFK"}] * size
    log(traffic_dir, carrier, guess, truth, day=day, instances=instances)


def test_an_airline_far_below_the_window_is_flagged(traffic_dir):
    carrier_round(traffic_dir, "aa", correct=True, size=300)
    carrier_round(traffic_dir, "UA", correct=False, size=120)
    carrier_round(traffic_dir, "DL", correct=False, size=20)

    groups = report(traffic_dir)["subgroups"]["groups"]

    carriers = {g["value"]: g for g in groups if g["column"] == "OP_UNIQUE_CARRIER"}
    assert set(carriers) == {"AA", "UA"}
    assert carriers["UA"]["degraded"] and not carriers["AA"]["degraded"]
    assert [(g["column"], g["value"], g["support"]) for g in groups[2:]] == [
        ("ORIGIN", "JFK", 440),
        ("DEST", "<missing>", 440),
    ]


def test_subgroups_only_use_the_recent_window(traffic_dir):
    carrier_round(traffic_dir, "AA", correct=True, size=200)
    carrier_round(traffic_dir, "UA", correct=False, size=200, day="2026-11-20")

    subgroups = report(traffic_dir)["subgroups"]

    assert subgroups["support"] == 200
    assert not any(group["degraded"] for group in subgroups["groups"])


def test_reports_without_inputs_or_labels(traffic_dir):
    assert "No labelled rounds yet." in render_markdown(report(traffic_dir))

    log(traffic_dir, "a", [ON], [ON], mode="sandbox")
    result = report(traffic_dir, mode="sandbox")

    assert result["subgroups"]["support"] == 0
    assert "only live traffic keeps them" in render_markdown(result)


def test_cli_writes_json_and_markdown(traffic_dir, tmp_path):
    log(traffic_dir, "a", [ON, SEVERE], [ON, ON], mode="sandbox")
    output = tmp_path / "reports"

    main(
        [
            "--mode",
            "sandbox",
            "--traffic-dir",
            str(traffic_dir),
            "--output",
            str(output),
        ]
    )

    folder = output / "sandbox"
    stored = json.loads((folder / "performance.json").read_text())
    assert stored["overall"]["confusion_matrix"] == [[1, 0, 1], [0, 0, 0], [0, 0, 0]]
    assert "| on_time | 1 | 0 | 1 |" in (folder / "performance.md").read_text()


def test_api_traffic_is_evaluated_with_auc(traffic_dir):
    client = TestClient(app)
    classes = [ON, MODERATE, SEVERE, ON]
    instances = [
        {
            "FL_DATE": "2025-05-15",
            "OP_UNIQUE_CARRIER": "AA",
            "ORIGIN": "JFK",
            "DEST": "LAX",
            "CRS_DEP_TIME": 900,
            "DISTANCE": DISTANCE_BY_CLASS[label],
        }
        for label in classes
    ]
    body = {"request_id": "r1", "mode": "live", "instances": instances}
    assert client.post("/classify", json=body).json() == {"classifications": classes}
    delivery = {"request_id": "r1", "mode": "live", "labels": classes}
    assert client.post("/labels", json=delivery).status_code == 200

    result = report(traffic_dir, min_support=1)

    assert result["overall"]["macro_f1"] == 1.0
    assert result["overall"]["auc_ovr_macro"] == 1.0
    assert result["subgroups"]["groups"][0]["value"] == "AA"
