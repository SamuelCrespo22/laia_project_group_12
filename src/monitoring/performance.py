"""Delayed-label evaluation: joins /classify batches with /labels deliveries.

A round is one leaderboard request, with the same ``request_id`` in both logs.
The report gives Macro F1, one-vs-rest AUC and the confusion matrix per round,
per day of prediction and overall. Airlines and airports whose Macro F1 over the
last ``window_days`` is more than ``degradation_margin`` below the overall score
of that window are flagged. Subgroups need the inputs, which only live traffic
keeps; AUC needs the class probabilities the API logs.

Usage: python -m src.monitoring.performance --mode live
"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from src.data.preprocessor import CLASSES, DEFAULT_CONFIG_PATH
from src.monitoring.traffic import normalize_mode

GROUPS = ("OP_UNIQUE_CARRIER", "ORIGIN", "DEST")
PROBABILITIES = [f"p_{name}" for name in CLASSES]
COLUMNS = ["request_id", "predicted_at", "model_version", "label", "prediction"]


def read(directory, kind, skipped):
    """The last entry per request_id in every ``<kind>-*.jsonl`` file."""
    entries = {}
    for path in sorted(directory.glob(f"{kind}-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
                entries[str(entry["request_id"])] = entry
            except (ValueError, TypeError, KeyError):
                skipped["unreadable_lines"] += 1
    return entries


def _aligned(values, size):
    return values if isinstance(values, list) and len(values) == size else [None] * size


def _group(value):
    text = "" if value is None else str(value).strip().upper()
    return text or "<missing>"


def flights(predictions, labels, skipped):
    """One row per labelled flight, with probabilities and groups when logged."""
    rows = []
    for request_id, delivery in labels.items():
        prediction = predictions.get(request_id)
        if prediction is None:
            skipped["labels_without_predictions"] += 1
            continue
        truth, guess = delivery.get("labels"), prediction.get("classifications")
        if (
            not isinstance(truth, list)
            or not isinstance(guess, list)
            or len(truth) != len(guess)
        ):
            skipped["misaligned_rounds"] += 1
            continue
        order = prediction.get("class_order")
        known = isinstance(order, list) and sorted(order) == sorted(CLASSES)
        probabilities = _aligned(
            prediction.get("probabilities") if known else None, len(truth)
        )
        instances = _aligned(prediction.get("instances"), len(truth))
        common = {
            "request_id": request_id,
            "predicted_at": str(prediction.get("received_at", "")),
            "model_version": (prediction.get("model") or {}).get("version"),
        }
        for label, predicted, row, instance in zip(
            truth, guess, probabilities, instances
        ):
            flight = {**common, "label": label, "prediction": predicted}
            if isinstance(row, list) and len(row) == len(CLASSES):
                flight.update({f"p_{name}": row[order.index(name)] for name in CLASSES})
            if isinstance(instance, dict):
                flight.update(
                    {column: _group(instance.get(column)) for column in GROUPS}
                )
            rows.append(flight)
    skipped["awaiting_labels"] += len(predictions.keys() - labels.keys())

    frame = pd.DataFrame(rows, columns=COLUMNS + PROBABILITIES + list(GROUPS))
    valid = frame["label"].isin(CLASSES) & frame["prediction"].isin(CLASSES)
    skipped["flights_with_unknown_classes"] += int((~valid).sum())
    return frame[valid]


def score(frame):
    """Macro F1 over the classes present, as ``f1_score(average="macro")``, so a
    small round is not penalised for a class it does not contain. AUC is the
    mean one-vs-rest AUC of the classes with both positives and negatives."""
    scored = frame.dropna(subset=PROBABILITIES)
    aucs = [
        roc_auc_score(scored["label"] == name, scored[f"p_{name}"])
        for name in CLASSES
        if (scored["label"] == name).nunique() == 2
    ]
    return {
        "support": len(frame),
        "macro_f1": float(
            f1_score(
                frame["label"], frame["prediction"], average="macro", zero_division=0
            )
        ),
        "accuracy": float((frame["label"] == frame["prediction"]).mean()),
        "auc_ovr_macro": float(np.mean(aucs)) if aucs else None,
        "auc_support": len(scored),
        "confusion_matrix": confusion_matrix(
            frame["label"], frame["prediction"], labels=list(CLASSES)
        ).tolist(),
    }


def subgroups(frame, min_support, margin):
    """Macro F1 of each airline and airport with enough flights."""
    frame = frame.dropna(subset=list(GROUPS))
    overall = score(frame)["macro_f1"] if len(frame) else None
    groups = []
    for column in GROUPS:
        for value, group in frame.groupby(column):
            if len(group) >= min_support:
                f1 = score(group)["macro_f1"]
                groups.append(
                    {
                        "column": column,
                        "value": value,
                        "support": len(group),
                        "macro_f1": f1,
                        "degraded": f1 < overall - margin,
                    }
                )
    return {"support": len(frame), "macro_f1": overall, "groups": groups}


def build_report(traffic_dir, mode, min_support, margin, window_days, now=None):
    mode = normalize_mode(mode)
    now = now or datetime.now(timezone.utc)
    skipped = Counter()
    directory = Path(traffic_dir) / mode
    frame = flights(
        read(directory, "predictions", skipped),
        read(directory, "labels", skipped),
        skipped,
    )
    since = (now - timedelta(days=window_days)).isoformat()
    rounds = [
        {
            "request_id": request_id,
            **group.iloc[0][["predicted_at", "model_version"]].to_dict(),
            **score(group),
        }
        for request_id, group in frame.groupby("request_id")
    ]
    return {
        "mode": mode,
        "generated_at": now.isoformat(timespec="seconds"),
        "settings": {
            "min_subgroup_support": min_support,
            "degradation_margin": margin,
            "window_days": window_days,
        },
        "overall": score(frame) if len(frame) else None,
        "rounds": sorted(rounds, key=lambda row: row["predicted_at"]),
        "days": {
            day: score(group)
            for day, group in frame.groupby(frame["predicted_at"].str[:10])
        },
        "subgroups": subgroups(
            frame[frame["predicted_at"] >= since], min_support, margin
        ),
        "skipped": {key: count for key, count in sorted(skipped.items()) if count},
    }


def _number(value):
    return "n/a" if value is None else f"{value:.4f}"


def render_markdown(report):
    overall, settings, groups = (
        report["overall"],
        report["settings"],
        report["subgroups"],
    )
    lines = [
        f"# Delayed-label evaluation ({report['mode']}), {report['generated_at']}",
        "",
    ]
    if overall is None:
        return "\n".join(lines + ["No labelled rounds yet."]) + "\n"
    lines += [
        f"{overall['support']} flights in {len(report['rounds'])} rounds: Macro F1 {_number(overall['macro_f1'])}, "
        f"AUC {_number(overall['auc_ovr_macro'])} ({overall['auc_support']} flights with probabilities).",
        "",
        "| Actual \\ predicted | " + " | ".join(CLASSES) + " |",
        "| :--- |" + " ---: |" * len(CLASSES),
        *(
            f"| {name} | " + " | ".join(map(str, row)) + " |"
            for name, row in zip(CLASSES, overall["confusion_matrix"])
        ),
        "",
        "| Day | Flights | Macro F1 | AUC |",
        "| :--- | ---: | ---: | ---: |",
        *(
            f"| {day} | {row['support']} | {_number(row['macro_f1'])} | {_number(row['auc_ovr_macro'])} |"
            for day, row in report["days"].items()
        ),
        "",
    ]
    flagged = [group for group in groups["groups"] if group["degraded"]]
    if not groups["support"]:
        lines.append(
            "Subgroups: no flights with stored inputs (only live traffic keeps them)."
        )
    else:
        lines.append(
            f"Subgroups, last {settings['window_days']} days ({groups['support']} flights, Macro F1 "
            f"{_number(groups['macro_f1'])}): {len(flagged)} airlines or airports more than "
            f"{settings['degradation_margin']:.2f} below."
        )
        lines += [
            f"- {g['column']} {g['value']}: Macro F1 {_number(g['macro_f1'])}, {g['support']} flights"
            for g in flagged
        ]
    if report["skipped"]:
        lines += [
            "",
            "Skipped: "
            + ", ".join(f"{key} {count}" for key, count in report["skipped"].items()),
        ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    settings = config["monitoring"]["performance"]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode", default="live", choices=["live", "sandbox", "unknown"]
    )
    default_traffic = os.environ.get("TRAFFIC_DIR") or config["serving"]["traffic_dir"]
    parser.add_argument("--traffic-dir", type=Path, default=Path(default_traffic))
    parser.add_argument("--output", type=Path, default=Path("reports/performance"))
    args = parser.parse_args(argv)

    report = build_report(
        args.traffic_dir,
        args.mode,
        settings["min_subgroup_support"],
        settings["degradation_margin"],
        settings["window_days"],
    )
    folder = args.output / report["mode"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "performance.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(report)
    (folder / "performance.md").write_text(markdown, encoding="utf-8")
    print(markdown.splitlines()[2] if report["overall"] else "No labelled rounds yet.")
    return report


if __name__ == "__main__":
    main()
