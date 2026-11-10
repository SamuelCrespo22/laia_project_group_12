"""Library-independent evaluation: labels, predictions and ordered probabilities."""

import numpy as np


class EvaluationAccumulator:
    """Exact sufficient statistics; memory scales with classes and subgroup counts."""

    def __init__(self, class_order):
        self.classes = list(class_order)
        self.matrix = np.zeros((len(self.classes), len(self.classes)), dtype=np.int64)
        self.loss = 0.0
        self.groups = {}

    def update(self, labels, predictions, probabilities, metadata=None):
        report = evaluate(labels, predictions, probabilities, self.classes)
        self.matrix += np.asarray(report["confusion_matrix"])
        self.loss += report["log_loss"] * report["support"]
        if metadata is not None:
            if len(metadata) != len(labels):
                raise ValueError("Metadata is not aligned with evaluated rows")
            for column in ("OP_UNIQUE_CARRIER", "ORIGIN"):
                values = metadata[column].astype("string").fillna("<missing>")
                groups = self.groups.setdefault(column, {})
                for value in values.unique():
                    mask = values.eq(value).to_numpy(dtype=bool)
                    if str(value) not in groups:
                        groups[str(value)] = EvaluationAccumulator(self.classes)
                    group = groups[str(value)]
                    group.update(
                        np.asarray(labels)[mask],
                        np.asarray(predictions)[mask],
                        np.asarray(probabilities)[mask],
                    )

    def result(self):
        if not self.matrix.sum():
            raise ValueError("Empty evaluation split")
        report = _summarize(self.matrix, self.loss, self.classes)
        if self.groups:
            report["subgroups"] = {
                c: {v: g.result() for v, g in groups.items()}
                for c, groups in self.groups.items()
            }
        return report


def _summarize(matrix, total_loss, classes):
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    diagonal = matrix.diagonal()
    precision = np.divide(
        diagonal, predicted, out=np.zeros(len(classes)), where=predicted > 0
    )
    recall = np.divide(diagonal, support, out=np.zeros(len(classes)), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(classes)),
        where=precision + recall > 0,
    )
    return {
        "support": int(support.sum()),
        "class_order": list(classes),
        "macro_f1": float(f1.mean()),
        "accuracy": float(diagonal.sum() / support.sum()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "log_loss": float(total_loss / support.sum()),
        "per_class": {
            c: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, c in enumerate(classes)
        },
        "confusion_matrix": matrix.tolist(),
    }


def evaluate(labels, predictions, probabilities, class_order, metadata=None):
    classes = list(class_order)
    y, pred = np.asarray(labels), np.asarray(predictions)
    proba = np.asarray(probabilities, dtype=float)
    if not len(y) or pred.shape != y.shape or proba.shape != (len(y), len(classes)):
        raise ValueError("Invalid or empty evaluation shapes")
    if len(set(classes)) != len(classes) or not set(y).union(pred).issubset(classes):
        raise ValueError("Invalid class order or unknown labels")
    if (
        not np.isfinite(proba).all()
        or (proba < 0).any()
        or (proba > 1).any()
        or not np.allclose(proba.sum(axis=1), 1)
    ):
        raise ValueError("Invalid probabilities")
    matrix = np.array(
        [[np.sum((y == a) & (pred == b)) for b in classes] for a in classes]
    )
    indices = np.array([classes.index(v) for v in y])
    loss = -np.log(
        np.clip(proba[np.arange(len(y)), indices], np.finfo(float).eps, 1)
    ).sum()
    result = _summarize(matrix, loss, classes)
    if metadata is not None:
        if len(metadata) != len(y):
            raise ValueError("Metadata is not aligned with evaluated rows")
        result["subgroups"] = {}
        for column in ("OP_UNIQUE_CARRIER", "ORIGIN"):
            values = metadata[column].astype("string").fillna("<missing>")
            result["subgroups"][column] = {}
            for value in sorted(values.unique()):
                mask = values.eq(value).to_numpy(dtype=bool)
                result["subgroups"][column][str(value)] = evaluate(
                    y[mask], pred[mask], proba[mask], classes
                )
    return result
