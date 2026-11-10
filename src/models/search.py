"""YAML search spaces and staged native GridSearchCV on training folds."""

import copy

import numpy as np
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, ParameterGrid, ParameterSampler
from threadpoolctl import threadpool_limits

from src.data.preprocessor import CLASSES
from src.models.factory import create_model


def apply_option(reference, family, value):
    result = copy.deepcopy(reference)
    if family == "normalization":
        result["scaling"] = value
    elif family == "imbalance":
        sampling = dict(value)
        method = sampling.pop("method")
        result.setdefault("params", {})["class_weight"] = sampling.pop(
            "class_weight", None
        )
        result["sampling"] = dict(
            sampling, method="passthrough" if method == "weights" else method
        )
    elif family.startswith("params."):
        result.setdefault("params", {})[family.removeprefix("params.")] = value
    elif family in (
        "outliers",
        "missing",
        "binning",
        "encoding",
        "engineering",
        "tsfresh",
        "selection",
    ):
        result.setdefault("features", {})[family] = copy.deepcopy(value)
    else:
        raise ValueError(f"Unknown experiment family: {family}")
    return result


def ablations(settings):
    reference = copy.deepcopy(settings["reference"])
    if not settings.get("families"):
        raise ValueError("Ablations need explicit families and a reference")
    experiments = [reference]
    for family, variants in settings["families"].items():
        for index, value in enumerate(variants):
            candidate = apply_option(reference, family, value)
            candidate["name"] = (
                f"{reference['name']}_{family.replace('.', '_')}_{index + 1}"
            )
            candidate["ablation_family"] = family
            experiments.append(candidate)
    return experiments


def rank(score):
    return (-score["cv_macro_f1"], score["cv_log_loss"])


def _default_spaces(model):
    return {
        "A": {
            "outliers": [
                {"method": "passthrough"},
                {"method": "zscore", "threshold": 3.0},
            ],
            "missing": [
                {"method": "simple", "strategy": "median"},
                {"method": "knn", "n_neighbors": 3},
            ],
            "normalization": ["standard", "robust", "minmax", "passthrough"],
            "binning": [
                {"enabled": False},
                {
                    "enabled": True,
                    "columns": ["DISTANCE"],
                    "n_bins": 5,
                    "encode": "onehot",
                },
            ],
            "encoding": ["passthrough", "onehot"],
            "engineering": [
                {"cyclic": False, "interactions": False},
                {"cyclic": True, "interactions": True},
            ],
        },
        "B": {
            "selection": [
                {"method": "passthrough"},
                {"method": "kbest", "k": 5},
                {"method": "percentile", "percentile": 50},
            ]
        },
        "C": {
            **(
                {"params.C": [0.1, 1.0, 10.0]}
                if model == "logistic"
                else {
                    "params.n_estimators": [50, 100],
                    "params.num_leaves": [15, 31],
                    "params.learning_rate": [0.05, 0.1],
                }
            ),
            "imbalance": [
                {"method": "weights", "class_weight": None},
                {"method": "weights", "class_weight": "balanced"},
                {"method": "over", "strategy": "auto"},
                {"method": "under", "strategy": "auto"},
            ],
        },
    }


def sequential(reference, settings, preprocessing, X, y, folds, seed, threads, record):
    """Native random candidates per stage; propagate configurations, never models."""
    current = copy.deepcopy(reference)
    stages = settings.get("stages", ["A", "B", "C"])
    if (
        not stages
        or len(set(stages)) != len(stages)
        or stages != sorted(stages)
        or set(stages) - {"A", "B", "C"}
    ):
        raise ValueError("Stages must be an ordered subset of A, B, C")
    allowed = {
        "A": {
            "outliers",
            "missing",
            "normalization",
            "binning",
            "encoding",
            "engineering",
            "tsfresh",
        },
        "B": {"selection"},
    }
    model = reference["model"]
    spaces = _default_spaces(model)
    chosen_score = None
    for stage in stages:
        space = settings.get("spaces", {}).get(stage, spaces[stage])
        if (
            not space
            or any(not isinstance(v, list) or not v for v in space.values())
            or (stage in allowed and set(space) - allowed[stage])
            or (
                stage == "C"
                and any(k != "imbalance" and not k.startswith("params.") for k in space)
            )
        ):
            raise ValueError(f"Invalid search space for stage {stage}")
        budget = settings.get("budget", 10)
        budget = budget.get(stage, 10) if isinstance(budget, dict) else budget
        if not isinstance(budget, int) or budget < 1:
            raise ValueError("Search budget must be a positive integer")
        # Translate the YAML alternatives once. SearchCV receives native pipeline
        # step parameters; correlated options (e.g. engineering + column routing)
        # stay together in each parameter dictionary.
        candidates = [copy.deepcopy(current)]
        count = min(budget, len(ParameterGrid(space)))
        for parameters in ParameterSampler(space, n_iter=count, random_state=seed):
            candidate = copy.deepcopy(current)
            for family, value in parameters.items():
                candidate = apply_option(candidate, family, value)
            candidates.append(candidate)
        distributions = []
        base_pipeline = None
        for candidate in candidates:
            pipeline, _ = create_model(candidate, preprocessing, seed, threads, CLASSES)
            parameters = {name: [step] for name, step in pipeline.steps}
            distributions.append(parameters)
            # An imbalanced-learn pipeline also accepts passthrough sampling;
            # a sklearn pipeline cannot host a fit_resample-only step.
            if (
                base_pipeline is None
                or pipeline.named_steps["sampling"] != "passthrough"
            ):
                base_pipeline = pipeline
        # Include the incoming reference, so at least one valid candidate remains
        # even if every random alternative fails. sklearn handles cloning and CV.
        search = GridSearchCV(
            base_pipeline,
            distributions,
            scoring={
                "macro_f1": make_scorer(
                    f1_score,
                    labels=sorted(CLASSES),
                    average="macro",
                    zero_division=0,
                    pos_label=None,
                ),
                "neg_log_loss": "neg_log_loss",
            },
            cv=folds,
            refit=False,
            n_jobs=1,
            error_score=np.nan,
        )
        with threadpool_limits(limits=threads):
            search.fit(X, y)
        cv = search.cv_results_
        evaluations = []
        for index, params in enumerate(cv["params"]):
            candidate = candidates[index]
            f1 = float(cv["mean_test_macro_f1"][index])
            loss = float(-cv["mean_test_neg_log_loss"][index])
            entry = {
                "model": model,
                "stage": stage,
                "candidate": "reference" if index == 0 else index - 1,
                "config": copy.deepcopy(candidate),
            }
            if not np.isfinite([f1, loss]).all():
                record(
                    {
                        **entry,
                        "state": "failed",
                        "error": "Nonfinite SearchCV scores; see sklearn fit warnings",
                    }
                )
                continue
            result = {
                "cv_macro_f1": f1,
                "cv_log_loss": loss,
                "mean_fit_seconds": float(cv["mean_fit_time"][index]),
                "folds": [
                    {
                        "fold": fold,
                        "train_rows": len(a),
                        "validation_rows": len(b),
                        "train_end": str(X.iloc[a].FL_DATE.max()),
                        "validation_start": str(X.iloc[b].FL_DATE.min()),
                        "metrics": {
                            "macro_f1": float(cv[f"split{fold}_test_macro_f1"][index]),
                            "log_loss": float(
                                -cv[f"split{fold}_test_neg_log_loss"][index]
                            ),
                        },
                    }
                    for fold, (a, b) in enumerate(folds)
                ],
            }
            record({**entry, "state": "completed", **result})
            evaluations.append((candidate, result))
        if not evaluations:
            raise ValueError(f"No valid search candidates for stage {stage}")
        current, chosen_score = min(evaluations, key=lambda item: rank(item[1]))
    return current, chosen_score
