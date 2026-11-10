"""Configuration generation only; all fitting/CV remains in experiments.py.

Sequential A/B/C uses sklearn ParameterSampler, never validation or a fitted mask.
"""

import copy

from sklearn.model_selection import ParameterGrid, ParameterSampler


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


def sequential(reference, settings, score, seed, record):
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
        if chosen_score is None:
            chosen_score = score(current)
        best = current
        record(
            {
                "model": model,
                "stage": stage,
                "candidate": "reference",
                "state": "completed",
                "config": copy.deepcopy(current),
                **chosen_score,
            }
        )
        count = min(budget, len(ParameterGrid(space)))
        for index, parameters in enumerate(
            ParameterSampler(space, n_iter=count, random_state=seed)
        ):
            candidate = copy.deepcopy(current)
            # Every candidate in a stage starts from the same incoming configuration.
            # current is only updated after all stage candidates have been evaluated.
            for family, value in parameters.items():
                candidate = apply_option(candidate, family, value)
            try:
                result = score(candidate)
            except Exception as error:
                record(
                    {
                        "model": model,
                        "stage": stage,
                        "candidate": index,
                        "state": "failed",
                        "config": candidate,
                        "error": str(error),
                    }
                )
                if not isinstance(error, (ValueError, TypeError)):
                    raise
                continue
            record(
                {
                    "model": model,
                    "stage": stage,
                    "candidate": index,
                    "state": "completed",
                    "config": copy.deepcopy(candidate),
                    **result,
                }
            )
            if rank(result) < rank(chosen_score):
                chosen_score, best = result, candidate
        # Keep the explicit incoming reference if every sampled configuration loses.
        current = best
    return current, chosen_score
