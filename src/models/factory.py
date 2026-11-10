"""Build native pipelines from an estimator and preprocessing settings."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
    KBinsDiscretizer,
    OneHotEncoder,
)
import pandas as pd

from src.data.contract import SPECS
from src.models.transformers import (
    FlightFeatures,
    OperationalFeatures,
    RouteIQR,
    ZScoreOutliers,
    TsfreshFeatures,
)


def _numeric_branch(config, options):
    missing = options.get("missing", {})
    method = missing.get("method", "simple")
    if method == "simple":
        imputer = SimpleImputer(
            strategy=missing.get("strategy", "median"), keep_empty_features=True
        )
    elif method == "knn":
        imputer = KNNImputer(
            n_neighbors=missing.get("n_neighbors", 5),
            weights=missing.get("weights", "uniform"),
            keep_empty_features=True,
        )
    else:
        raise ValueError(f"Unknown missing-value method: {method}")
    scaling = config.get("scaling", "passthrough")
    scalers = {
        "standard": StandardScaler,
        "robust": RobustScaler,
        "minmax": MinMaxScaler,
    }
    if scaling != "passthrough" and scaling not in scalers:
        raise ValueError(f"Unknown scaling: {scaling}")
    return [
        ("imputer", imputer),
        ("scaler", scalers[scaling]() if scaling != "passthrough" else "passthrough"),
    ]


def _outliers(options, preprocessing, categorical):
    method = options.get(
        "method", "legacy_iqr" if options.get("enabled") else "passthrough"
    )
    if method == "passthrough":
        return "passthrough"
    if method == "zscore":
        return ZScoreOutliers(
            options.get("columns", ["DISTANCE", "CRS_ELAPSED_TIME"]),
            options.get("threshold", 3.0),
        )
    if method == "legacy_iqr":
        if categorical:
            raise ValueError(
                "RouteIQR is legacy-only; use zscore or passthrough in modular"
            )
        transformer = RouteIQR(
            **{k: v for k, v in options.items() if k not in ("enabled", "method")}
        )
        if any(
            c not in preprocessing.feature_allowlist
            or SPECS[c].kind not in ("int", "float")
            for c in transformer.columns
        ):
            raise ValueError("IQR requires allowed numeric columns")
        return transformer
    raise ValueError(f"Unknown outlier method: {method}")


def _column_transformer(config, preprocessing, outliers, tsfresh, seed):
    """Route each feature once: numeric, binned numeric, or categorical."""
    options = config.get("features", {})
    categorical = options.get("mode", "minimal_numeric") == "modular"
    columns = list(
        options.get(
            "columns",
            [
                "MONTH",
                "DAY_OF_WEEK",
                "CRS_DEP_TIME",
                "CRS_ARR_TIME",
                "CRS_ELAPSED_TIME",
                "DISTANCE",
                "DISTANCE_GROUP",
            ],
        )
    )
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Numeric features must be nonempty and unique")
    for column in columns:
        if (
            column not in preprocessing.feature_allowlist
            or SPECS[column].kind not in ("int", "float")
            or not SPECS[column].pre_departure
        ):
            raise ValueError(f"Not an allowed numeric feature: {column}")
    if isinstance(outliers, ZScoreOutliers) and not set(outliers.columns).issubset(
        columns
    ):
        raise ValueError("Z-score requires selected numeric columns")
    engineering = options.get("engineering", {})
    if engineering.get("cyclic"):
        columns += [
            name + suffix
            for name in ("CRS_DEP_TIME", "CRS_ARR_TIME", "MONTH", "DAY_OF_WEEK")
            for suffix in ("_sin", "_cos")
        ]
    if engineering.get("interactions"):
        columns.append("scheduled_speed")
    if isinstance(outliers, RouteIQR):
        columns += [c + "_route_" + outliers.action for c in outliers.columns]
    if tsfresh != "passthrough":
        columns += ["tsfresh_" + name for name in tsfresh.statistics]
    binning = options.get("binning", {})
    binned = list(binning.get("columns", [])) if binning.get("enabled") else []
    if len(set(binned)) != len(binned) or not set(binned).issubset(columns):
        raise ValueError("Binning columns must be unique selected numeric features")
    branches = []
    plain = [c for c in columns if c not in binned]
    if plain:
        branches.append(("numeric", Pipeline(_numeric_branch(config, options)), plain))
    if binned:
        branch = _numeric_branch(config, options)
        branch.append(
            (
                "bins",
                KBinsDiscretizer(
                    n_bins=binning.get("n_bins", 5),
                    encode=binning.get("encode", "onehot"),
                    strategy=binning.get("strategy", "quantile"),
                    subsample=None,
                    random_state=seed,
                ),
            )
        )
        branches.append(("binned", Pipeline(branch), binned))
    encoding = options.get("encoding", "onehot" if categorical else "passthrough")
    if encoding not in ("onehot", "passthrough"):
        raise ValueError(f"Unknown encoding: {encoding}")
    if not categorical and encoding != "passthrough":
        raise ValueError(
            "Use modular mode for categorical encoding; minimal_numeric stays numeric"
        )
    if encoding == "onehot":
        categories = options.get(
            "categorical_columns", ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]
        )
        if any(
            c not in preprocessing.feature_allowlist
            or SPECS[c].kind not in ("code", "text")
            for c in categories
        ):
            raise ValueError("Encoding requires allowed categorical columns")
        branch = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="__MISSING__",
                        missing_values=pd.NA,
                        keep_empty_features=True,
                    ),
                ),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )
        branches.append(("categorical", branch, list(categories)))
    return ColumnTransformer(branches, remainder="drop")


def _selection(options):
    method = options.get("method", "kbest" if options.get("enabled") else "passthrough")
    if method == "kbest":
        return SelectKBest(f_classif, k=options.get("k", "all"))
    if method == "percentile":
        return SelectPercentile(f_classif, percentile=options.get("percentile", 50))
    if method == "passthrough":
        return "passthrough"
    raise ValueError(f"Unknown selector: {method}")


def create_estimator(name, params, seed, threads):
    """Resolve a named experiment into a fresh sklearn-compatible estimator."""
    params = dict(params, random_state=seed)
    if name == "dummy":
        return DummyClassifier(**params)
    params["n_jobs"] = threads
    if name == "logistic":
        return LogisticRegression(**params)
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("LightGBM required: install requirements.txt") from exc
        return LGBMClassifier(**params)
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(estimator, config, preprocessing, seed=42):
    """Compose preprocessing and an already-created estimator; never inspect it.

    Native nested parameters are available through set_params, for example
    model__C or columns__numeric__imputer__n_neighbors. All fitted state stays
    inside the returned sklearn (or imbalanced-learn) Pipeline.
    """
    options = config.get("features", {})
    mode = options.get("mode", "minimal_numeric")
    if mode not in ("minimal_numeric", "modular"):
        raise ValueError(f"Feature mode unavailable (Issue #4?): {mode}")
    history_options = options.get("tsfresh", {})
    history = (
        history_options.get("history_column", "history")
        if history_options.get("enabled")
        else None
    )
    tsfresh = (
        TsfreshFeatures(
            history,
            tuple(history_options.get("statistics", ["mean", "standard_deviation"])),
        )
        if history
        else "passthrough"
    )
    outliers = _outliers(options.get("outliers", {}), preprocessing, mode == "modular")
    engineering = options.get("engineering", {})
    sampling = config.get("sampling", {})
    method = sampling.get("method", "passthrough")
    sampler = "passthrough"
    pipeline_type = Pipeline
    if method != "passthrough":
        from imblearn.pipeline import Pipeline as SamplingPipeline
        from imblearn.over_sampling import RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler

        samplers = {"over": RandomOverSampler, "under": RandomUnderSampler}
        if method not in samplers:
            raise ValueError(f"Unknown sampling method: {method}")
        sampler = samplers[method](
            random_state=seed, sampling_strategy=sampling.get("strategy", "auto")
        )
        pipeline_type = SamplingPipeline
    engineering = (
        OperationalFeatures(**engineering)
        if any(engineering.values())
        else "passthrough"
    )
    columns = _column_transformer(config, preprocessing, outliers, tsfresh, seed)
    return pipeline_type(
        [
            ("features", FlightFeatures(preprocessing, history)),
            ("outliers", outliers),
            ("engineering", engineering),
            ("tsfresh", tsfresh),
            ("columns", columns),
            ("selection", _selection(options.get("selection", {}))),
            ("sampling", sampler),
            ("model", estimator),
        ]
    )


def create_model(experiment, preprocessing, seed, threads, class_order):
    """Translate a YAML experiment; estimator creation is separate from composition."""
    estimator = create_estimator(
        experiment["model"], experiment.get("params", {}), seed, threads
    )
    config = {
        key: experiment[key]
        for key in ("features", "scaling", "sampling")
        if key in experiment
    }
    config.setdefault(
        "scaling", "standard" if experiment["model"] == "logistic" else "passthrough"
    )
    return build_pipeline(estimator, config, preprocessing, seed), tuple(
        sorted(class_order)
    )
