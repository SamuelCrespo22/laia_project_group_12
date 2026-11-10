"""Feature encoders for the output of ``build_features`` (Issue #4, roadmap 1.4).

``build_feature_encoder`` returns an unfitted ``ColumnTransformer``. The training
pipeline fits it on the training split only and saves it with the classifier, so
the API applies exactly the same encoding.

Rare categories are grouped together. Categories never seen during fitting and
missing values produce a valid output instead of an error. Only standard
scikit-learn classes are used, so the fitted object loads with ``joblib`` without
importing project code.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data.preprocessor import DEFAULT_CONFIG_PATH
from src.data.preprocessor import load_config as load_preprocessing_config

MODEL_TYPES = ("linear", "tree")
# LightGBM treats negative categorical values as missing.
UNKNOWN_CODE = -1


@dataclass(frozen=True)
class FeatureConfig:
    categorical: tuple[str, ...]
    numeric: tuple[str, ...]
    min_frequency: float


def validate_feature_config(
    categorical: Sequence[str],
    numeric: Sequence[str],
    min_frequency: float,
    allowlist: Sequence[str],
) -> FeatureConfig:
    """Reject column lists outside the allowlist and invalid rarity thresholds."""
    categorical, numeric = tuple(categorical), tuple(numeric)
    columns = categorical + numeric
    if not columns:
        raise ValueError("No feature columns configured")
    duplicates = sorted({c for c in columns if columns.count(c) > 1})
    if duplicates:
        raise ValueError(f"Feature columns configured more than once: {duplicates}")
    unknown = [c for c in columns if c not in allowlist]
    if unknown:
        raise ValueError(f"Feature columns not in the allowlist: {unknown}")
    min_frequency = float(min_frequency)
    if not 0 < min_frequency < 1:
        raise ValueError("min_frequency must be a fraction between 0 and 1")
    return FeatureConfig(categorical, numeric, min_frequency)


def load_feature_config(path: Optional[Path | str] = None) -> FeatureConfig:
    """Read the ``features`` block of ``configs/config.yaml``."""
    config_path = Path(path or DEFAULT_CONFIG_PATH)
    features = yaml.safe_load(config_path.read_text(encoding="utf-8"))["features"]
    return validate_feature_config(
        features["categorical"],
        features["numeric"],
        features["min_frequency"],
        load_preprocessing_config(config_path).feature_allowlist,
    )


@lru_cache(maxsize=1)
def default_feature_config() -> FeatureConfig:
    return load_feature_config()


def _categorical_encoder(
    model_type: str, min_frequency: float
) -> OneHotEncoder | OrdinalEncoder:
    if model_type == "linear":
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=min_frequency
        )
    return OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=UNKNOWN_CODE,
        encoded_missing_value=UNKNOWN_CODE,
        min_frequency=min_frequency,
    )


def _numeric_encoder(model_type: str) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median"))]
    if model_type == "linear":
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def build_feature_encoder(
    model_type: str, config: Optional[FeatureConfig] = None
) -> ColumnTransformer:
    """Unfitted encoder for ``"linear"`` or ``"tree"`` models.

    Categorical columns come first in the output and any column not configured
    is dropped. For tree models, the categorical columns are the positions
    ``0..len(config.categorical) - 1`` of the output.
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}: {model_type!r}")
    config = config or default_feature_config()
    return ColumnTransformer(
        [
            (
                "categorical",
                _categorical_encoder(model_type, config.min_frequency),
                list(config.categorical),
            ),
            ("numeric", _numeric_encoder(model_type), list(config.numeric)),
        ],
        remainder="drop",
    )
