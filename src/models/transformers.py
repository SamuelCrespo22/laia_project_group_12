"""Serializable, record-local features and train-only contextual statistics."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from src.data.contract import SPECS, _valid_mask
from src.data.preprocessor import build_features


def validate_scheduled_features(X):
    """Apply contract ranges; retain rows and replace nonpositive duration by null.

    Required departure time cannot be null. Optional schedule fields may be null.
    Impossible HHMM, negative distance and any infinite numeric input fail early.
    Nonpositive scheduled duration is an unusable source value, not a flight to drop.
    No finite upper cutoff is imposed on valid durations, distance or delay labels.
    """
    result = X.copy()
    for column in ("CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DISTANCE"):
        if column not in result:
            if SPECS[column].request == "required":
                raise ValueError(f"Missing required scheduled feature: {column}")
            continue
        values = pd.to_numeric(result[column], errors="raise").astype("Float64")
        present = values.dropna()
        if not np.isfinite(present.astype(float)).all():
            raise ValueError(f"Nonfinite scheduled feature: {column}")
        spec = SPECS[column]
        if spec.request == "required" and values.isna().any():
            raise ValueError(f"Null required scheduled feature: {column}")
        if column == "CRS_ELAPSED_TIME":
            values = values.where(values > 0, pd.NA)
        elif not _valid_mask(spec, present).all() or (
            spec.hhmm and not present.mod(1).eq(0).all()
        ):
            raise ValueError(f"Invalid scheduled feature: {column}")
        result[column] = values
    return result


class FlightFeatures(TransformerMixin, BaseEstimator):
    """Thin adapter; the frozen preprocessing config travels with the artifact."""

    def __init__(self, config, history_column=None):
        self.config = config
        self.history_column = history_column

    def fit(self, X, y=None):
        return self

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault("history_column", None)

    def transform(self, X):
        result = build_features(validate_scheduled_features(X), self.config)
        history_column = getattr(self, "history_column", None)
        if history_column:
            if history_column not in X:
                raise ValueError(
                    f"Causal tsfresh requires explicit history: {history_column}"
                )
            result[history_column] = X[history_column]
        return result


class ZScoreOutliers(TransformerMixin, BaseEstimator):
    """Train-fitted StandardScaler z-scores mark cells missing, preserving rows."""

    def __init__(self, columns=("DISTANCE", "CRS_ELAPSED_TIME"), threshold=3.0):
        self.columns = columns
        self.threshold = threshold

    def fit(self, X, y=None):
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("Z-score threshold must be positive")
        self.scaler_ = StandardScaler().fit(X[list(self.columns)].astype(float))
        return self

    def transform(self, X):
        result = X.copy()
        values = X[list(self.columns)].astype(float)
        scores = self.scaler_.transform(values)
        for i, column in enumerate(self.columns):
            result[column] = values[column].mask(np.abs(scores[:, i]) > self.threshold)
        return result


class TsfreshFeatures(TransformerMixin, BaseEstimator):
    """Record-local actual observations strictly before scheduled departure.

    Each history is a list of {timestamp, value}; no cross-flight state or target.
    BTS has no such source, so enabling this on ordinary BTS fails explicitly.
    """

    STATISTICS = ("mean", "standard_deviation", "minimum", "maximum", "length")

    def __init__(
        self, history_column="history", statistics=("mean", "standard_deviation")
    ):
        self.history_column = history_column
        self.statistics = statistics

    def fit(self, X, y=None):
        if not self.statistics or not set(self.statistics).issubset(self.STATISTICS):
            raise ValueError("Unsupported tsfresh statistics")
        return self

    def transform(self, X):
        from tsfresh import extract_features

        if self.history_column not in X:
            raise ValueError(
                "tsfresh needs real per-record history; BTS does not provide it"
            )
        rows = []
        for i, (_, record) in enumerate(X.iterrows()):
            hhmm = int(record["CRS_DEP_TIME"])
            cutoff = record["FL_DATE"].normalize() + pd.Timedelta(
                minutes=(hhmm // 100) * 60 + hhmm % 100
            )
            history = record[self.history_column]
            if history is None or history is pd.NA:
                continue
            if not isinstance(history, (list, tuple, np.ndarray)):
                raise ValueError("History must be an explicit sequence of observations")
            for point in history:
                timestamp = pd.Timestamp(point["timestamp"])
                value = float(point["value"])
                if (
                    timestamp.tz is not None
                    or pd.isna(timestamp)
                    or timestamp >= cutoff
                    or not np.isfinite(value)
                ):
                    raise ValueError(
                        "History must be finite and strictly before departure"
                    )
                rows.append({"id": i, "time": timestamp, "value": value})
        result = X.copy()
        extracted = None
        if rows:
            extracted = extract_features(
                pd.DataFrame(rows),
                column_id="id",
                column_sort="time",
                column_value="value",
                default_fc_parameters={s: None for s in self.statistics},
                n_jobs=0,
                disable_progressbar=True,
                impute_function=None,
            )
        for statistic in self.statistics:
            values = (
                extracted["value__" + statistic].reindex(range(len(X))).to_numpy()
                if extracted is not None
                else np.full(len(X), np.nan)
            )
            result["tsfresh_" + statistic] = values
        return result


class OperationalFeatures(TransformerMixin, BaseEstimator):
    """Optional deterministic cyclic and scheduled speed features."""

    def __init__(self, cyclic=False, interactions=False):
        self.cyclic = cyclic
        self.interactions = interactions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()
        if self.cyclic:
            for name in ("CRS_DEP_TIME", "CRS_ARR_TIME"):
                value = pd.to_numeric(X[name]).astype(float)
                minutes = ((value // 100) % 24) * 60 + value % 100
                result[name + "_sin"] = np.sin(2 * np.pi * minutes / 1440)
                result[name + "_cos"] = np.cos(2 * np.pi * minutes / 1440)
            for name, period in [("MONTH", 12), ("DAY_OF_WEEK", 7)]:
                phase = 2 * np.pi * (X[name].astype(float) - 1) / period
                result[name + "_sin"] = np.sin(phase)
                result[name + "_cos"] = np.cos(phase)
        if self.interactions:
            duration = X["CRS_ELAPSED_TIME"].astype(float)
            result["scheduled_speed"] = X["DISTANCE"].astype(float) / duration.where(
                duration > 0
            )
        return result


class RouteIQR(TransformerMixin, BaseEstimator):
    """Learn route IQR bounds on train; flag/deviation only, never drop rows."""

    def __init__(
        self,
        columns=("DISTANCE", "CRS_ELAPSED_TIME"),
        min_group_size=20,
        factor=1.5,
        action="flag",
    ):
        self.columns = columns
        self.min_group_size = min_group_size
        self.factor = factor
        self.action = action

    def fit(self, X, y=None):
        if self.action not in ("flag", "deviation"):
            raise ValueError("outlier action must be flag or deviation")
        if self.min_group_size < 1 or self.factor < 0:
            raise ValueError("Invalid IQR settings")
        self.global_ = {}
        self.routes_ = {}
        for column in self.columns:
            values = X[column].astype(float)
            if not np.isfinite(values.dropna()).all():
                raise ValueError(f"Nonfinite IQR input: {column}")
            if not values.notna().any():
                raise ValueError(f"No train observations for IQR: {column}")
            self.global_[column] = self._stats(values)
            self.routes_[column] = {}
            for key, group in X.groupby(["ORIGIN", "DEST"], dropna=True):
                if group[column].notna().sum() >= self.min_group_size:
                    self.routes_[column][key] = self._stats(group[column].astype(float))
        return self

    def _stats(self, values):
        q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
        return (q1 - self.factor * (q3 - q1), median, q3 + self.factor * (q3 - q1))

    def transform(self, X):
        result = X.copy()
        for column in self.columns:
            stats = [
                self.routes_[column].get((o, d), self.global_[column])
                for o, d in zip(X["ORIGIN"], X["DEST"])
            ]
            low, median, high = np.asarray(stats).reshape(-1, 3).T
            values = X[column].astype(float).to_numpy()
            if np.isinf(values).any():
                raise ValueError(f"Nonfinite IQR input: {column}")
            outside = (values < low) | (values > high)
            result[column + "_route_" + self.action] = (
                outside.astype(float)
                if self.action == "flag"
                else np.where(outside, values - median, 0.0)
            )
        return result
