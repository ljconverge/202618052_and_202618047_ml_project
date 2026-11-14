"""Importable transformers; keep this module beside train.py and app.py."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def validate_raw(frame, expected):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Sensor input must be a named table.")
    if frame.empty:
        raise ValueError("No sensor records supplied.")
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are not allowed.")
    missing = sorted(set(expected) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(expected))
    if missing or extra:
        raise ValueError(f"Schema mismatch. Missing: {missing}; unexpected: {extra}")
    ordered = frame.loc[:, expected].replace("na", np.nan)
    numeric = ordered.apply(pd.to_numeric, errors="coerce")
    if (ordered.notna() & numeric.isna()).any().any():
        raise ValueError("Invalid nonnumeric sensor values; use blank or 'na' for missing data.")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Infinite sensor values are not accepted.")
    # sklearn trees convert inputs to float32. Reject values that would overflow
    # during engineering or that conversion rather than silently replacing them.
    if (numeric.abs() > 1e30).any().any():
        raise ValueError("Sensor magnitude exceeds the supported numeric range (1e30).")
    if numeric.isna().all(axis=1).any():
        raise ValueError("A record has no observed sensors; request a new measurement.")
    return numeric


class FeatureEngineer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        validate_raw(X, self.feature_names_in_.tolist())
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        raw = validate_raw(X, self.feature_names_in_.tolist())
        result = raw.copy()
        result["fe_missing_count"] = raw.isna().sum(axis=1)
        result["fe_missing_ratio"] = raw.isna().mean(axis=1)
        for prefix in ("ag_", "ay_", "az_", "ba_", "cn_", "cs_", "ee_"):
            cols = [c for c in raw if c.startswith(prefix)]
            if cols:
                # An entirely missing histogram is unknown, rather than zero volume.
                result[f"fe_{prefix}sum"] = raw[cols].sum(axis=1, min_count=1)
                result[f"fe_{prefix}max"] = raw[cols].max(axis=1)
        return result


class MissingnessPruner(TransformerMixin, BaseEstimator):
    def __init__(self, max_missing=0.70):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        if not 0 <= self.max_missing < 1:
            raise ValueError("max_missing must be in [0, 1).")
        self.columns_ = X.columns[X.isna().mean() <= self.max_missing].tolist()
        if not self.columns_:
            raise ValueError("No features survive missingness pruning.")
        return self

    def transform(self, X):
        check_is_fitted(self, "columns_")
        return X.loc[:, self.columns_]
