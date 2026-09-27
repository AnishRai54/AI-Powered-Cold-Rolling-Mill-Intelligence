"""Target construction, feature engineering and leakage-safe preprocessing."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def anomaly_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if "anomaly" in c.lower() or "fault" in c.lower()]


def derive_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Derive binary event and fault-family labels solely from supplied labels."""
    output = df.copy()
    labels = anomaly_columns(output)
    if not labels:
        raise ValueError("No anomaly/fault label columns were detected; supervised training is unavailable.")
    flags = output[labels].fillna(False).astype(bool)
    output["anomaly_present"] = flags.any(axis=1).astype(int)
    family_map = {
        "reduction": "Reduction", "electric": "Electric", "bearing": "Bearing", "workroll": "Work Roll",
        "work_roll": "Work Roll",
    }
    output["fault_family"] = "Normal"
    # Priority only resolves multi-label records for a single multiclass model; original flags remain intact.
    for token, family in family_map.items():
        columns = [c for c in labels if token in c.lower()]
        if columns:
            output.loc[flags[columns].any(axis=1), "fault_family"] = family
    return output


def build_features(df: pd.DataFrame, drop_columns: Iterable[str] = ()) -> pd.DataFrame:
    """Create physically interpretable aggregate features and remove labels/provenance."""
    remove = set(drop_columns) | set(anomaly_columns(df)) | {"anomaly_present", "fault_family", "source_file", "batch_id", "batch_row"}
    features = df.drop(columns=[c for c in remove if c in df], errors="ignore").copy()
    numeric = features.select_dtypes(include=np.number).columns
    for prefix in ("force_", "torque_", "motor_power_", "roll_speed_", "work_roll_mileage_", "tension_"):
        columns = [c for c in numeric if c.startswith(prefix)]
        if columns:
            clean = prefix.rstrip("_")
            features[f"{clean}_mean"] = features[columns].mean(axis=1)
            features[f"{clean}_max"] = features[columns].max(axis=1)
    if {"thickness_entry", "thickness_exit"}.issubset(features.columns):
        features["total_thickness_reduction"] = features["thickness_entry"] - features["thickness_exit"]
        features["total_reduction_ratio"] = features["total_thickness_reduction"] / features["thickness_entry"].replace(0, np.nan)
    return features.replace([np.inf, -np.inf], np.nan)


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Return a transformer that is fit only inside the training pipeline."""
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ], remainder="drop")
