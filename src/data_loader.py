"""Dataset discovery and safe loading for rolling mill source files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".parquet", ".json"}


def discover_datasets(data_dir: str | Path = "data/raw") -> list[Path]:
    """Return supported local source files without modifying them."""
    root = Path(data_dir)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Read an individual supported tabular dataset."""
    source = Path(path)
    readers = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".parquet": pd.read_parquet,
        ".json": pd.read_json,
    }
    return readers[source.suffix.lower()](source)


def load_unified_data(data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Combine schema-compatible source batches and retain provenance/order."""
    files = discover_datasets(data_dir)
    if not files:
        raise FileNotFoundError("No supported dataset was found in data/raw. Add a CSV, Excel, Parquet, or JSON file.")
    frames: list[pd.DataFrame] = []
    expected: list[str] | None = None
    for batch_id, path in enumerate(files, start=1):
        frame = read_dataset(path).copy()
        if expected is None:
            expected = list(frame.columns)
        if list(frame.columns) != expected:
            raise ValueError(f"Schema mismatch in {path.name}; source batches cannot be safely unified.")
        frame["source_file"] = path.name
        frame["batch_id"] = batch_id
        frame["batch_row"] = range(len(frame))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return JSON-serializable data quality and feature profile."""
    anomaly_columns = [c for c in df.columns if "anomaly" in c.lower() or "fault" in c.lower()]
    timestamp_columns = [c for c in df.columns if any(k in c.lower() for k in ("time", "date", "timestamp"))]
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return {
        "rows": int(len(df)), "columns": int(df.shape[1]), "numeric_features": numeric_columns,
        "categorical_features": categorical_columns, "timestamp_columns": timestamp_columns,
        "anomaly_columns": anomaly_columns, "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()), "source_files": sorted(df["source_file"].unique().tolist()) if "source_file" in df else [],
    }


def write_profile(profile: dict[str, Any], destination: str | Path = "data/processed/dataset_profile.json") -> None:
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text(json.dumps(profile, indent=2), encoding="utf-8")
