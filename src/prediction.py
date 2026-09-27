"""Model loading and single-record inference."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.maintenance import health_score, recommended_area, risk_priority
from src.preprocessing import build_features


def load_artifacts(model_dir: str = "models") -> tuple[Any, dict[str, Any], Any | None]:
    root = Path(model_dir)
    classifier = joblib.load(root / "classification" / "anomaly_classifier.joblib")
    metadata = json.loads((root / "metadata" / "model_metadata.json").read_text(encoding="utf-8"))
    ae_path = root / "deep_learning" / "mlp_autoencoder.joblib"
    return classifier, metadata, joblib.load(ae_path) if ae_path.exists() else None


def predict_record(record: dict[str, Any], model_dir: str = "models") -> dict[str, Any]:
    """Infer safely after validating the serialized model feature contract."""
    classifier, metadata, autoencoder = load_artifacts(model_dir)
    frame = build_features(pd.DataFrame([record])).reindex(columns=metadata["features"])
    if frame.isna().all(axis=1).any():
        raise ValueError("Input did not contain usable model features.")
    probability = float(classifier.predict_proba(frame)[0, list(classifier.classes_).index(1)])
    anomaly = probability >= .5
    error = None
    if autoencoder:
        transformed = autoencoder["preprocessor"].transform(frame)
        transformed = transformed.toarray() if hasattr(transformed, "toarray") else transformed
        error = float(autoencoder["detector"].score_samples(transformed)[0])
    score = health_score(probability, error, metadata.get("autoencoder", {}).get("threshold"))
    fault = "Normal operating pattern"
    if anomaly:
        fault_path = Path(model_dir) / "classification" / "fault_family_classifier.joblib"
        fault = str(joblib.load(fault_path).predict(frame)[0]) if fault_path.exists() else "Anomaly investigation required"
    return {"prediction": "ANOMALY" if anomaly else "NORMAL", "anomaly_probability": probability, "anomaly_score": error,
            "health_score": score, "priority": risk_priority(score), "likely_fault": fault,
            "recommended_investigation": recommended_area("Normal" if not anomaly else fault)}
