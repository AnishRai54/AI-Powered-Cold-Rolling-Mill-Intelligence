"""Project-defined, non-safety-critical maintenance intelligence helpers."""
from __future__ import annotations

import pandas as pd


def health_score(anomaly_probability: float, reconstruction_error: float | None = None, threshold: float | None = None) -> float:
    """Produce a transparent project-defined 0–100 health indicator."""
    score = 100.0 * (1.0 - float(anomaly_probability))
    if reconstruction_error is not None and threshold and threshold > 0:
        score -= min(25.0, 25.0 * reconstruction_error / threshold)
    return round(max(0.0, min(100.0, score)), 1)


def risk_priority(score: float) -> str:
    if score <= 30: return "CRITICAL"
    if score <= 60: return "HIGH"
    if score <= 80: return "MEDIUM"
    return "LOW"


def recommended_area(fault: str) -> str:
    mapping = {"Electric": "Drive motor, power supply, and electrical condition", "Bearing": "Bearing temperature, lubrication, and vibration",
               "Work Roll": "Work-roll wear, surface condition, and roll change history", "Reduction": "Reduction schedule, gauges, and strip thickness control"}
    return mapping.get(fault, "Review process trend and inspect the highest-contributing variables")


def alerts_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    alerts = frame.loc[frame.get("anomaly_present", pd.Series(False, index=frame.index)).astype(bool)].copy()
    if alerts.empty: return pd.DataFrame(columns=["record", "severity", "fault", "investigation"])
    alerts["severity"] = "HIGH"
    alerts["investigation"] = alerts["fault_family"].map(recommended_area)
    return alerts.reset_index(names="record")[["record", "severity", "fault_family", "investigation"]]
