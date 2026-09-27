"""Optional SHAP explanations with a safe native-importance fallback."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def shap_values_for_tree_pipeline(pipeline: Any, x: pd.DataFrame, max_rows: int = 300) -> tuple[np.ndarray, list[str]]:
    """Calculate TreeSHAP values for a fitted sklearn tree pipeline.

    SHAP is an optional runtime dependency so the application remains usable in
    constrained environments. Callers should display an install message on ImportError.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install the optional 'shap' dependency to calculate SHAP values.") from exc
    sample = x.head(max_rows)
    transformed = pipeline.named_steps["preprocess"].transform(sample)
    transformed = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    values = explainer.shap_values(transformed)
    if isinstance(values, list): values = values[-1]
    return np.asarray(values), pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
