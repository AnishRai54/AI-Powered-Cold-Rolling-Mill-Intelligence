from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_unified_data
from src.preprocessing import build_features, derive_targets
from src.prediction import load_artifacts, predict_record


def test_data_loading_and_target_derivation():
    frame = load_unified_data()
    enriched = derive_targets(frame)
    assert len(enriched) > 0
    assert {"anomaly_present", "fault_family"}.issubset(enriched.columns)
    assert set(enriched["anomaly_present"].unique()).issubset({0, 1})


def test_features_exclude_labels_and_keep_process_inputs():
    enriched = derive_targets(load_unified_data().head(30))
    features = build_features(enriched)
    assert "anomaly_present" not in features
    assert not any(name.startswith("Anomaly_") for name in features)
    assert "force_1" in features


def test_missing_label_columns_is_actionable():
    with pytest.raises(ValueError, match="No anomaly/fault label"):
        derive_targets(pd.DataFrame({"force_1": [1.0, 2.0]}))


def test_raw_sources_were_copied_not_overwritten():
    sources = list(Path("data/raw").glob("tcm5_dataset_*.csv"))
    assert len(sources) == 6


def test_model_loading_and_invalid_input_are_actionable():
    classifier, metadata, _ = load_artifacts()
    assert classifier is not None
    assert "features" in metadata
    with pytest.raises(ValueError, match="usable model features"):
        predict_record({"not_a_process_feature": 1})
