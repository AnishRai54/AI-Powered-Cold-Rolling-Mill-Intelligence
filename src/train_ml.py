"""Reproducible supervised and reconstruction-model training entrypoint."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.anomaly_detection import ReconstructionAutoencoder
from src.data_loader import load_unified_data, profile_dataset, write_profile
from src.preprocessing import build_features, derive_targets, make_preprocessor

LOGGER = logging.getLogger(__name__)
RANDOM_STATE = 42


def chronological_split(df: pd.DataFrame, test_size: float = .20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the latest batch/order, avoiding a random future-information split."""
    ordered = df.sort_values([c for c in ("batch_id", "batch_row") if c in df]).reset_index(drop=True)
    cut = int(len(ordered) * (1 - test_size))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def candidate_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=600, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=220, min_samples_leaf=2, n_jobs=-1, class_weight="balanced_subsample", random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesClassifier(n_estimators=260, min_samples_leaf=1, n_jobs=-1, class_weight="balanced", random_state=RANDOM_STATE),
    }
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(n_estimators=250, learning_rate=.06, num_leaves=31, subsample=.85,
                                              colsample_bytree=.85, class_weight="balanced", random_state=RANDOM_STATE,
                                              verbosity=-1, n_jobs=-1)
    except ImportError:
        LOGGER.info("LightGBM not installed; skipped.")
    return models


def _binary_metrics(y_true: pd.Series, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    result = {
        "accuracy": accuracy_score(y_true, predictions), "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0), "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0), "macro_f1": f1_score(y_true, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, predictions, average="weighted", zero_division=0),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = roc_auc_score(y_true, probabilities)
        result["pr_auc"] = average_precision_score(y_true, probabilities)
    return {key: round(float(value), 5) for key, value in result.items()}


def _importance(model: Pipeline, features: list[str]) -> list[dict[str, float | str]]:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_).ravel()
    else:
        return []
    names = model.named_steps["preprocess"].get_feature_names_out()
    rows = sorted(({"feature": str(name).replace("numeric__", "").replace("categorical__", ""), "importance": float(value)} for name, value in zip(names, values)), key=lambda x: x["importance"], reverse=True)
    return rows[:30]


def train_pipeline(data_dir: str = "data/raw", model_dir: str = "models", test_size: float = .20,
                   enable_cv: bool = True, sample_for_cv: int = 30000) -> dict[str, Any]:
    """Train, evaluate, serialize and document models using the discovered labels."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    raw = load_unified_data(data_dir)
    enriched = derive_targets(raw)
    features = build_features(enriched)
    train_df, test_df = chronological_split(enriched, test_size)
    x_train, x_test = features.loc[train_df.index], features.loc[test_df.index]
    y_train, y_test = train_df["anomaly_present"], test_df["anomaly_present"]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Chronological holdout does not contain both classes; supervised binary evaluation is not valid.")

    comparison: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    for name, estimator in candidate_models().items():
        pipeline = Pipeline([("preprocess", make_preprocessor(x_train)), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, list(pipeline.classes_).index(1)]
        predictions = pipeline.predict(x_test)
        metrics = _binary_metrics(y_test, probabilities, predictions)
        record = {"model": name, **metrics}
        if enable_cv:
            cv_sample = min(sample_for_cv, len(x_train))
            subset = np.linspace(0, len(x_train) - 1, cv_sample, dtype=int)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_val_score(pipeline, x_train.iloc[subset], y_train.iloc[subset], scoring="f1", cv=cv, n_jobs=1)
            record["cv_f1_mean"] = round(float(scores.mean()), 5)
            record["cv_f1_std"] = round(float(scores.std()), 5)
        comparison.append(record)
        fitted[name] = pipeline
        LOGGER.info("Trained %s: F1 %.4f", name, metrics["f1"])
    comparison.sort(key=lambda row: (row["f1"], row["pr_auc"]), reverse=True)
    best_name = comparison[0]["model"]
    best = fitted[best_name]

    class_path = Path(model_dir) / "classification" / "anomaly_classifier.joblib"
    class_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, class_path)
    (Path(model_dir) / "classification" / "feature_names.json").write_text(json.dumps(features.columns.tolist(), indent=2), encoding="utf-8")

    # Multiple supplied fault families justify a separate multiclass classifier.
    fault_train = train_df.loc[train_df["anomaly_present"].eq(1)]
    fault_test = test_df.loc[test_df["anomaly_present"].eq(1)]
    fault_result: dict[str, Any] = {"available": False}
    if fault_train["fault_family"].nunique() > 1 and fault_test["fault_family"].nunique() > 1:
        fault_model = Pipeline([
            ("preprocess", make_preprocessor(x_train)),
            ("model", ExtraTreesClassifier(n_estimators=220, min_samples_leaf=1, n_jobs=-1,
                                             class_weight="balanced", random_state=RANDOM_STATE)),
        ])
        fault_model.fit(features.loc[fault_train.index], fault_train["fault_family"])
        fault_predictions = fault_model.predict(features.loc[fault_test.index])
        fault_result = {
            "available": True, "model": "Extra Trees multiclass fault-family classifier",
            "classes": sorted(fault_train["fault_family"].unique().tolist()),
            "accuracy": round(float(accuracy_score(fault_test["fault_family"], fault_predictions)), 5),
            "macro_f1": round(float(f1_score(fault_test["fault_family"], fault_predictions, average="macro", zero_division=0)), 5),
            "weighted_f1": round(float(f1_score(fault_test["fault_family"], fault_predictions, average="weighted", zero_division=0)), 5),
            "report": classification_report(fault_test["fault_family"], fault_predictions, output_dict=True, zero_division=0),
        }
        joblib.dump(fault_model, Path(model_dir) / "classification" / "fault_family_classifier.joblib")

    # MLP reconstruction autoencoder uses a sample of normal records to stay laptop-friendly.
    preprocessor = make_preprocessor(x_train)
    normal_train = x_train.loc[y_train.eq(0)]
    normal_test = x_test.loc[y_test.eq(0)]
    ae_preprocessor = preprocessor.fit(normal_train)
    train_matrix = ae_preprocessor.transform(normal_train)
    valid_matrix = ae_preprocessor.transform(normal_test)
    train_matrix = train_matrix.toarray() if hasattr(train_matrix, "toarray") else train_matrix
    valid_matrix = valid_matrix.toarray() if hasattr(valid_matrix, "toarray") else valid_matrix
    max_normal = min(18000, len(train_matrix))
    autoencoder = ReconstructionAutoencoder(RANDOM_STATE).fit(train_matrix[:max_normal], valid_matrix[:min(8000, len(valid_matrix))])
    test_matrix = ae_preprocessor.transform(x_test)
    test_matrix = test_matrix.toarray() if hasattr(test_matrix, "toarray") else test_matrix
    ae_scores = autoencoder.score_samples(test_matrix)
    ae_predictions = autoencoder.predict(test_matrix)
    ae_metrics = _binary_metrics(y_test, ae_scores, ae_predictions)
    ae_dir = Path(model_dir) / "deep_learning"
    ae_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessor": ae_preprocessor, "detector": autoencoder}, ae_dir / "mlp_autoencoder.joblib")

    profile = profile_dataset(raw)
    write_profile(profile)
    best_prob = best.predict_proba(x_test)[:, list(best.classes_).index(1)]
    output = {
        "training_date_utc": datetime.now(UTC).isoformat(), "dataset_version": ", ".join(profile["source_files"]),
        "validation_strategy": "Chronological batch/order holdout: last 20% held out; 3-fold stratified CV only on training subset.",
        "training_parameters": {"random_state": RANDOM_STATE, "holdout_fraction": test_size,
                                "classification_threshold": 0.5, "cross_validation_enabled": enable_cv,
                                "cv_sample_cap": sample_for_cv, "autoencoder_architecture": [64, 16, 64],
                                "autoencoder_normal_training_cap": 18000, "autoencoder_threshold_quantile": 0.995},
        "target": "anomaly_present", "fault_label": "fault_family", "features": features.columns.tolist(),
        "rows": len(raw), "train_rows": len(train_df), "test_rows": len(test_df), "class_distribution": {str(k): int(v) for k, v in enriched["anomaly_present"].value_counts().items()},
        "comparison": comparison, "best_model": best_name, "best_metrics": comparison[0],
        "fault_classification": fault_result,
        "classification_report": classification_report(y_test, (best_prob >= .5).astype(int), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, (best_prob >= .5).astype(int)).tolist(), "feature_importance": _importance(best, features.columns.tolist()),
        "autoencoder": {"model": "MLP reconstruction autoencoder", "threshold": autoencoder.threshold_, "metrics": ae_metrics,
                        "normal_training_rows": int(min(max_normal, len(train_matrix))), "note": "No timestamp exists, so LSTM/GRU sequence modelling was not scientifically supported."},
    }
    metadata_path = Path(model_dir) / "metadata" / "model_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    enriched.to_parquet("data/processed/cold_rolling_unified.parquet", index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cold rolling anomaly models.")
    parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation.")
    parser.add_argument("--test-size", type=float, default=.20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = train_pipeline(test_size=args.test_size, enable_cv=not args.no_cv)
    print(json.dumps({"best_model": result["best_model"], "best_metrics": result["best_metrics"], "rows": result["rows"]}, indent=2))


if __name__ == "__main__":
    main()
