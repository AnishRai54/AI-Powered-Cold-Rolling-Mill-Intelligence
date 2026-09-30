"""Shared industrial Streamlit views driven only by local data/model artifacts."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.cards import metric_grid, status_badge
from components.theme import inject_theme as inject_industrial_theme
from src.data_loader import load_unified_data, profile_dataset
from src.maintenance import alerts_from_frame, health_score, recommended_area, risk_priority
from src.preprocessing import build_features, derive_targets
from src.train_ml import train_pipeline


@st.cache_data(show_spinner=False)
def data() -> pd.DataFrame:
    return derive_targets(load_unified_data())


@st.cache_data(show_spinner=False)
def metadata() -> dict[str, Any] | None:
    path = Path("models/metadata/model_metadata.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_resource(show_spinner=False)
def model_bundle() -> tuple[Any, Any] | None:
    classifier_path = Path("models/classification/anomaly_classifier.joblib")
    auto_path = Path("models/deep_learning/mlp_autoencoder.joblib")
    if not classifier_path.exists(): return None
    return joblib.load(classifier_path), joblib.load(auto_path) if auto_path.exists() else None


@st.cache_resource(show_spinner=False)
def fault_model() -> Any | None:
    path = Path("models/classification/fault_family_classifier.joblib")
    return joblib.load(path) if path.exists() else None


def inject_theme() -> None:
    """Backward-compatible theme entry point used by the Streamlit app."""
    inject_industrial_theme()


def require_data() -> pd.DataFrame | None:
    try: return data()
    except Exception as exc:
        st.error(f"Dataset unavailable: {exc}")
        return None


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    return {"Rolling force": [c for c in numeric if c.startswith("force_")], "Torque": [c for c in numeric if c.startswith("torque_")],
            "Roll speed": [c for c in numeric if c.startswith("roll_speed_")], "Tension": [c for c in numeric if c.startswith("tension_")],
            "Motor power": [c for c in numeric if c.startswith("motor_power_")]}


def recent_prediction(df: pd.DataFrame) -> dict[str, Any]:
    row = df.iloc[-1:]
    bundle = model_bundle()
    meta = metadata()
    if not bundle or not meta:
        return {"probability": np.nan, "status": "MODEL NOT TRAINED", "health": np.nan, "priority": "—", "error": np.nan}
    classifier, auto = bundle
    values = build_features(row).reindex(columns=meta["features"])
    probability = float(classifier.predict_proba(values)[0, list(classifier.classes_).index(1)])
    error = np.nan
    if auto:
        matrix = auto["preprocessor"].transform(values)
        matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        error = float(auto["detector"].score_samples(matrix)[0])
    score = health_score(probability, error if not np.isnan(error) else None, meta.get("autoencoder", {}).get("threshold"))
    return {"probability": probability, "status": "ANOMALY" if probability >= .5 else "NORMAL", "health": score, "priority": risk_priority(score), "error": error}


def hero() -> None:
    copy, image = st.columns((1.35, 1), gap="large", vertical_alignment="center")
    with copy:
        st.markdown('''<section class="hero-copy"><div class="eyebrow">Tandem Cold Rolling · Intelligence Layer</div><h1>Cold Rolling Mill AI<br>Command Center</h1><p>Process monitoring, anomaly detection, and maintenance decision support, grounded in the supplied datasets.</p><span class="live-pill"><i class="live-dot"></i> LOCAL MODEL ARTIFACTS</span></section>''', unsafe_allow_html=True)
    with image:
        st.image("assets/images/cold-rolling-mill.jpg", use_container_width=True)
        st.caption("Illustrative manufacturing image · Unsplash. Not a photograph of the supplied data source.")


def page_header(title: str, section: str, description: str) -> None:
    """Render a consistent, compact heading for non-dashboard views."""
    st.markdown(
        f'<header class="page-header"><div class="eyebrow">{section}</div>'
        f'<h1>{title}</h1><p>{description}</p></header>',
        unsafe_allow_html=True,
    )
def _navigate_primary(page: str) -> None:
    """Switch the shell navigation from the dashboard's explicit CTA buttons."""
    st.session_state["primary_navigation"] = page
    st.session_state["workspace_tools"] = "None"


def section_heading(title: str, label: str) -> None:
    """Render a consistent premium section header."""
    st.markdown(f'<div class="section-title"><h3>{title}</h3><span>{label}</span></div><div class="section-rule"></div>', unsafe_allow_html=True)


def polish_chart(figure: go.Figure, height: int = 330) -> go.Figure:
    """Apply the dark command-center visual system to Plotly charts."""
    figure.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(7,16,25,.38)",
                         margin=dict(l=12, r=16, t=52, b=12), font=dict(family="Manrope, sans-serif", color="#c7dce2"),
                         title=dict(font=dict(size=15, color="#eaf8fa"), x=0.02), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    figure.update_xaxes(gridcolor="rgba(139,181,196,.10)", zerolinecolor="rgba(139,181,196,.14)")
    figure.update_yaxes(gridcolor="rgba(139,181,196,.10)", zerolinecolor="rgba(139,181,196,.14)")
    return figure


def friendly_dashboard_view() -> None:
    df = require_data()
    if df is None: return
    hero(); current = recent_prediction(df); profile = profile_dataset(df)
    kpis = [("Mill Health Score", "—" if np.isnan(current["health"]) else f"{current['health']:.1f}/100", "Project-defined health framework"),
            ("Current Anomaly Risk", "—" if np.isnan(current["probability"]) else f"{current['probability']:.1%}", "Model probability, not certainty"),
            ("Predicted Fault", current["status"], "Latest supplied record"), ("Active Labelled Events", str(int(df["anomaly_present"].sum())), "Across loaded batches")]
    cols = st.columns(4)
    for col, (label, value, note) in zip(cols, kpis):
        col.markdown(f'<div class="kpi"><label>{label}</label><strong>{value}</strong><span>{note}</span></div>', unsafe_allow_html=True)
    section_heading("Tandem mill flow", "five-stand process context")
    cards = ['Entry Coil'] + [f'Stand {i}' for i in range(1, 6)] + ['Exit Coil']
    flow = ''.join(f'<div class="stand">{card}<br><small class="muted">dataset-driven</small></div>' + ('' if i == len(cards)-1 else '<div class="arrow">→</div>') for i, card in enumerate(cards))
    st.markdown(f'<div class="process">{flow}</div>', unsafe_allow_html=True)
    view = df.tail(min(800, len(df))).copy(); view["record"] = np.arange(len(view))
    groups = feature_groups(df); available = [c for v in groups.values() for c in v]
    section_heading("Operational signals", "latest 800 observations")
    left, right = st.columns((1.35, 1), gap="large")
    if available:
        trend = px.line(view, x="record", y=available[:min(3,len(available))], title="Recent process-variable trend", color_discrete_sequence=["#64e6de", "#74b8ff", "#ffad5b"])
        left.plotly_chart(polish_chart(trend), use_container_width=True)
    timeline = px.scatter(view, x="record", y="anomaly_present", color="fault_family", title="Labelled anomaly timeline", color_discrete_sequence=["#6fe0a9", "#ffad5b", "#ff7b7b", "#b69cff", "#74b8ff"])
    right.plotly_chart(polish_chart(timeline), use_container_width=True)
    st.caption(f"Loaded {profile['rows']:,} records from {len(profile['source_files'])} source batches. No timestamp is provided; record sequence is used only as an observation order.")


def data_explorer() -> None:
    df = require_data()
    if df is None: return
    page_header("Data Explorer", "DATA QUALITY · DISCOVERY", "Inspect the unified batches, feature distributions, relationships, and data-quality checks.")
    profile = profile_dataset(df)
    st.json({k: v for k, v in profile.items() if k not in ("numeric_features", "categorical_features")})
    tabs = st.tabs(["Profile", "Distributions", "Relationships", "Quality checks"])
    with tabs[0]:
        st.dataframe(df.head(100), use_container_width=True); st.dataframe(df.describe().T, use_container_width=True)
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    with tabs[1]:
        selected = st.selectbox("Variable", numeric, index=0); st.plotly_chart(px.histogram(df, x=selected, color="fault_family", nbins=60, template="plotly_dark"), use_container_width=True)
    with tabs[2]:
        x = st.selectbox("X axis", numeric, index=0, key="exp_x"); y = st.selectbox("Y axis", numeric, index=min(1,len(numeric)-1), key="exp_y")
        sample = df.sample(min(7000, len(df)), random_state=42); st.plotly_chart(px.scatter(sample, x=x, y=y, color="fault_family", template="plotly_dark", title="Process relationship (sampled)"), use_container_width=True)
    with tabs[3]:
        st.write({"missing cells": int(df.isna().sum().sum()), "duplicate rows": int(df.duplicated().sum()), "labelled events": int(df.anomaly_present.sum())})
        st.dataframe(df.isna().sum().sort_values(ascending=False).rename("missing").to_frame().head(25))


def monitoring() -> None:
    df = require_data()
    if df is None: return
    page_header("Process Monitoring", "PROCESS · SIGNALS", "Explore recorded process variables by signal family and observation window.")
    groups = feature_groups(df); group = st.selectbox("Signal family", list(groups)); signals = groups[group]
    window = st.slider("Recent observations", 100, min(3000, len(df)), min(800, len(df)), step=100)
    plot = df.tail(window).copy(); plot["record"] = np.arange(len(plot))
    st.plotly_chart(polish_chart(px.line(plot, x="record", y=signals, title=f"{group} across available stands", color_discrete_sequence=["#64e6de", "#74b8ff", "#ffad5b", "#b69cff", "#6fe0a9"])), use_container_width=True)


def simple_mode() -> bool:
    """Return whether the visitor opted into the plain-language interface."""
    return st.session_state.get("user_mode", "Simple") == "Simple"


def friendly_feature_name(feature: str) -> str:
    names = {
        "thickness_entry": "Entry thickness", "thickness_exit": "Exit thickness",
        "roll_speed": "Roll speed", "motor_power": "Motor power",
        "work_roll_diam": "Work roll diameter", "work_roll_mileage": "Work roll mileage",
        "force": "Rolling force", "torque": "Rolling torque", "gap": "Roll gap",
        "reduction": "Thickness reduction", "tension": "Strip tension",
    }
    for prefix, label in names.items():
        if feature.startswith(prefix):
            suffix = feature.removeprefix(prefix).strip("_")
            return f"{label} {suffix}" if suffix and suffix not in {"mean", "max", "ratio"} else f"{label} {suffix}".strip()
    return feature.replace("_", " ").title()


def technical_help(feature: str) -> str:
    descriptions = {
        "thickness_entry": "Thickness of the strip before it enters the rolling mill.",
        "thickness_exit": "Thickness of the strip after rolling. In a reduction process it is normally lower than entry thickness.",
        "force": "Force applied by the rolls to reduce strip thickness.",
        "torque": "Turning effort required to rotate the rolls.",
        "roll_speed": "Speed at which the rolls move the strip through a stand.",
        "gap": "Distance between the working rolls at a stand.",
        "motor_power": "Power used by the drive motor at a stand.",
        "tension": "Pulling force that keeps the strip stable between stands.",
        "reduction": "Amount of thickness reduction applied at a stand.",
        "work_roll_diam": "Diameter of the work roll in contact with the strip.",
    }
    return next((text for prefix, text in descriptions.items() if feature.startswith(prefix)), "Dataset-derived process input used by the saved model.")


def status_copy(probability: float, priority: str) -> tuple[str, str, str, str]:
    if priority == "CRITICAL":
        return "CRITICAL", "Immediate investigation recommended by the model", "The model detected strongly unusual operating conditions. Follow approved site procedures and engineering review.", "danger"
    if probability >= .5 or priority in {"HIGH", "MEDIUM"}:
        return "WARNING", "Some parameters need attention", "The model found conditions that differ from its learned normal patterns. Review the suggested investigation area.", "warning"
    return "NORMAL", "Operating conditions look expected", "Based on the supplied operating conditions, the model does not detect a strong anomaly.", ""


def dashboard_help() -> None:
    with st.expander("Understanding this dashboard"):
        st.markdown("""
**What is cold rolling?** Steel strip is passed through rolls to make it thinner and achieve the required shape or properties.

**What is a rolling stand?** A rolling stand is one set of rolls in the mill. This dataset represents five stands working in sequence.

**What is rolling force and strip tension?** Rolling force reduces thickness; strip tension keeps material stable between stands.

**What is anomaly detection?** It highlights operating conditions that differ from patterns learned from normal dataset records.

**What is predictive maintenance?** It uses model outputs to guide where to investigate. It is not an OEM maintenance instruction.

**What are machine learning, SHAP, and reconstruction error?** Machine learning finds patterns in examples. SHAP explains a model's feature contributions. Reconstruction error is a technical novelty signal; both are available in Engineer mode.
        """)


def user_friendly_dashboard() -> None:
    df = require_data()
    meta = metadata()
    if df is None:
        return
    current = recent_prediction(df)
    probability = 0.0 if np.isnan(current["probability"]) else current["probability"]
    title, subtitle, explanation, tone = status_copy(probability, current["priority"])
    hero()
    if simple_mode() and not st.session_state.get("welcome_dismissed", False):
        st.info("Welcome to Cold Rolling Mill Intelligence. Use this dashboard to see whether conditions look expected and where the model suggests attention.")
        welcome_left, welcome_right, _ = st.columns((1, 1, 4))
        welcome_left.button("Start Analysis", key="start_analysis", on_click=_navigate_primary, args=("What-If Simulation",))
        welcome_right.button("Dismiss", key="dismiss_welcome", on_click=lambda: st.session_state.update(welcome_dismissed=True))
    st.markdown(f'<section class="status-summary {tone}"><div class="status-summary-title">Mill status</div><h2>{title}</h2><strong>{subtitle}</strong><p>{explanation}</p></section>', unsafe_allow_html=True)
    quality_note = "Quality model unavailable" if simple_mode() else "No measured quality target supplied"
    metric_grid([
        ("Mill health", "—" if np.isnan(current["health"]) else f"{current['health']:.1f}/100", "A project-defined overall condition indicator", tone),
        ("Anomaly risk", "—" if np.isnan(current["probability"]) else f"{current['probability']:.1%}", "How unusual conditions appear compared with learned normal data", tone),
        ("Quality", "Not available", quality_note, "warning"),
        ("Maintenance", current["priority"], "AI-assisted investigation priority, not a plant safety classification", tone),
    ])
    st.markdown('<div class="workflow">' + ''.join(f'<div class="workflow-step"><strong>STEP {index}</strong>{label}</div>' + ('' if index == 6 else '<div class="workflow-arrow">→</div>') for index, label in enumerate(["Enter conditions", "Run AI analysis", "Review health", "Check quality", "Review anomaly risk", "Review guidance"], start=1)) + '</div>', unsafe_allow_html=True)
    if probability >= .5:
        area = recommended_area("Normal")
        st.warning(f"AI-assisted investigation guidance: {area}. This is not a certified maintenance procedure.")
    else:
        st.success("No strong anomaly is detected for the latest dataset record. Continue normal engineering review and monitoring.")
    if meta:
        factors = ", ".join(friendly_feature_name(item["feature"]) for item in meta.get("feature_importance", [])[:3])
        st.info(f"Why this result? The saved model generally gives the most weight to {factors}. Open Explainable AI for the full evidence.")
    dashboard_help()
    if not simple_mode():
        section_heading("Engineer view", "dataset signals and process path")
        cards = ["Entry coil"] + [f"Stand {index}" for index in range(1, 6)] + ["Exit coil"]
        flow = "".join(f'<div class="stand">{card}<br><small class="muted">dataset-driven</small></div>' + ("" if index == len(cards) - 1 else '<div class="arrow">→</div>') for index, card in enumerate(cards))
        st.markdown(f'<div class="process">{flow}</div>', unsafe_allow_html=True)
        with st.expander("Technical details"):
            st.write({"model_status": current["status"], "classification_probability": current["probability"], "reconstruction_error": current["error"], "model_priority": current["priority"]})


def friendly_quality_view() -> None:
    df = require_data()
    if df is None:
        return
    page_header("Quality Check", "QUALITY · CAPABILITY STATUS", "See whether a measured quality prediction is available for this dataset.")
    st.markdown('<section class="status-summary warning"><div class="status-summary-title">Quality status</div><h2>NOT AVAILABLE</h2><strong>A quality outcome has not been supplied</strong><p>The dataset has process inputs but no approved measured quality result. The app therefore does not make up a quality prediction.</p></section>', unsafe_allow_html=True)
    st.info("To enable quality prediction, add governed quality measurements such as a validated grade, surface outcome, or measured material property, then train and validate a dedicated model.")
    if not simple_mode():
        with st.expander("Technical details"):
            st.write("Available inputs include material thickness, reductions, rolling force, torque, tension, speed, roll gap, and motor power. No continuous quality target is present.")


def friendly_prediction_view() -> None:
    df = require_data()
    meta = metadata()
    bundle = model_bundle()
    if df is None or not meta or not bundle:
        st.warning("AI analysis is unavailable because saved model files could not be loaded. Please contact the administrator or train the model from the tools section.")
        return
    page_header("Run AI Analysis", "GUIDED ANALYSIS", "Start with the latest dataset values, change the conditions you want to explore, then review the result in plain language.")
    base = df.iloc[-1:].copy()
    baseline = _predict_frame(base, meta, bundle)
    simple_fields = [field for field in ["thickness_entry", "thickness_exit", "reduction_1", "force_1", "roll_speed_1", "torque_1", "motor_power_1", "tension_0"] if field in base.columns]
    numeric_fields = [column for column in base.select_dtypes(include=np.number).columns if column not in {"anomaly_present", "batch_id", "batch_row"}]
    edited = base.copy()
    with st.form("friendly_prediction_form"):
        st.caption("Smart defaults are taken from the latest supplied dataset record. Typical ranges are dataset ranges, not engineering limits.")
        if simple_mode():
            fields = simple_fields
            st.subheader("Material and rolling conditions")
        else:
            fields = st.multiselect("Parameters to adjust", numeric_fields, default=simple_fields, help="Choose the process values to change for this what-if analysis.")
            st.subheader("Selected process conditions")
        controls = st.columns(2)
        for index, field in enumerate(fields):
            minimum, maximum, current_value = float(df[field].min()), float(df[field].max()), float(base.iloc[0][field])
            controls[index % 2].caption(f"Typical dataset range: {minimum:.4g} – {maximum:.4g} · Current: {current_value:.4g}")
            edited.loc[edited.index[0], field] = controls[index % 2].number_input(friendly_feature_name(field), min_value=minimum, max_value=maximum, value=current_value, format="%.6g", help=technical_help(field))
        submitted = st.form_submit_button("Run AI Analysis", type="primary")
    invalid_thickness = edited.iloc[0].get("thickness_exit", 0) > edited.iloc[0].get("thickness_entry", 0)
    if submitted and invalid_thickness:
        st.warning("Please check the thickness values. Exit thickness is normally expected to be lower than entry thickness for a reduction process.")
    elif submitted:
        with st.spinner("Analyzing operating conditions..."):
            st.session_state["friendly_simulation"] = _predict_frame(edited, meta, bundle)
            st.session_state["friendly_changed_features"] = fields
    result = st.session_state.get("friendly_simulation")
    if result:
        title, subtitle, explanation, tone = status_copy(result["probability"], result["priority"])
        st.markdown(f'<section class="status-summary {tone}"><div class="status-summary-title">AI analysis result</div><h2>{title}</h2><strong>{subtitle}</strong><p>{explanation}</p></section>', unsafe_allow_html=True)
        metric_grid([
            ("Mill health", f"{result['health']:.1f}/100", "Project-defined condition indicator", tone),
            ("Anomaly risk", f"{result['probability']:.1%}", "How unusual the conditions appear", tone),
            ("Maintenance attention", result["priority"], "Model-led investigation priority", tone),
            ("Quality", "Not available", "No measured quality target was supplied", "warning"),
        ])
        factors = ", ".join(friendly_feature_name(item["feature"]) for item in meta.get("feature_importance", [])[:3])
        st.info(f"Why? The saved model generally gives most weight to {factors}. These are model evidence, not confirmed physical causes.")
        if result["probability"] >= .5:
            st.warning(f"AI-assisted investigation guidance: {recommended_area(result['family'])}")
        if not simple_mode():
            with st.expander("Technical details"):
                st.write({"baseline_probability": baseline["probability"], "simulated_probability": result["probability"], "reconstruction_error": result["error"], "fault_family": result["family"], "changed_features": st.session_state.get("friendly_changed_features", [])})


def friendly_anomaly_view() -> None:
    df = require_data()
    meta = metadata()
    bundle = model_bundle()
    if df is None or not meta or not bundle:
        st.warning("Anomaly checking is unavailable because saved model files could not be loaded.")
        return
    result = _predict_frame(df.iloc[-1:].copy(), meta, bundle)
    title, subtitle, explanation, tone = status_copy(result["probability"], result["priority"])
    page_header("Check Anomaly", "AI ANALYSIS · OPERATING CONDITIONS", "Find out whether the latest dataset conditions look unusual to the saved model.")
    st.markdown(f'<section class="status-summary {tone}"><div class="status-summary-title">Anomaly check</div><h2>{title}</h2><strong>{subtitle}</strong><p>{explanation}</p></section>', unsafe_allow_html=True)
    metric_grid([("Anomaly risk score", f"{result['probability']:.1%}", "Shows how unusual current conditions appear compared with normal data", tone), ("Maintenance attention", result["priority"], "Suggested review priority", tone), ("Likely investigation", result["family"], "Available when the model finds an anomaly", ""), ("Quality", "Not available", "No measured quality outcome in this dataset", "warning")])
    if not simple_mode():
        with st.expander("Technical details"):
            st.write({"classifier_probability": result["probability"], "reconstruction_error": result["error"], "reconstruction_threshold": meta["autoencoder"]["threshold"]})


def friendly_maintenance_view() -> None:
    df = require_data()
    meta = metadata()
    bundle = model_bundle()
    if df is None or not meta or not bundle:
        st.warning("Maintenance guidance is unavailable because saved model files could not be loaded.")
        return
    result = _predict_frame(df.iloc[-1:].copy(), meta, bundle)
    title, subtitle, explanation, tone = status_copy(result["probability"], result["priority"])
    page_header("Maintenance Guidance", "AI-ASSISTED INVESTIGATION", "Use the model output to focus inspection discussion. It does not replace approved maintenance procedures.")
    st.markdown(f'<section class="status-summary {tone}"><div class="status-summary-title">Maintenance status</div><h2>{result["priority"]}</h2><strong>{subtitle}</strong><p>{explanation}</p></section>', unsafe_allow_html=True)
    area = recommended_area(result["family"] if result["probability"] >= .5 else "Normal")
    st.markdown(f'<div class="glass-card"><strong>AI-assisted investigation guidance</strong><br>{area}<br><span class="muted">This is model-generated context, not a certified maintenance procedure or safety instruction.</span></div>', unsafe_allow_html=True)
    if not simple_mode():
        with st.expander("Technical details"):
            st.write({"health_score": result["health"], "anomaly_probability": result["probability"], "fault_family": result["family"], "reconstruction_error": result["error"]})


def anomaly() -> None:
    df = require_data()
    if df is None: return
    page_header("Anomaly Detection", "MODEL OUTPUT · ANOMALY", "Review learned anomaly scores alongside the labels supplied with the datasets.")
    meta = metadata(); current = recent_prediction(df)
    if not meta: st.warning("Train a model from Model Training to enable learned anomaly scores."); return
    c1,c2,c3 = st.columns(3); c1.metric("Latest classification risk", f"{current['probability']:.1%}"); c2.metric("Autoencoder reconstruction error", f"{current['error']:.5f}"); c3.metric("Learned threshold", f"{meta['autoencoder']['threshold']:.5f}")
    dist = df["fault_family"].value_counts().rename_axis("fault family").reset_index(name="records")
    st.plotly_chart(polish_chart(px.bar(dist, x="fault family", y="records", color="fault family", title="Dataset-provided anomaly/fault labels", color_discrete_sequence=["#6fe0a9", "#ffad5b", "#ff7b7b", "#b69cff", "#74b8ff"])), use_container_width=True)
    st.caption("The reconstruction threshold is the 99.5th percentile of held-out normal reconstruction error. This is a project prototype, not a certified alarm setting.")


def maintenance() -> None:
    df = require_data()
    if df is None: return
    page_header("Predictive Maintenance", "MAINTENANCE · PRIORITY", "Use model outputs and labelled events to guide investigation, not to authorize plant actions.")
    current = recent_prediction(df); risk = current['priority']; cls = risk.lower() if risk != '—' else 'medium'
    st.markdown(f'<span class="badge {cls}">{risk}</span> &nbsp; Project-defined maintenance priority for the latest record', unsafe_allow_html=True)
    st.write(f"Suggested investigation: {recommended_area('Normal' if current['status']=='NORMAL' else '')}.")
    alerts = alerts_from_frame(df)
    st.dataframe(alerts.head(100), use_container_width=True)
    st.warning("This system is an AI decision-support prototype and does not replace plant safety procedures, OEM limits, or engineering approval.")


def prediction() -> None:
    df = require_data()
    if df is None: return
    page_header("AI Prediction", "WHAT-IF · MODEL SIMULATION", "Adjust a copy of the latest record and inspect how the saved model responds.")
    meta = metadata(); bundle = model_bundle()
    if not meta or not bundle: st.warning("No saved model. Open Model Training and train the pipeline first."); return
    base = df.iloc[-1:].copy(); raw_numeric = [c for c in base.select_dtypes(include=np.number).columns if c not in {"anomaly_present", "batch_id", "batch_row"}]
    edited = base.copy()
    with st.form("prediction_form"):
        st.caption("Inputs start at the latest measured record. This is a model simulation, not an operating recommendation.")
        selected = st.multiselect("Variables to adjust", raw_numeric, default=raw_numeric[:8])
        cols = st.columns(2)
        for i, column in enumerate(selected): edited.loc[edited.index[0], column] = cols[i%2].number_input(column, value=float(base.iloc[0][column]), format="%.6g")
        submitted = st.form_submit_button("Run model simulation")
    if submitted:
        classifier, auto = bundle; values = build_features(edited).reindex(columns=meta['features']); prob = float(classifier.predict_proba(values)[0, list(classifier.classes_).index(1)])
        err = np.nan
        if auto:
            matrix=auto['preprocessor'].transform(values); matrix=matrix.toarray() if hasattr(matrix,'toarray') else matrix; err=float(auto['detector'].score_samples(matrix)[0])
        score=health_score(prob, err if not np.isnan(err) else None, meta['autoencoder']['threshold'])
        family = None
        multiclass = fault_model()
        if prob >= .5 and multiclass is not None:
            family = str(multiclass.predict(values)[0])
        title = f"Prediction: {'ANOMALY' if prob >= .5 else 'NORMAL'}"
        if family: title += f" | likely fault family: {family}"
        st.success(f"{title} | probability {prob:.1%} | health score {score:.1f}/100")
        st.json({"anomaly_score": err, "maintenance_priority": risk_priority(score), "important_features": meta['feature_importance'][:5]})


def deep_learning() -> None:
    page_header("Reconstruction Autoencoder", "DEEP LEARNING · NOVELTY", "Inspect the reconstruction-based detector and its measured training and holdout results.")
    meta = metadata()
    if not meta: st.warning("Train the pipeline to create an autoencoder artifact."); return
    dl = meta['autoencoder']; st.metric("Anomaly threshold", f"{dl['threshold']:.5f}")
    st.dataframe(pd.DataFrame([dl['metrics']]), use_container_width=True)
    st.write(f"Training normal records: {dl['normal_training_rows']:,}")
    st.info("A compact MLP autoencoder was used because the dataset has no timestamps or defined sequence ordering for defensible LSTM/GRU windows. Its score is reconstruction error, not a fault probability.")


def explainability() -> None:
    page_header("Explainable AI", "MODEL INTERPRETATION", "Compare global feature importance with an optional TreeSHAP sample explanation.")
    meta = metadata()
    if not meta: st.warning("Train the pipeline first."); return
    importance = pd.DataFrame(meta['feature_importance'])
    if importance.empty: st.info("The selected estimator did not expose native feature importance."); return
    st.plotly_chart(polish_chart(px.bar(importance.head(20).sort_values('importance'), x='importance', y='feature', orientation='h', title='Global native model importance', color_discrete_sequence=['#64e6de'])), use_container_width=True)
    if st.button("Calculate TreeSHAP sample explanation"):
        try:
            from src.explainability import shap_values_for_tree_pipeline
            df = require_data(); bundle = model_bundle()
            if df is not None and bundle:
                values, names = shap_values_for_tree_pipeline(bundle[0], build_features(df.tail(300)).reindex(columns=meta['features']))
                shap_frame = pd.DataFrame({"feature": names, "mean |SHAP|": np.mean(np.abs(values), axis=0)}).sort_values("mean |SHAP|", ascending=False).head(20)
                st.plotly_chart(px.bar(shap_frame.sort_values("mean |SHAP|"), x="mean |SHAP|", y="feature", orientation="h", template="plotly_dark", title="TreeSHAP global importance (latest sample)"), use_container_width=True)
        except ImportError as exc: st.info(str(exc))
        except Exception as exc: st.error(f"SHAP calculation was unavailable: {exc}")
    st.caption("Native importance is always available; TreeSHAP is available when its optional dependency is installed. Importance is explanatory evidence, not causal proof.")


def simulator() -> None: prediction()


def training() -> None:
    page_header("Model Training", "EXPERIMENTS · ARTIFACTS", "Train and save leakage-aware classifiers and the reconstruction detector from local data.")
    df = require_data()
    if df is None: return
    st.write("Target: `anomaly_present`, derived from the supplied boolean anomaly labels. Features exclude all supplied labels and provenance columns to avoid direct leakage.")
    enable_cv = st.checkbox("Run 3-fold cross-validation on training subset", value=True)
    if st.button("Train and save models", type="primary"):
        with st.spinner("Training baseline, ensemble, and autoencoder models. This may take a few minutes on the full dataset..."):
            try:
                result = train_pipeline(enable_cv=enable_cv)
                data.clear(); metadata.clear(); model_bundle.clear()
                fault_model.clear()
                st.success(f"Saved best model: {result['best_model']}")
                st.dataframe(pd.DataFrame(result['comparison']), use_container_width=True)
            except Exception as exc:
                st.error(f"Model training could not complete: {exc}")
    meta = metadata()
    if meta: st.dataframe(pd.DataFrame(meta['comparison']), use_container_width=True)


def evaluation() -> None:
    page_header("Model Evaluation", "VALIDATION · METRICS", "Review measured holdout and fault-family classification results for the saved models.")
    meta = metadata()
    if not meta: st.warning("Train models to populate measured evaluation results."); return
    st.dataframe(pd.DataFrame(meta['comparison']), use_container_width=True)
    c1,c2=st.columns(2); c1.write("Confusion matrix [actual rows: normal, anomaly]"); c1.dataframe(pd.DataFrame(meta['confusion_matrix'], index=['Actual normal','Actual anomaly'], columns=['Pred normal','Pred anomaly']))
    c2.json(meta['classification_report'])
    fault = meta.get("fault_classification", {})
    if fault.get("available"):
        st.subheader("Fault-family classification (labelled anomaly records only)")
        st.write({"model": fault["model"], "accuracy": fault["accuracy"], "macro_f1": fault["macro_f1"], "weighted_f1": fault["weighted_f1"]})
        report = pd.DataFrame(fault["report"]).T
        st.dataframe(report, use_container_width=True)
    st.caption(meta['validation_strategy'])


def reports() -> None:
    df = require_data()
    if df is None: return
    page_header("Reports & Downloads", "EXPORTS · DATA", "Download a technical summary and bounded samples of the processed dataset.")
    meta=metadata(); profile=profile_dataset(df)
    report={"dataset_summary": profile, "model_metadata": meta, "limitations": ["Dataset has no timestamp; chronological order is inferred from source batch and row.", "No continuous quality target was supplied; no quality-regression model is claimed.", "Source files appear to be simulated/experimental rolling data and must not be represented as plant telemetry."], "safety_disclaimer": "AI decision-support prototype; not a substitute for plant safety procedures, OEM limits, or engineering approval."}
    st.download_button("Download JSON technical report", json.dumps(report, indent=2, default=str), "cold_rolling_ai_report.json", "application/json")
    # Keep exports useful but bounded so opening this page stays responsive on Community Cloud.
    sample = df.head(10000)
    csv = sample.to_csv(index=False).encode(); st.download_button("Download processed data sample (CSV)", csv, "cold_rolling_processed_sample.csv", "text/csv")
    if st.button("Prepare Excel export", key="prepare_excel_export"):
        with st.spinner("Preparing the bounded Excel export..."):
            spreadsheet = io.BytesIO()
            with pd.ExcelWriter(spreadsheet, engine="openpyxl") as writer:
                sample.to_excel(writer, sheet_name="processed_sample", index=False)
                pd.DataFrame(meta.get("comparison", []) if meta else []).to_excel(writer, sheet_name="model_comparison", index=False)
        st.download_button("Download processed data sample (Excel)", spreadsheet.getvalue(), "cold_rolling_processed_sample.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.caption("Excel generation is on demand to keep the report page responsive.")
    st.info("CSV, Excel, and JSON downloads are enabled. PDF requires a reporting renderer and is intentionally not generated with unverified layout dependencies.")


def settings() -> None:
    page_header("Settings", "CONFIGURATION · REFERENCE", "Review the local data paths and project-defined model settings.")
    st.json({"data_directory":"data/raw", "processed_directory":"data/processed", "model_directory":"models", "random_seed":42, "classification_threshold":0.5, "autoencoder_threshold_method":"99.5th percentile of held-out normal reconstruction errors"})


def about() -> None:
    df = require_data()
    if df is None:
        return
    page_header("About Project", "PROJECT · SCOPE", "A local, reproducible analytics prototype for tandem cold-rolling data.")
    profile = profile_dataset(df)
    section_heading("Dataset at a glance", "loaded local records")
    left, right = st.columns(2, gap="large")
    source_counts = df["source_file"].value_counts().sort_index().rename_axis("source batch").reset_index(name="records")
    family_counts = df["fault_family"].value_counts().rename_axis("derived label").reset_index(name="records")
    left.plotly_chart(
        polish_chart(px.bar(source_counts, x="source batch", y="records", text="records", title="Records by source batch", color_discrete_sequence=["#64e6de"])),
        use_container_width=True,
    )
    right.plotly_chart(
        polish_chart(px.bar(family_counts, x="records", y="derived label", text="records", title="Derived fault-family labels", orientation="h", color_discrete_sequence=["#ffad5b"])),
        use_container_width=True,
    )
    st.caption(f"{profile['rows']:,} rows across {len(profile['source_files'])} supplied batches; {profile['columns']} columns, {profile['missing_cells']:,} missing cells, and {profile['duplicate_rows']:,} duplicate rows. Fault families are derived display categories; a record with multiple supplied labels is assigned one family for this chart.")

    section_heading("How the project fits together", "system mind map")
    nodes = [
        (0, 0, "COLD ROLLING<br>AI COMMAND CENTER", "#167d87", 15),
        (-2.6, 1.8, "DATA & SIGNALS", "#216c9f", 12),
        (-5.4, 2.6, "6 CSV source batches", "#123349", 10),
        (-5.4, 1.8, f"{profile['rows']:,} records · {profile['columns']} columns", "#123349", 10),
        (-5.4, 1.0, "Process values + supplied labels", "#123349", 10),
        (2.6, 1.8, "MODELS", "#94612c", 12),
        (5.4, 2.6, "Supervised anomaly classifier", "#3b2b1d", 10),
        (5.4, 1.8, "Fault-family classifier", "#3b2b1d", 10),
        (5.4, 1.0, "MLP reconstruction detector", "#3b2b1d", 10),
        (-2.6, -1.8, "APP CAPABILITIES", "#29805e", 12),
        (-5.4, -1.0, "Monitoring + exploration", "#17372e", 10),
        (-5.4, -1.8, "Prediction + explanations", "#17372e", 10),
        (-5.4, -2.6, "Maintenance + reports", "#17372e", 10),
        (2.6, -1.8, "LIMITS & SAFE USE", "#ae5151", 12),
        (5.4, -1.0, "Experimental/simulated data", "#402526", 10),
        (5.4, -1.8, "No timestamp or quality target", "#402526", 10),
        (5.4, -2.6, "Decision support; human review", "#402526", 10),
    ]
    links = [(0, 1), (1, 2), (1, 3), (1, 4), (0, 5), (5, 6), (5, 7), (5, 8),
             (0, 9), (9, 10), (9, 11), (9, 12), (0, 13), (13, 14), (13, 15), (13, 16)]
    mind_map = go.Figure()
    for source, target in links:
        x0, y0, *_ = nodes[source]
        x1, y1, *_ = nodes[target]
        mind_map.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="rgba(139,181,196,.48)", width=2))
    for x, y, label, color, size in nodes:
        mind_map.add_annotation(
            x=x, y=y, text=label, showarrow=False, bgcolor=color,
            bordercolor="rgba(220,245,248,.28)", borderwidth=1, borderpad=8,
            font=dict(family="Manrope, sans-serif", size=size, color="#f1fbfc"),
        )
    mind_map.update_layout(
        template="plotly_dark", height=510, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(7,16,25,.38)",
        margin=dict(l=12, r=12, t=20, b=16), showlegend=False,
        xaxis=dict(range=[-7.1, 7.1], visible=False, fixedrange=True),
        yaxis=dict(range=[-3.2, 3.2], visible=False, fixedrange=True),
    )
    st.plotly_chart(mind_map, use_container_width=True, config={"displayModeBar": False})
    st.info("The supplied files have no provenance confirming real plant telemetry. No continuous quality target or timestamp is available, so the app does not claim quality prediction or time-based forecasting. Outputs are decision support only and do not replace safety procedures, OEM limits, or engineering approval.")


# Premium task views.  They intentionally reuse the cached data and serialized
# artifacts above; presentation changes do not alter the project's ML methods.
def hero() -> None:
    st.markdown(
        '''<section class="hero"><div class="eyebrow">Tandem cold rolling · intelligence layer</div>
        <h1>AI-Powered Cold Rolling Mill Intelligence</h1>
        <p>Dataset intelligence, anomaly detection, explainability, and predictive-maintenance decision support for a five-stand tandem cold rolling mill.</p>
        <span class="live-pill"><i class="live-dot"></i> MODEL / DATASET VIEW · NOT LIVE PLANT CONNECTED</span></section>''',
        unsafe_allow_html=True,
    )
    action_a, action_b, _ = st.columns((1, 1, 3))
    action_a.button("Explore Mill Intelligence", key="hero_explore", on_click=_navigate_primary, args=("Mill Intelligence",))
    action_b.button("Run Prediction", key="hero_prediction", on_click=_navigate_primary, args=("What-If Simulation",))


def _predict_frame(frame: pd.DataFrame, meta: dict[str, Any], bundle: tuple[Any, Any]) -> dict[str, Any]:
    """Run the existing saved artifacts for one frame and return display values."""
    classifier, auto = bundle
    values = build_features(frame).reindex(columns=meta["features"])
    probability = float(classifier.predict_proba(values)[0, list(classifier.classes_).index(1)])
    error = np.nan
    if auto:
        matrix = auto["preprocessor"].transform(values)
        matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        error = float(auto["detector"].score_samples(matrix)[0])
    health = health_score(probability, error if not np.isnan(error) else None, meta.get("autoencoder", {}).get("threshold"))
    family = "Normal operating pattern"
    multiclass = fault_model()
    if probability >= .5 and multiclass is not None:
        family = str(multiclass.predict(values)[0])
    return {"probability": probability, "error": error, "health": health, "priority": risk_priority(health), "family": family, "values": values}


def dashboard() -> None:
    df = require_data()
    if df is None:
        return
    hero()
    current = recent_prediction(df)
    profile = profile_dataset(df)
    metric_grid([
        ("Overall mill health", "—" if np.isnan(current["health"]) else f"{current['health']:.1f}/100", "Project-defined health indicator", ""),
        ("Anomaly risk", "—" if np.isnan(current["probability"]) else f"{current['probability']:.1%}", "Latest-record model probability", "warning" if current["probability"] >= .5 else ""),
        ("Latest model status", current["status"], "Saved classifier output", "danger" if current["status"] == "ANOMALY" else ""),
        ("Labelled events", f"{int(df['anomaly_present'].sum()):,}", "Across supplied local batches", ""),
    ])
    section_heading("Tandem mill flow", "five-stand process context")
    cards = ["Entry coil"] + [f"Stand {i}" for i in range(1, 6)] + ["Exit coil"]
    flow = "".join(f'<div class="stand">{card}<br><small class="muted">dataset-driven</small></div>' + ("" if index == len(cards) - 1 else '<div class="arrow">→</div>') for index, card in enumerate(cards))
    st.markdown(f'<div class="process">{flow}</div>', unsafe_allow_html=True)
    view = df.tail(min(800, len(df))).copy()
    view["record"] = np.arange(len(view))
    available = [column for group in feature_groups(df).values() for column in group]
    section_heading("Operational signals", "latest 800 observations")
    left, right = st.columns((1.35, 1), gap="large")
    if available:
        left.plotly_chart(polish_chart(px.line(view, x="record", y=available[:3], title="Recent process-variable trend", color_discrete_sequence=["#64e6de", "#74b8ff", "#ffad5b"])), use_container_width=True)
    right.plotly_chart(polish_chart(px.scatter(view, x="record", y="anomaly_present", color="fault_family", title="Dataset-labelled anomaly timeline", color_discrete_sequence=["#6fe0a9", "#ffad5b", "#ff7b7b", "#b69cff", "#74b8ff"])), use_container_width=True)
    st.caption(f"Model / Dataset View · {profile['rows']:,} records from {len(profile['source_files'])} source batches. No timestamp is supplied, so record sequence is observation order only.")


def quality_prediction() -> None:
    """Show the honest quality-model capability state without inventing a target."""
    df = require_data()
    if df is None:
        return
    page_header("Quality Prediction", "QUALITY · CAPABILITY STATUS", "Review the material and rolling inputs available for quality modelling.")
    st.markdown('<div class="alert-card"><strong>Quality model unavailable</strong><br><span class="muted">The supplied dataset has no measured continuous quality target or approved quality class. A quality prediction would be fabricated, so this platform does not create one.</span></div>', unsafe_allow_html=True)
    metric_grid([
        ("Quality target", "Not supplied", "No label for supervised quality training", "warning"),
        ("Material inputs", "Available", "Thickness, width, yield strength", ""),
        ("Rolling inputs", "Available", "Reduction, force, torque, gap", ""),
        ("Model status", "Not trained", "Awaiting governed quality measurements", "warning"),
    ])
    section_heading("Available modelling inputs", "dataset schema")
    grouped = {"Material parameters": ["thickness_entry", "thickness_exit", "width", "ys_entry", "ys_exit"], "Rolling & mechanical parameters": ["reduction_1", "force_1", "torque_1", "gap_1", "motor_power_1"], "Process parameters": ["tension_0", "tension_1", "roll_speed_1", "work_roll_diam_1", "work_roll_mileage_1"]}
    columns = st.columns(3)
    for column, (title, fields) in zip(columns, grouped.items()):
        available = [field for field in fields if field in df.columns]
        column.markdown(f'<div class="glass-card"><strong>{title}</strong><br><span class="muted">{" · ".join(available)}</span></div>', unsafe_allow_html=True)
    st.info("To enable this page, add a governed quality target (for example, a validated grade, surface-quality outcome, or measured property), then retrain and validate a dedicated quality model.")


def anomaly() -> None:
    df = require_data()
    meta = metadata()
    bundle = model_bundle()
    if df is None or not meta or not bundle:
        st.warning("Saved model artifacts are unavailable. Train the pipeline from Model Training to enable this view.")
        return
    result = _predict_frame(df.iloc[-1:].copy(), meta, bundle)
    status = "ANOMALY" if result["probability"] >= .5 else "NORMAL"
    page_header("Anomaly Detection", "MODEL OUTPUT · ANOMALY", "Inspect the current saved-model output and the reconstruction-based novelty signal.")
    metric_grid([
        ("Anomaly score", f"{result['probability']:.1%}", "Classifier probability", "danger" if status == "ANOMALY" else ""),
        ("Anomaly status", status, "Threshold: 50% classifier probability", "danger" if status == "ANOMALY" else ""),
        ("Risk level", result["priority"], "Project-defined maintenance band", "warning" if result["priority"] in {"MEDIUM", "HIGH"} else ""),
        ("Novelty error", f"{result['error']:.5f}", f"Autoencoder threshold {meta['autoencoder']['threshold']:.5f}", ""),
    ])
    gauge = go.Figure(go.Indicator(mode="gauge+number", value=result["probability"] * 100, number={"suffix": "%"}, title={"text": "Latest anomaly probability"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#ff8484" if status == "ANOMALY" else "#64e6de"}, "steps": [{"range": [0, 50], "color": "rgba(116,221,165,.18)"}, {"range": [50, 80], "color": "rgba(255,195,107,.16)"}, {"range": [80, 100], "color": "rgba(255,132,132,.16)"}] }))
    left, right = st.columns((1, 1.25), gap="large")
    left.plotly_chart(polish_chart(gauge, 280), use_container_width=True)
    family_counts = df["fault_family"].value_counts().rename_axis("Fault family").reset_index(name="Records")
    right.plotly_chart(polish_chart(px.bar(family_counts, x="Fault family", y="Records", color="Fault family", title="Supplied-label distribution", color_discrete_sequence=["#6fe0a9", "#ffad5b", "#ff7b7b", "#b69cff", "#74b8ff"]), 280), use_container_width=True)
    st.caption("The reconstruction threshold is learned from held-out normal records. Neither output is a certified plant alarm setting.")


def maintenance() -> None:
    df = require_data()
    meta = metadata()
    bundle = model_bundle()
    if df is None or not meta or not bundle:
        st.warning("Saved model artifacts are unavailable. Train the pipeline from Model Training to enable this view.")
        return
    result = _predict_frame(df.iloc[-1:].copy(), meta, bundle)
    page_header("Predictive Maintenance", "MAINTENANCE · PRIORITY", "Use project-defined model outputs to focus investigation; do not use them as an operating authorization.")
    metric_grid([
        ("Machine health score", f"{result['health']:.1f}/100", "Project-defined composite score", "danger" if result["priority"] == "CRITICAL" else ""),
        ("Maintenance priority", result["priority"], "Derived from current model output", "warning" if result["priority"] in {"MEDIUM", "HIGH"} else ""),
        ("Potential fault", result["family"], "Fault-family classifier when anomalous", ""),
        ("Recommended area", "Investigation", "Review recommended area below", ""),
    ])
    section_heading("Recommended investigation", "model-generated guidance")
    st.markdown(f'<div class="glass-card"><strong>{recommended_area(result["family"] if result["probability"] >= .5 else "Normal")}</strong><br><span class="muted">This is model-generated investigation context, not an OEM maintenance procedure or safety instruction.</span></div>', unsafe_allow_html=True)
    section_heading("Labelled event queue", "dataset evidence")
    alerts = alerts_from_frame(df)
    st.dataframe(alerts.head(100), use_container_width=True, height=320)
    st.warning("This system is a decision-support prototype. Immediate safety concerns must follow approved plant procedures, OEM limits, and engineering review.")


def fault_analysis() -> None:
    df = require_data()
    meta = metadata()
    bundle = model_bundle()
    if df is None or not meta or not bundle:
        st.warning("Saved model artifacts are unavailable. Train the pipeline from Model Training to enable this view.")
        return
    result = _predict_frame(df.iloc[-1:].copy(), meta, bundle)
    page_header("Fault Analysis", "FAULT FAMILY · INVESTIGATION", "Combine the existing fault-family classifier with actual global feature-importance evidence.")
    status = "No predicted fault" if result["probability"] < .5 else result["family"]
    metric_grid([("Fault family", status, "Latest record", "danger" if result["probability"] >= .5 else ""), ("Fault confidence", f"{result['probability']:.1%}", "Binary anomaly classifier probability", ""), ("Affected parameters", "Top model features", "Global, not causal", ""), ("Investigation area", "Model guidance", "See recommendation below", "")])
    importance = pd.DataFrame(meta.get("feature_importance", [])).head(15)
    left, right = st.columns((1.3, 1), gap="large")
    left.plotly_chart(polish_chart(px.bar(importance.sort_values("importance"), x="importance", y="feature", orientation="h", title="Global feature contribution", color_discrete_sequence=["#64e6de"])), use_container_width=True)
    family_counts = df.loc[df["anomaly_present"] == 1, "fault_family"].value_counts().rename_axis("Fault family").reset_index(name="Labelled events")
    right.plotly_chart(polish_chart(px.pie(family_counts, values="Labelled events", names="Fault family", hole=.58, title="Labelled anomaly families", color_discrete_sequence=["#ffad5b", "#ff7b7b", "#b69cff", "#74b8ff"])), use_container_width=True)
    st.markdown(f'<div class="glass-card"><strong>Recommended investigation area</strong><br>{recommended_area(result["family"] if result["probability"] >= .5 else "Normal")}</div>', unsafe_allow_html=True)


def prediction() -> None:
    df = require_data()
    meta = metadata()
    bundle = model_bundle()
    if df is None or not meta or not bundle:
        st.warning("No saved model artifacts. Open Model Training and train the pipeline first.")
        return
    page_header("What-If Simulation", "PREDICTION · MODEL SIMULATION", "Change a copy of the latest recorded process values and compare the existing model output.")
    base = df.iloc[-1:].copy()
    baseline = _predict_frame(base, meta, bundle)
    raw_numeric = [column for column in base.select_dtypes(include=np.number).columns if column not in {"anomaly_present", "batch_id", "batch_row"}]
    edited = base.copy()
    with st.form("prediction_form"):
        st.caption("Inputs begin at the latest dataset record. This is a model simulation, not an operating recommendation.")
        selected = st.multiselect("Variables to adjust", raw_numeric, default=raw_numeric[:8])
        controls = st.columns(2)
        for index, column in enumerate(selected):
            edited.loc[edited.index[0], column] = controls[index % 2].number_input(column, value=float(base.iloc[0][column]), format="%.6g")
        submitted = st.form_submit_button("Run simulation", type="primary")
    if submitted:
        st.session_state["simulation_result"] = _predict_frame(edited, meta, bundle)
        st.session_state["simulation_changed"] = selected
    simulated = st.session_state.get("simulation_result")
    if simulated:
        delta = simulated["probability"] - baseline["probability"]
        metric_grid([
            ("Current prediction", f"{baseline['probability']:.1%}", f"Health {baseline['health']:.1f}/100", ""),
            ("Simulated prediction", f"{simulated['probability']:.1%}", f"Health {simulated['health']:.1f}/100", "danger" if simulated["probability"] >= .5 else ""),
            ("Prediction change", f"{delta:+.1%}", "Simulated minus current", "warning" if delta > 0 else ""),
            ("Risk change", f"{baseline['priority']} → {simulated['priority']}", "Project-defined priority bands", ""),
        ])
        changed = st.session_state.get("simulation_changed", [])
        st.info(f"Changed features: {', '.join(changed) if changed else 'None'}. Important global features remain available in Explainable AI.")


def simulator() -> None:
    prediction()


def evaluation() -> None:
    meta = metadata()
    if not meta:
        st.warning("Train models to populate measured evaluation results.")
        return
    page_header("Model Performance", "VALIDATION · MEASURED METRICS", "Measured holdout performance for the saved anomaly model; no metrics are fabricated.")
    metrics = meta["best_metrics"]
    metric_grid([("Accuracy", f"{metrics['accuracy']:.1%}", "Holdout set", ""), ("Precision", f"{metrics['precision']:.1%}", "Anomaly class", ""), ("Recall", f"{metrics['recall']:.1%}", "Anomaly class", "warning"), ("F1 score", f"{metrics['f1']:.1%}", "Anomaly class", ""), ("ROC-AUC", f"{metrics['roc_auc']:.3f}", "Holdout set", ""), ("PR-AUC", f"{metrics['pr_auc']:.3f}", "Holdout set", ""), ("Balanced accuracy", f"{metrics['balanced_accuracy']:.1%}", "Holdout set", "")])
    comparison = pd.DataFrame(meta["comparison"])
    left, right = st.columns((1.2, 1), gap="large")
    left.plotly_chart(polish_chart(px.bar(comparison, x="model", y=["f1", "pr_auc", "roc_auc"], barmode="group", title="Measured model comparison", color_discrete_sequence=["#64e6de", "#ffad5b", "#74b8ff"])), use_container_width=True)
    matrix = np.asarray(meta["confusion_matrix"])
    right.plotly_chart(polish_chart(go.Figure(go.Heatmap(z=matrix, x=["Predicted normal", "Predicted anomaly"], y=["Actual normal", "Actual anomaly"], text=matrix, texttemplate="%{text}", colorscale=[[0, "#102938"], [1, "#64e6de"]], showscale=False)).update_layout(title="Holdout confusion matrix")), use_container_width=True)
    section_heading("Feature importance", "saved selected model")
    importance = pd.DataFrame(meta["feature_importance"]).head(20)
    st.plotly_chart(polish_chart(px.bar(importance.sort_values("importance"), x="importance", y="feature", orientation="h", title="Global native feature importance", color_discrete_sequence=["#64e6de"])), use_container_width=True)
    st.caption(meta["validation_strategy"])


def explainability() -> None:
    """Show stored model evidence and optional, real TreeSHAP calculations."""
    meta = metadata()
    if not meta:
        st.warning("Train the pipeline first to create explainability metadata.")
        return
    page_header("Explainable AI", "MODEL INTERPRETATION", "See what influenced the anomaly model through saved native importance and optional TreeSHAP evidence.")
    importance = pd.DataFrame(meta.get("feature_importance", []))
    if importance.empty:
        st.info("The selected model did not expose native feature importance.")
        return
    metric_grid([
        ("Explanation source", "Native + TreeSHAP", "TreeSHAP is calculated on demand", ""),
        ("Global features", str(len(importance)), "Saved model importance entries", ""),
        ("Individual view", "On demand", "Latest-record TreeSHAP values", ""),
        ("Interpretation", "Non-causal", "Evidence, not a process cause", "warning"),
    ])
    tabs = st.tabs(["Global feature importance", "Individual prediction explanation"])
    with tabs[0]:
        st.plotly_chart(polish_chart(px.bar(importance.head(20).sort_values("importance"), x="importance", y="feature", orientation="h", title="Global native feature importance", color_discrete_sequence=["#64e6de"])), use_container_width=True)
        st.caption("Native importance is saved during model training and is available without recomputing the model.")
    with tabs[1]:
        st.caption("TreeSHAP is optional and calculated from the existing saved tree model; no surrogate or fake contribution values are used.")
        if st.button("Calculate latest-record TreeSHAP explanation", type="primary"):
            try:
                from src.explainability import shap_values_for_tree_pipeline

                df = require_data()
                bundle = model_bundle()
                if df is not None and bundle:
                    values, names = shap_values_for_tree_pipeline(bundle[0], build_features(df.tail(300)).reindex(columns=meta["features"]))
                    st.session_state["tree_shap_latest"] = (values, names)
            except ImportError as exc:
                st.info(str(exc))
            except Exception as exc:
                st.error(f"TreeSHAP calculation was unavailable: {exc}")
        cached_shap = st.session_state.get("tree_shap_latest")
        if cached_shap:
            values, names = cached_shap
            local = pd.DataFrame({"feature": names, "SHAP contribution": values[-1]}).sort_values("SHAP contribution")
            left, right = st.columns((1, 1), gap="large")
            left.plotly_chart(polish_chart(px.bar(local.head(12), x="SHAP contribution", y="feature", orientation="h", title="Negative contribution", color_discrete_sequence=["#ff8484"]), 360), use_container_width=True)
            right.plotly_chart(polish_chart(px.bar(local.tail(12), x="SHAP contribution", y="feature", orientation="h", title="Positive contribution", color_discrete_sequence=["#64e6de"]), 360), use_container_width=True)
            st.caption("Contributions describe the selected model's output for the last sampled record. They do not establish physical causation.")


def monitoring() -> None:
    """Mill-intelligence view based entirely on the latest supplied dataset record."""
    df = require_data()
    if df is None:
        return
    page_header("Mill Intelligence", "PROCESS · FIVE-STAND TANDEM MILL", "Review key process parameters across the tandem mill using the supplied dataset, not a live plant connection.")
    latest = df.iloc[-1]
    def value(name: str, digits: int = 3) -> str:
        raw = latest.get(name, np.nan)
        return "—" if pd.isna(raw) else f"{float(raw):,.{digits}f}"
    metric_grid([
        ("Entry thickness", value("thickness_entry"), "Dataset units not supplied", ""),
        ("Exit thickness", value("thickness_exit"), "Dataset units not supplied", ""),
        ("Total reduction", value("total_reduction_ratio"), "Derived model feature", ""),
        ("Mean motor power", value("motor_power_mean"), "Derived model feature", ""),
    ])
    section_heading("Tandem process path", "latest dataset record")
    cards = ["Entry"] + [f"Stand {stand}" for stand in range(1, 6)] + ["Exit"]
    st.markdown('<div class="process">' + ''.join(f'<div class="stand">{card}<br><small class="muted">recorded view</small></div>' + ('' if index == len(cards) - 1 else '<div class="arrow">→</div>') for index, card in enumerate(cards)) + '</div>', unsafe_allow_html=True)
    rows = []
    for stand in range(1, 6):
        rows.append({"Stand": f"Stand {stand}", "Reduction": value(f"reduction_{stand}"), "Rolling force": value(f"force_{stand}"), "Torque": value(f"torque_{stand}"), "Roll speed": value(f"roll_speed_{stand}"), "Roll gap": value(f"gap_{stand}"), "Motor power": value(f"motor_power_{stand}")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Values are a Model / Dataset View. The source does not provide units, timestamps, or a real-time connection.")
    section_heading("Operational signal explorer", "recorded history")
    groups = feature_groups(df)
    group = st.selectbox("Signal family", list(groups))
    signals = groups[group]
    window = st.slider("Recent observations", 100, min(3000, len(df)), min(800, len(df)), step=100)
    plot = df.tail(window).copy()
    plot["record"] = np.arange(len(plot))
    st.plotly_chart(polish_chart(px.line(plot, x="record", y=signals, title=f"{group} across available stands", color_discrete_sequence=["#64e6de", "#74b8ff", "#ffad5b", "#b69cff", "#6fe0a9"])), use_container_width=True)
