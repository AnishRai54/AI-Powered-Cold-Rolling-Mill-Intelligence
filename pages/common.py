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
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {--canvas:#071019;--surface:#0d1b27;--line:rgba(139,181,196,.17);--text:#ecf5f7;--muted:#91aab5;--cyan:#64e6de;--blue:#74b8ff;--orange:#ffad5b;--green:#6fe0a9;}
    .stApp {background:radial-gradient(circle at 86% -8%,rgba(49,125,154,.22),transparent 28rem),radial-gradient(circle at 9% 28%,rgba(24,97,114,.15),transparent 24rem),var(--canvas);color:var(--text);font-family:'Manrope',sans-serif;}
    [data-testid="stHeader"] {background:rgba(7,16,25,.66);backdrop-filter:blur(14px);}
    [data-testid="stSidebar"] {background:linear-gradient(180deg,#0c1a26 0%,#08131d 100%);border-right:1px solid var(--line);}
    [data-testid="stSidebar"] > div:first-child {padding-top:1.25rem;}.block-container {max-width:1500px;padding-top:1.35rem;padding-bottom:2rem;}
    h1,h2,h3 {color:var(--text)!important;letter-spacing:-.035em;} h1{font-weight:800!important;} h2{font-weight:750!important;}p,label,.stCaption {color:var(--muted);}
    [data-testid="stRadio"] label {border-radius:9px;padding:.33rem .52rem;margin:.05rem 0;transition:.2s ease;font-size:.89rem;}[data-testid="stRadio"] label:hover {background:rgba(100,230,222,.08);color:var(--text);}[data-testid="stRadio"] label:has(input:checked) {background:linear-gradient(90deg,rgba(100,230,222,.16),rgba(116,184,255,.08));border:1px solid rgba(100,230,222,.15);}[data-testid="stRadio"] label:has(input:checked) p {color:#e9ffff;font-weight:700;}
    .brand {padding:.3rem .15rem 1.35rem;display:flex;gap:.72rem;align-items:center}.brand-mark{width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,var(--cyan),var(--blue));box-shadow:0 0 25px rgba(100,230,222,.28);display:grid;place-items:center;color:#06202a;font-size:18px;font-weight:900}.brand-title{font-size:.84rem;font-weight:800;letter-spacing:.13em;color:#effbfc}.brand-sub{font-family:'DM Mono';font-size:.61rem;letter-spacing:.09em;color:#7f9da9;margin-top:2px}
    .hero {position:relative;overflow:hidden;padding:2rem 2.2rem 2.05rem;border-radius:22px;background:linear-gradient(118deg,rgba(15,43,59,.98),rgba(14,77,96,.92) 60%,rgba(167,83,35,.84));border:1px solid rgba(139,226,222,.19);box-shadow:0 20px 45px rgba(0,0,0,.20);margin-bottom:1.2rem;}.hero:after{content:'';position:absolute;right:-70px;top:-120px;width:380px;height:380px;border:1px solid rgba(169,244,236,.17);border-radius:50%;box-shadow:0 0 0 44px rgba(169,244,236,.055),0 0 0 88px rgba(169,244,236,.035)}.hero h1 {position:relative;margin:.55rem 0 0;color:#fff;font-size:clamp(1.7rem,3.1vw,2.65rem);line-height:1.12;max-width:790px;}.hero p {position:relative;margin:.7rem 0 0;color:#d9f1f4;font-size:1rem;max-width:740px;line-height:1.65;}.eyebrow{position:relative;font-family:'DM Mono';font-size:.68rem;letter-spacing:.13em;color:#bcefed;text-transform:uppercase}.live-pill{position:relative;display:inline-flex;align-items:center;gap:.42rem;padding:.32rem .62rem;border-radius:99px;background:rgba(3,21,29,.28);border:1px solid rgba(222,255,251,.18);font-family:'DM Mono';font-size:.66rem;color:#e5ffff}.live-dot{height:6px;width:6px;border-radius:50%;background:var(--cyan);box-shadow:0 0 11px var(--cyan)}
    .kpi {position:relative;overflow:hidden;background:linear-gradient(145deg,rgba(18,41,55,.96),rgba(10,25,36,.96));border:1px solid var(--line);border-radius:16px;padding:1.08rem 1.1rem;min-height:121px;box-shadow:0 10px 22px rgba(0,0,0,.10);transition:transform .2s ease,border-color .2s ease}.kpi:hover{transform:translateY(-3px);border-color:rgba(100,230,222,.38)}.kpi:before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:linear-gradient(180deg,var(--cyan),var(--blue));}.kpi label{display:block;color:#8ea8b4;font-family:'DM Mono';font-size:.65rem;letter-spacing:.08em;text-transform:uppercase}.kpi strong{display:block;font-size:1.72rem;letter-spacing:-.045em;color:#f4ffff;margin:.48rem 0 .26rem;font-weight:800}.kpi span{color:#79c9bb;font-size:.72rem;line-height:1.35;display:block}
    .section-title{display:flex;align-items:center;justify-content:space-between;margin:1.55rem 0 .6rem}.section-title h3{font-size:1rem!important;letter-spacing:-.01em;margin:0}.section-title span{font-family:'DM Mono';font-size:.65rem;color:#6f919f;letter-spacing:.08em;text-transform:uppercase}.section-rule{height:1px;background:linear-gradient(90deg,var(--line),transparent);margin-bottom:.75rem}
    .badge {display:inline-block;padding:.32rem .68rem;border-radius:99px;font-family:'DM Mono';letter-spacing:.05em;font-weight:700;font-size:.67rem}.low{background:rgba(43,151,114,.18);border:1px solid rgba(111,224,169,.24);color:#9bf0c3}.medium{background:rgba(203,146,31,.18);border:1px solid rgba(255,204,106,.23);color:#ffe2a0}.high,.critical{background:rgba(210,73,74,.18);border:1px solid rgba(255,148,148,.23);color:#ffb5b5}
    .process {display:flex;align-items:center;gap:8px;overflow-x:auto;padding:.45rem 0 1.1rem}.stand{position:relative;min-width:116px;text-align:left;border:1px solid var(--line);border-radius:13px;padding:.85rem .8rem;background:linear-gradient(145deg,#102432,#0c1b27);font-size:.83rem;font-weight:700}.stand:before{content:'';display:block;width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 10px rgba(111,224,169,.8);margin-bottom:.45rem}.arrow{color:#4d7889;font-size:18px}.muted{color:#7f9ba7;font-family:'DM Mono';font-size:.61rem}.footer{border-top:1px solid var(--line);color:#718c97;text-align:center;padding:1.25rem 0 .15rem;margin-top:1rem;font-family:'DM Mono';font-size:.68rem;letter-spacing:.04em}
    [data-testid="stMetric"]{padding:1rem;border-radius:14px;background:var(--surface);border:1px solid var(--line)} [data-testid="stMetricLabel"]{font-family:'DM Mono';font-size:.67rem;text-transform:uppercase;letter-spacing:.06em;color:#91aab5}[data-testid="stMetricValue"]{font-weight:800;color:#edf9fb}.stTabs [data-baseweb="tab-list"]{gap:.35rem;border-bottom:1px solid var(--line)}.stTabs [data-baseweb="tab"]{height:38px;border-radius:8px 8px 0 0;padding:0 14px;color:#8fa9b4}.stTabs [aria-selected="true"]{background:rgba(100,230,222,.10)!important;color:#dffffd!important}.stButton>button,[data-testid="stDownloadButton"]>button{border-radius:9px;font-weight:700;border:1px solid rgba(100,230,222,.33);background:linear-gradient(135deg,#167d87,#216c9f);color:white;box-shadow:0 6px 16px rgba(16,107,122,.20)}.stButton>button:hover,[data-testid="stDownloadButton"]>button:hover{border-color:var(--cyan);color:white;box-shadow:0 7px 24px rgba(100,230,222,.19)}[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}.stAlert{border-radius:12px}.js-plotly-plot .plotly .modebar{background:rgba(10,25,36,.65)!important;border-radius:8px}
    .page-header{padding:.35rem 0 1.1rem;margin-bottom:1.2rem;border-bottom:1px solid var(--line)}.page-header .eyebrow{color:var(--cyan)}.page-header h1{font-size:2.15rem!important;line-height:1.18;margin:.42rem 0 .35rem;letter-spacing:0!important}.page-header p{max-width:820px;margin:0;line-height:1.6}.hero-layout{align-items:center;margin-bottom:1.1rem}.hero-copy{padding:1.8rem 0 1.9rem;border-bottom:1px solid var(--line)}.hero-copy h1{font-size:2.55rem!important;line-height:1.12;letter-spacing:0!important;margin:.55rem 0}.hero-copy p{max-width:680px;line-height:1.65;color:#c4d9df}.hero-image img{height:280px;object-fit:cover;border-radius:12px;border:1px solid var(--line)}
    h1,h2,h3{letter-spacing:0!important} [data-testid="stVerticalBlock"]{gap:1rem} [data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-testid="stSelectbox"]>div,[data-testid="stMultiSelect"]>div{border-radius:8px} [data-testid="stTextInput"] input:focus,[data-testid="stNumberInput"] input:focus{border-color:var(--cyan);box-shadow:0 0 0 1px var(--cyan)}
    [data-testid="stDataFrame"],[data-testid="stTable"]{border-radius:8px} .stButton>button,[data-testid="stDownloadButton"]>button{min-height:2.55rem} .stTabs [data-baseweb="tab"]{letter-spacing:0}
    @media(max-width:760px){.block-container{padding-top:1rem}.page-header h1{font-size:1.8rem!important}.hero-copy h1{font-size:2rem!important}.hero-image img{height:190px}.hero-copy{padding:1rem 0}}
    </style>""", unsafe_allow_html=True)


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


def dashboard() -> None:
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
    if signals: st.plotly_chart(polish_chart(px.box(df.sample(min(10000,len(df)), random_state=42), y=signals, color="fault_family", title="Distribution by fault family", color_discrete_sequence=["#6fe0a9", "#ffad5b", "#ff7b7b", "#b69cff", "#74b8ff"])), use_container_width=True)
    st.info("Charts show recorded dataset variables. They are not live plant control signals.")


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
            except Exception as exc: st.exception(exc)
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
    csv = df.head(50000).to_csv(index=False).encode(); st.download_button("Download processed data sample (CSV)", csv, "cold_rolling_processed_sample.csv", "text/csv")
    spreadsheet = io.BytesIO()
    with pd.ExcelWriter(spreadsheet, engine="openpyxl") as writer:
        df.head(50000).to_excel(writer, sheet_name="processed_sample", index=False)
        pd.DataFrame(meta.get("comparison", []) if meta else []).to_excel(writer, sheet_name="model_comparison", index=False)
    st.download_button("Download processed data sample (Excel)", spreadsheet.getvalue(), "cold_rolling_processed_sample.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
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
