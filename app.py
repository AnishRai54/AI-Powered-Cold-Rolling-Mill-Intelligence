"""Streamlit entry point for the Cold Rolling Mill Intelligence platform."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from components.layout import render_topbar
from pages import common


st.set_page_config(page_title="Cold Rolling Mill Intelligence", page_icon="M", layout="wide", initial_sidebar_state="expanded")
common.inject_theme()

PRIMARY_PAGES = {
    "Overview": common.dashboard,
    "Mill Intelligence": common.monitoring,
    "Quality Prediction": common.quality_prediction,
    "Anomaly Detection": common.anomaly,
    "Predictive Maintenance": common.maintenance,
    "Fault Analysis": common.fault_analysis,
    "Explainable AI": common.explainability,
    "What-If Simulation": common.simulator,
    "Model Performance": common.evaluation,
    "About System": common.about,
}
TOOLS_PAGES = {
    "Data Explorer": common.data_explorer,
    "Deep Learning": common.deep_learning,
    "Model Training": common.training,
    "Reports & Downloads": common.reports,
    "Settings": common.settings,
}

with st.sidebar:
    st.markdown('''<div class="brand"><div class="brand-mark">M</div><div><div class="brand-title">MILL INTELLIGENCE</div><div class="brand-sub">AI COMMAND CENTER · V1.1</div></div></div>''', unsafe_allow_html=True)
    st.markdown('<div class="nav-caption">Industrial intelligence</div>', unsafe_allow_html=True)
    st.selectbox(
        "User mode",
        ["Simple", "Engineer"],
        help="Simple mode explains results in plain language. Engineer mode exposes detailed process and model information.",
        key="user_mode",
    )
    primary_choice = st.radio("Primary navigation", list(PRIMARY_PAGES), label_visibility="collapsed", key="primary_navigation")
    st.markdown('<div class="nav-caption">Workspace & tools</div>', unsafe_allow_html=True)
    tool_choice = st.radio("Workspace tools", ["None"] + list(TOOLS_PAGES), label_visibility="collapsed", key="workspace_tools")
    st.divider()
    st.markdown('<span class="live-pill"><i class="live-dot"></i> LOCAL MODEL ARTIFACTS</span>', unsafe_allow_html=True)
    st.caption("Decision-support prototype · not a plant control system")

project_root = Path(__file__).resolve().parent
render_topbar(
    model_ready=(project_root / "models" / "classification" / "anomaly_classifier.joblib").exists(),
    data_ready=(project_root / "data" / "raw").exists(),
)
(TOOLS_PAGES[tool_choice] if tool_choice != "None" else PRIMARY_PAGES[primary_choice])()
st.markdown('<footer class="footer">AI-Powered Cold Rolling Mill Intelligence<br>Industrial AI · Mechanical Engineering · Predictive Maintenance<br>Built for industrial analytics and engineering intelligence · Model v1.0 · Application v1.1</footer>', unsafe_allow_html=True)
