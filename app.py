"""Streamlit entry point for Cold Rolling Mill AI Command Center."""
from __future__ import annotations
import streamlit as st
from pages import common

st.set_page_config(page_title="Cold Rolling Mill AI", page_icon="🏭", layout="wide", initial_sidebar_state="expanded")
common.inject_theme()
PAGES = {
    "🏭 Executive Dashboard": common.dashboard, "📊 Data Explorer": common.data_explorer,
    "⚙️ Process Monitoring": common.monitoring, "🚨 Anomaly Detection": common.anomaly,
    "🔧 Predictive Maintenance": common.maintenance, "🤖 AI Prediction": common.prediction,
    "🧠 Deep Learning": common.deep_learning, "🔍 Explainable AI": common.explainability,
    "🎛️ What-If Simulator": common.simulator, "🧪 Model Training": common.training,
    "📈 Model Evaluation": common.evaluation, "📋 Reports": common.reports,
    "⚙️ Settings": common.settings, "ℹ️ About Project": common.about,
}
with st.sidebar:
    st.markdown('''<div class="brand"><div class="brand-mark">M</div><div><div class="brand-title">MILL INTELLIGENCE</div><div class="brand-sub">AI COMMAND CENTER v1.0</div></div></div>''', unsafe_allow_html=True)
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.markdown('<span class="live-pill"><i class="live-dot"></i> SYSTEM READY</span>', unsafe_allow_html=True)
    st.caption("Local, reproducible industrial-AI prototype")
PAGES[choice]()
st.markdown('<div class="footer">Cold Rolling Mill AI • Decision-support prototype • Not a plant safety or control system</div>', unsafe_allow_html=True)
