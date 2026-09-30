"""Application shell helpers shared by all Streamlit views."""
from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st


def render_topbar(model_ready: bool, data_ready: bool) -> None:
    """Render the compact system-status bar without claiming plant connectivity."""
    model = "READY" if model_ready else "UNAVAILABLE"
    data = "READY" if data_ready else "UNAVAILABLE"
    stamp = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")
    st.markdown(
        f'''<header class="topbar"><div class="topbar-brand"><div class="topbar-logo">M</div><div><div class="topbar-title">Cold Rolling Mill Intelligence</div><div class="topbar-meta">MODEL {model} &nbsp;|&nbsp; DATA {data} &nbsp;|&nbsp; UI {stamp}</div></div></div><div class="system-status"><span class="status-dot"></span>SYSTEM ONLINE</div></header>''',
        unsafe_allow_html=True,
    )
