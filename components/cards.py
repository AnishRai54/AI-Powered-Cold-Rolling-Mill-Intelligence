"""Small, reusable HTML presentation helpers."""
from __future__ import annotations

from html import escape
from typing import Iterable

import streamlit as st


def metric_grid(items: Iterable[tuple[str, str, str, str]]) -> None:
    """Render responsive metric cards: label, value, supporting note, tone."""
    cards = []
    for label, value, note, tone in items:
        tone_class = tone if tone in {"warning", "danger"} else ""
        cards.append(
            f'<article class="metric-card {tone_class}"><span class="metric-icon" aria-hidden="true">&#9670;</span>'
            f'<div class="metric-label">{escape(label)}</div><strong class="metric-value">{escape(value)}</strong>'
            f'<div class="metric-note">{escape(note)}</div></article>'
        )
    st.markdown(f'<section class="metric-grid">{"".join(cards)}</section>', unsafe_allow_html=True)


def status_badge(label: str) -> None:
    tone = label.lower().replace(" ", "-")
    st.markdown(f'<span class="status-badge {escape(tone)}">{escape(label)}</span>', unsafe_allow_html=True)
