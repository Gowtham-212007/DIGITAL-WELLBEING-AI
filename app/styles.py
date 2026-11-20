"""
UI Styling, Custom CSS, and Dark Glassmorphism Plotly Theme Helpers
for Digital Well-Being AI.
"""

import plotly.graph_objects as go
import streamlit as st


def apply_dark_plotly_theme(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply consistent dark glassmorphism theme to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        autosize=True,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,18,38,0.45)",
        font=dict(family="Inter, system-ui, sans-serif", color="#E2E8F0", size=13),
        margin=dict(l=24, r=24, t=38, b=28),
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="#38BDF8", font=dict(color="#F8FAFC", size=12)),
    )
    return fig


def show_interactive_chart(fig: go.Figure, chart_key: str, height: int = 380) -> None:
    """Render an interactive Plotly chart with consistent dark styling and toolbar."""
    apply_dark_plotly_theme(fig, height=height)
    config = {
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": False,
        "doubleClick": "reset",
        "toImageButtonOptions": {
            "format": "png",
            "filename": chart_key,
            "height": 900,
            "width": 1600,
            "scale": 2,
        },
    }
    try:
        st.plotly_chart(fig, use_container_width=True, config=config, key=chart_key)
    except TypeError:
        st.plotly_chart(fig, config=config, key=chart_key)


def inject_custom_css() -> None:
    """Inject modern organized dark glassmorphic UI styles."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        .stApp {
            background: radial-gradient(circle at 15% 10%, #0F172A 0%, #070914 55%, #05070F 100%);
            color: #F8FAFC;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 3rem;
            max-width: 1460px;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #090D1C 0%, #060913 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.14);
        }

        .hero-container {
            background: linear-gradient(135deg, rgba(30, 27, 75, 0.88) 0%, rgba(15, 23, 42, 0.92) 50%, rgba(8, 47, 73, 0.88) 100%);
            border: 1px solid rgba(99, 102, 241, 0.35);
            border-radius: 20px;
            padding: 24px 30px;
            margin-bottom: 18px;
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.42);
        }

        .hero-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 12px;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(16, 185, 129, 0.14);
            border: 1px solid rgba(16, 185, 129, 0.38);
            color: #34D399;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
        }

        .info-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.32);
            color: #7DD3FC;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(17, 24, 39, 0.92), rgba(11, 15, 29, 0.92));
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
        }

        div[data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
            font-weight: 600 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #F8FAFC !important;
            font-weight: 800 !important;
        }

        .section-heading {
            font-size: 20px;
            font-weight: 800;
            color: #F8FAFC;
            margin-top: 12px;
            margin-bottom: 4px;
        }

        .section-description {
            color: #94A3B8;
            font-size: 13.5px;
            margin-bottom: 16px;
        }

        .glass-card {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }

        .rec-card {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.85), rgba(11, 17, 32, 0.92));
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-left: 4px solid #6366F1;
            border-radius: 14px;
            padding: 14px 18px;
            margin-bottom: 12px;
        }

        .rec-priority {
            font-size: 11px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 6px;
            background: rgba(99, 102, 241, 0.18);
            color: #A5B4FC;
            margin-left: 8px;
        }

        .footer-text {
            text-align: center;
            color: #64748B;
            font-size: 12.5px;
            margin-top: 32px;
            padding-top: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
