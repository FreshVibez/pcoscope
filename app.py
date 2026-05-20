"""Streamlit app for PCOScope."""

from __future__ import annotations

import base64
import html
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.explain import global_feature_importance, recommended_follow_up, risk_category, top_prediction_factors
from src.qwen_assistant import generate_assistant_reply, generate_qwen_explanation, qwen_is_configured, qwen_provider_name
from src.train import DATA_PATH, MODEL_PATH, train_best_model

APP_DIR = Path(__file__).resolve().parent
ASSISTANT_ASSET_PATH = APP_DIR / "assets" / "assistant-character.png"


st.set_page_config(
    page_title="PCOScope",
    page_icon="PCOScope",
    layout="wide",
    initial_sidebar_state="expanded",
)


FIELD_GROUPS = {
    "Basic Profile": [
        "Age (yrs)",
        "Weight (Kg)",
        "Height(Cm)",
        "BMI",
        "Blood Group",
        "Pulse rate(bpm)",
        "RR (breaths/min)",
        "Hb(g/dl)",
        "Marraige Status (Yrs)",
        "Pregnant(Y/N)",
        "No. of abortions",
        "BP _Systolic (mmHg)",
        "BP _Diastolic (mmHg)",
    ],
    "Cycle Pattern": [
        "Cycle(R/I)",
        "Cycle length(days)",
    ],
    "Hormonal Markers": [
        "I   beta-HCG(mIU/mL)",
        "II    beta-HCG(mIU/mL)",
        "FSH(mIU/mL)",
        "LH(mIU/mL)",
        "FSH/LH",
        "TSH (mIU/L)",
        "AMH(ng/mL)",
        "PRL(ng/mL)",
        "Vit D3 (ng/mL)",
        "PRG(ng/mL)",
    ],
    "Metabolic Markers": [
        "RBS(mg/dl)",
        "Waist(inch)",
        "Hip(inch)",
        "Waist:Hip Ratio",
        "Weight gain(Y/N)",
    ],
    "Symptoms And Lifestyle": [
        "hair growth(Y/N)",
        "Skin darkening (Y/N)",
        "Hair loss(Y/N)",
        "Pimples(Y/N)",
        "Fast food (Y/N)",
        "Reg.Exercise(Y/N)",
    ],
    "Ultrasound Features": [
        "Follicle No. (L)",
        "Follicle No. (R)",
        "Avg. F size (L) (mm)",
        "Avg. F size (R) (mm)",
        "Endometrium (mm)",
    ],
}


QUICK_QUESTIONS = [
    "What does my risk score mean?",
    "Why do these factors matter?",
    "What should I ask my clinician?",
    "What symptoms overlap with endometriosis?",
]


NAV_ITEMS = [
    "Overview",
    "Risk Screening",
    "Results",
    "Explainability",
    "AI Assistant",
    "Clinical Rationale",
    "Model Performance",
]

NAV_ICONS = {
    "Overview": "▦",
    "Risk Screening": "▤",
    "Results": "◉",
    "Explainability": "✧",
    "AI Assistant": "♡",
    "Clinical Rationale": "♧",
    "Model Performance": "⌁",
}


def image_data_uri(path: Path) -> str:
    """Return a local image as a CSS-safe data URI."""

    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@st.cache_resource
def load_or_train_model() -> dict:
    """Load the saved model or train one on first app launch."""

    if Path(MODEL_PATH).exists():
        return joblib.load(MODEL_PATH)
    return train_best_model(DATA_PATH, MODEL_PATH)


def apply_theme() -> None:
    """Apply the PCOScope visual system."""

    assistant_image = image_data_uri(ASSISTANT_ASSET_PATH)
    assistant_background = (
        f'background-image: url("{assistant_image}") !important;'
        if assistant_image
        else ""
    )

    st.markdown(
        """
        <style>
        :root {
            --pcos-bg: #fff8fb;
            --pcos-surface: #ffffff;
            --pcos-sidebar: #fff5fa;
            --pcos-soft: #f8e4f0;
            --pcos-soft-2: #fdeef5;
            --pcos-border: #efd8e5;
            --pcos-border-strong: #e4bfd3;
            --pcos-ink: #2b1826;
            --pcos-muted: #735c6d;
            --pcos-accent: #f3b4cf;
            --pcos-accent-2: #eca0c2;
            --pcos-accent-text: #3b2032;
            --pcos-shadow: 0 10px 26px rgba(90, 31, 73, 0.10);
        }

        .stApp {
            background: var(--pcos-bg);
            color: var(--pcos-ink);
        }

        .block-container {
            max-width: 1240px;
            padding-top: 3.6rem;
            padding-bottom: 5rem;
        }

        section[data-testid="stSidebar"] {
            background: var(--pcos-sidebar) !important;
            border-right: 1px solid var(--pcos-border);
        }

        section[data-testid="stSidebar"] * {
            color: var(--pcos-ink) !important;
        }

        .sidebar-brand {
            border-bottom: 1px solid var(--pcos-border);
            padding: 0.15rem 0 1.15rem 0;
            margin-bottom: 1.35rem;
        }

        .sidebar-brand-row {
            display: flex;
            gap: 0.75rem;
            align-items: center;
        }

        .sidebar-mark {
            width: 2.8rem;
            height: 2.8rem;
            border-radius: 999px;
            background: var(--pcos-accent);
            color: var(--pcos-accent-text) !important;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 0.9rem;
        }

        .sidebar-title {
            color: var(--pcos-accent-text) !important;
            font-weight: 800;
            font-size: 1.35rem;
            line-height: 1.05;
        }

        .sidebar-subtitle {
            color: var(--pcos-muted) !important;
            font-size: 0.86rem;
            margin-top: 0.1rem;
        }

        .sidebar-footer {
            border-top: 1px solid var(--pcos-border);
            margin-top: 2rem;
            padding-top: 0.85rem;
            color: var(--pcos-muted) !important;
            font-size: 0.82rem;
            line-height: 1.4;
        }

        .stRadio [role="radiogroup"] {
            gap: 0.22rem;
        }

        .stRadio label {
            border-radius: 12px;
            padding: 0.55rem 0.65rem;
        }

        .stRadio label:has(input:checked) {
            background: var(--pcos-soft);
            color: var(--pcos-accent-text) !important;
            font-weight: 750;
        }

        h1, h2, h3, h4, h5, h6,
        p, li, label,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stWidgetLabel"],
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
            color: var(--pcos-ink) !important;
            letter-spacing: 0;
        }

        h1 {
            font-size: 2.25rem;
            line-height: 1.08;
            margin-bottom: 0.45rem;
            font-weight: 800;
        }

        h2 {
            font-size: 1.45rem;
        }

        h3 {
            font-size: 1.08rem;
        }

        .stCaptionContainer,
        [data-testid="stCaptionContainer"],
        [data-testid="stMarkdownContainer"] small {
            color: var(--pcos-muted) !important;
        }

        input, textarea, select,
        [data-baseweb="input"],
        [data-baseweb="select"] {
            color: var(--pcos-ink) !important;
            background: #ffffff !important;
        }

        input, textarea,
        [data-baseweb="input"] {
            border-color: var(--pcos-border) !important;
        }

        [data-testid="stTabs"] [role="tablist"] {
            gap: 0.35rem;
            border-bottom: 1px solid var(--pcos-border);
        }

        [data-testid="stTabs"] [role="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 0.65rem 0.9rem;
            color: var(--pcos-muted) !important;
            background: transparent;
        }

        [data-testid="stTabs"] [aria-selected="true"] {
            background: #ffffff;
            color: var(--pcos-accent-text) !important;
            border-bottom: 2px solid var(--pcos-accent);
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--pcos-border);
            border-radius: 14px;
            padding: 1rem;
            box-shadow: var(--pcos-shadow);
        }

        div[data-testid="stMetric"] label {
            color: var(--pcos-muted) !important;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--pcos-border);
            border-radius: 16px;
            background: #ffffff;
            box-shadow: 0 5px 16px rgba(90, 31, 73, 0.06);
        }

        .pcos-banner {
            border: 1px solid var(--pcos-border);
            border-radius: 28px;
            background: linear-gradient(115deg, #ffe4eb 0%, #fff8fb 52%, #fff1f7 100%);
            padding: 3rem 3.2rem;
            margin: 1.2rem 0 1.35rem 0;
            box-shadow: var(--pcos-shadow);
        }

        .pcos-kicker {
            color: var(--pcos-accent-text);
            font-size: 0.86rem;
            font-weight: 700;
            letter-spacing: 0;
            margin-bottom: 1.15rem;
            display: inline-flex;
            align-items: center;
            border: 1px solid var(--pcos-border);
            border-radius: 999px;
            padding: 0.34rem 0.7rem;
            background: rgba(255, 255, 255, 0.78);
        }

        .pcos-subtitle {
            color: var(--pcos-muted);
            max-width: 760px;
            line-height: 1.55;
            margin: 0.35rem 0 0.9rem 0;
            font-size: 1.08rem;
        }

        .pcos-note {
            background: #ffffff;
            border: 1px solid var(--pcos-border);
            border-radius: 16px;
            padding: 1rem 1.15rem;
            color: var(--pcos-ink);
            margin: 0.9rem 0 1.35rem 0;
            box-shadow: 0 5px 16px rgba(90, 31, 73, 0.05);
        }

        .pcos-panel {
            background: #ffffff;
            border: 1px solid var(--pcos-border);
            border-radius: 18px;
            padding: 1.35rem;
            box-shadow: var(--pcos-shadow);
            min-height: 100%;
        }

        .pcos-panel-title {
            font-weight: 700;
            color: var(--pcos-ink);
            margin-bottom: 0.35rem;
            font-size: 1.02rem;
        }

        .pcos-small {
            color: var(--pcos-muted);
            font-size: 0.9rem;
            line-height: 1.45;
        }

        .pcos-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            border: 1px solid var(--pcos-border);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.78);
            padding: 0.62rem 0.85rem;
            margin-bottom: 1.65rem;
            box-shadow: 0 5px 16px rgba(90, 31, 73, 0.05);
        }

        .pcos-topbar-title {
            color: var(--pcos-ink);
            font-weight: 760;
        }

        .pcos-topbar-subtitle {
            color: var(--pcos-muted);
            margin-left: 0.35rem;
        }

        .pcos-status {
            color: var(--pcos-muted);
            border: 1px solid var(--pcos-border);
            border-radius: 999px;
            padding: 0.24rem 0.65rem;
            background: #ffffff;
            white-space: nowrap;
            font-size: 0.86rem;
        }

        .pcos-dot {
            display: inline-block;
            width: 0.52rem;
            height: 0.52rem;
            border-radius: 999px;
            background: var(--pcos-accent-2);
            margin-right: 0.45rem;
        }

        .pcos-icon {
            width: 2.55rem;
            height: 2.55rem;
            border-radius: 999px;
            background: var(--pcos-soft);
            color: var(--pcos-accent-text);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            margin-bottom: 1.25rem;
        }

        .pcos-card-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
            margin: 1.35rem 0 2rem 0;
        }

        .pcos-step-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .pcos-step-number {
            color: #b997aa;
            font-size: 2rem;
            font-weight: 850;
            margin-bottom: 0.65rem;
        }

        .pcos-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-top: 0.85rem;
        }

        .pcos-pill {
            background: var(--pcos-soft-2);
            color: var(--pcos-accent-text);
            border-radius: 999px;
            padding: 0.18rem 0.58rem;
            font-size: 0.83rem;
        }

        .pcos-bar-row {
            margin: 0.92rem 0;
        }

        .pcos-bar-head {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.28rem;
            color: var(--pcos-ink);
            font-weight: 650;
        }

        .pcos-bar-track {
            height: 0.58rem;
            background: var(--pcos-soft-2);
            border-radius: 999px;
            overflow: hidden;
        }

        .pcos-bar-fill {
            height: 100%;
            background: var(--pcos-accent);
            border-radius: 999px;
        }

        .pcos-factor-list {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.95rem;
            margin-top: 1rem;
        }

        .pcos-factor-card {
            border: 1px solid var(--pcos-border);
            border-radius: 16px;
            padding: 1rem;
            background: #ffffff;
        }

        .stButton > button,
        .stFormSubmitButton > button {
            border-radius: 999px;
            border: 1px solid var(--pcos-accent);
            background: var(--pcos-accent);
            color: var(--pcos-accent-text);
            font-weight: 650;
            min-height: 2.65rem;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            border-color: var(--pcos-accent-2);
            background: var(--pcos-accent-2);
            color: var(--pcos-accent-text);
        }

        .stDataFrame {
            border: 1px solid var(--pcos-border);
            border-radius: 14px;
        }

        div[data-testid="stAlert"] {
            border-radius: 14px;
        }

        .st-key-assistant_fab {
            position: fixed;
            right: 1.25rem;
            bottom: 1.25rem;
            z-index: 9999;
            width: 4.5rem;
        }

        .st-key-assistant_fab button {
            width: 4.5rem !important;
            height: 4.5rem !important;
            border-radius: 999px !important;
            padding: 0 !important;
            box-shadow: 0 12px 32px rgba(31, 41, 51, 0.22);
            background-color: #ffffff !important;
            __ASSISTANT_BACKGROUND__
            background-size: 92% auto !important;
            background-repeat: no-repeat !important;
            background-position: center bottom !important;
            border: 2px solid var(--pcos-border) !important;
            color: transparent !important;
            font-size: 0 !important;
            overflow: hidden !important;
            animation: pcos-float 2.8s ease-in-out infinite;
        }

        .st-key-assistant_fab button *,
        .st-key-assistant_fab button p,
        .st-key-assistant_fab button span,
        .st-key-assistant_fab button div {
            color: transparent !important;
            font-size: 0 !important;
            line-height: 0 !important;
            opacity: 0 !important;
        }

        .st-key-assistant_fab button:hover {
            background-color: #fff8fc !important;
            border-color: var(--pcos-accent) !important;
        }

        .st-key-assistant_fab:before {
            content: "Ask AI";
            position: absolute;
            right: 4.9rem;
            bottom: 1.25rem;
            background: #ffffff;
            color: var(--pcos-accent-text);
            border: 1px solid var(--pcos-border);
            border-radius: 999px;
            padding: 0.45rem 0.75rem;
            font-weight: 750;
            box-shadow: 0 8px 22px rgba(31, 41, 51, 0.16);
            opacity: 0;
            transform: translateX(0.35rem);
            transition: all 0.18s ease;
            pointer-events: none;
            white-space: nowrap;
        }

        .st-key-assistant_fab:hover:before {
            opacity: 1;
            transform: translateX(0);
        }

        @keyframes pcos-float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-0.35rem); }
        }

        .st-key-assistant_drawer {
            position: fixed;
            right: 1.25rem;
            bottom: 6.35rem;
            width: 430px;
            max-width: calc(100vw - 2.5rem);
            height: min(640px, calc(100vh - 7.6rem));
            overflow-y: auto;
            z-index: 9998;
            background: #ffffff;
            border: 1px solid var(--pcos-border);
            border-radius: 22px;
            padding: 0;
            box-shadow: 0 18px 48px rgba(31, 41, 51, 0.24);
        }

        .st-key-assistant_drawer > div {
            position: relative;
        }

        .assistant-window {
            display: flex;
            flex-direction: column;
            min-height: 0;
        }

        .assistant-header {
            position: relative;
            z-index: 3;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            padding: 1.05rem 3.75rem 1.05rem 1.25rem;
            background: rgba(255, 255, 255, 0.96);
            border-bottom: 1px solid var(--pcos-border);
            border-radius: 22px 22px 0 0;
            backdrop-filter: blur(8px);
        }

        .assistant-title {
            font-weight: 800;
            color: var(--pcos-ink);
            line-height: 1.2;
        }

        .assistant-subtitle {
            color: var(--pcos-muted);
            font-size: 0.82rem;
            margin-top: 0.1rem;
        }

        .assistant-online {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            color: var(--pcos-muted);
            font-size: 0.82rem;
            font-weight: 700;
            margin-top: 0.18rem;
        }

        .assistant-online:before {
            content: "";
            width: 0.48rem;
            height: 0.48rem;
            border-radius: 999px;
            background: var(--pcos-accent);
            box-shadow: 0 0 0 4px rgba(243, 180, 207, 0.25);
        }

        .assistant-body {
            padding: 1.25rem 1.35rem 1.35rem;
        }

        .st-key-assistant_drawer .assistant-body {
            display: flex;
            flex-direction: column;
            min-height: 0;
            flex: 1;
            padding: 1.15rem 1.35rem 0;
            overflow: hidden;
        }

        .assistant-status {
            border: 1px solid var(--pcos-border);
            background: #fff8fb;
            color: var(--pcos-ink);
            border-radius: 16px;
            padding: 1rem 1.15rem;
            line-height: 1.45;
            margin-bottom: 1rem;
        }

        .assistant-prompt-title {
            color: var(--pcos-muted);
            font-size: 0.86rem;
            font-weight: 700;
            margin: 0.7rem 0 0.55rem 0;
        }

        .assistant-prompt-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.55rem;
            margin-bottom: 0.85rem;
        }

        .assistant-messages {
            border-top: 1px solid var(--pcos-border);
            margin-top: 1rem;
            padding-top: 1rem;
            padding-bottom: 0.85rem;
            overflow-y: auto;
            min-height: 190px;
            flex: 1;
        }

        .assistant-message {
            display: flex;
            gap: 0.8rem;
            align-items: flex-start;
            margin: 0.9rem 0;
        }

        .assistant-message.user {
            flex-direction: row-reverse;
        }

        .assistant-avatar {
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex: 0 0 auto;
            background: var(--pcos-soft);
            color: var(--pcos-accent-text);
            font-weight: 800;
        }

        .assistant-bubble {
            border: 1px solid var(--pcos-border);
            border-radius: 18px;
            padding: 0.95rem 1.1rem;
            color: var(--pcos-ink);
            background: #ffffff;
            line-height: 1.45;
            max-width: 82%;
            width: fit-content;
        }

        .assistant-message.user .assistant-bubble {
            background: var(--pcos-soft-2);
            border-color: var(--pcos-border-strong);
            text-align: left;
        }

        .assistant-message.user .assistant-avatar {
            background: var(--pcos-accent);
        }

        .assistant-typing {
            display: inline-flex;
            gap: 0.2rem;
            align-items: center;
            min-width: 2.8rem;
        }

        .assistant-typing span {
            width: 0.36rem;
            height: 0.36rem;
            border-radius: 999px;
            background: var(--pcos-accent);
            opacity: 0.35;
            animation: assistant-dot 1.15s infinite ease-in-out;
        }

        .assistant-typing span:nth-child(2) { animation-delay: 0.18s; }
        .assistant-typing span:nth-child(3) { animation-delay: 0.36s; }

        @keyframes assistant-dot {
            0%, 80%, 100% { opacity: 0.3; transform: translateY(0); }
            40% { opacity: 1; transform: translateY(-0.16rem); }
        }

        .st-key-assistant_drawer .stButton > button {
            min-height: 2.55rem;
            font-size: 0.85rem;
            padding-left: 0.9rem !important;
            padding-right: 0.9rem !important;
        }

        .st-key-assistant_drawer .st-key-assistant_close_wrap {
            position: absolute;
            top: 0.9rem;
            right: 1rem;
            z-index: 10001;
            width: 2.25rem;
            height: 0;
        }

        .st-key-assistant_drawer .st-key-assistant_close_wrap button {
            min-height: 2.25rem !important;
            width: 2.25rem !important;
            border-radius: 999px !important;
            padding: 0 !important;
            background: var(--pcos-soft) !important;
            border-color: var(--pcos-border) !important;
            color: var(--pcos-accent-text) !important;
            box-shadow: 0 6px 16px rgba(90, 31, 73, 0.16) !important;
        }

        .st-key-assistant_drawer div[data-testid="stForm"] {
            position: sticky;
            bottom: 0;
            z-index: 6;
            background: linear-gradient(180deg, rgba(255,255,255,0.88), #ffffff 18%);
            border-top: 1px solid var(--pcos-border);
            padding: 0.95rem 0.2rem 0.8rem;
            margin: 0;
            border-radius: 0 0 22px 22px;
        }

        .st-key-assistant_drawer div[data-testid="stForm"] input {
            min-height: 2.65rem;
        }

        @media (max-width: 980px) {
            .block-container {
                padding-top: 3.2rem;
            }

            .pcos-card-grid,
            .pcos-step-grid,
            .pcos-factor-list {
                grid-template-columns: 1fr;
            }

            .pcos-topbar {
                align-items: flex-start;
                border-radius: 18px;
                flex-direction: column;
            }

            .pcos-banner {
                padding: 2rem;
            }

            .st-key-assistant_drawer {
                right: 0.75rem;
                bottom: 5.75rem;
                width: calc(100vw - 1.5rem);
                height: calc(100vh - 6.8rem);
            }

            .assistant-window {
                min-height: 0;
            }

            .st-key-assistant_drawer .st-key-assistant_close_wrap {
                top: 0.9rem;
                right: 1rem;
            }
        }
        </style>
        """.replace("__ASSISTANT_BACKGROUND__", assistant_background),
        unsafe_allow_html=True,
    )


def friendly_feature_name(name: str) -> str:
    """Clean transformed feature names for display."""

    return (
        name.replace("num__", "")
        .replace("cat__", "")
        .replace("_", " ")
        .replace("classifier__", "")
        .strip()
    )


def page_title(icon: str, title: str, subtitle: str) -> None:
    """Render a consistent page heading."""

    st.markdown(
        f"""
        <div style="display:flex; gap:1rem; align-items:center; margin:0.6rem 0 1.6rem 0;">
            <div class="pcos-icon" style="margin-bottom:0;">{html.escape(icon)}</div>
            <div>
                <h1 style="margin:0;">{html.escape(title)}</h1>
                <p class="pcos-subtitle">{html.escape(subtitle)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def bar_rows_markup(rows: list[tuple[str, float]], scale: float | None = None) -> str:
    """Build horizontal progress-bar markup in the app palette."""

    if not rows:
        return '<div class="pcos-small">No values available.</div>'

    max_value = scale if scale is not None else max(value for _, value in rows) or 1
    markup = []
    for label, value in rows:
        width = max(2, min(100, (value / max_value) * 100))
        display_value = f"{value:.0f}%" if scale is not None else f"{width:.0f}% rel."
        markup.append(
            '<div class="pcos-bar-row">'
            '<div class="pcos-bar-head">'
            f'<span>{html.escape(label)}</span>'
            f'<span>{display_value}</span>'
            '</div>'
            f'<div class="pcos-bar-track"><div class="pcos-bar-fill" style="width:{width:.1f}%"></div></div>'
            '</div>'
        )
    return "".join(markup)


def render_bar_rows(rows: list[tuple[str, float]], scale: float | None = None) -> None:
    """Render horizontal progress bars in the app palette."""

    st.markdown(bar_rows_markup(rows, scale), unsafe_allow_html=True)


def render_factor_cards(rows: list[tuple[str, str]]) -> None:
    """Render explanatory factor cards."""

    markup = []
    for title, body in rows:
        markup.append(
            '<div class="pcos-factor-card">'
            '<div style="display:flex; gap:0.75rem; align-items:flex-start;">'
            '<div class="pcos-icon" style="width:1.9rem;height:1.9rem;margin:0;font-size:0.85rem;">i</div>'
            '<div>'
            f'<div class="pcos-panel-title">{html.escape(title)}</div>'
            f'<div class="pcos-small">{html.escape(body)}</div>'
            '</div>'
            '</div>'
            '</div>'
        )
    st.markdown(f'<div class="pcos-factor-list">{"".join(markup)}</div>', unsafe_allow_html=True)


def render_header() -> None:
    """Render the app-level top bar."""

    st.markdown(
        """
        <div class="pcos-topbar">
            <div>
                <span class="pcos-topbar-title">PCOScope</span>
                <span class="pcos-topbar-subtitle">Explainable PCOS Risk Screening</span>
            </div>
            <div class="pcos-status"><span class="pcos-dot"></span>Research prototype, not for clinical use</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_tab() -> None:
    """Render the landing workspace."""

    st.markdown(
        """
        <div class="pcos-banner">
            <div class="pcos-kicker">♡ Women's health decision support</div>
            <h1>PCOScope</h1>
            <p class="pcos-subtitle">
                Explainable PCOS risk screening assistant for structured clinical review.
                Enter available clinical features, inspect contributing factors, and prepare
                a sensible follow-up conversation with a clinician.
            </p>
        </div>
        <div class="pcos-note">
            <strong>A screening tool, not a diagnosis.</strong><br>
            PCOScope estimates risk and supports clinical decision-making. It does not diagnose
            PCOS, prescribe treatment, or replace assessment by a qualified clinician.
        </div>
        <h2>What PCOScope Does</h2>
        <p class="pcos-subtitle">A focused workspace for screening, explanation, and clinician handoff.</p>
        <div class="pcos-card-grid">
            <div class="pcos-panel">
                <div class="pcos-icon">▤</div>
                <div class="pcos-panel-title">Structured screening</div>
                <div class="pcos-small">Capture clinical, hormonal, metabolic, symptom, and ultrasound features in one organised intake.</div>
            </div>
            <div class="pcos-panel">
                <div class="pcos-icon">✧</div>
                <div class="pcos-panel-title">Transparent explanations</div>
                <div class="pcos-small">Every score includes the contributing factors and why they may matter.</div>
            </div>
            <div class="pcos-panel">
                <div class="pcos-icon">♧</div>
                <div class="pcos-panel-title">Clinician-friendly</div>
                <div class="pcos-small">Frames outputs as risk categories and recommended follow-up, not diagnoses.</div>
            </div>
            <div class="pcos-panel">
                <div class="pcos-icon">⌁</div>
                <div class="pcos-panel-title">Performance honesty</div>
                <div class="pcos-small">Recall and F1 are prioritised because missed cases carry the highest cost in screening.</div>
            </div>
        </div>
        <div class="pcos-step-grid">
            <div class="pcos-panel">
                <div class="pcos-step-number">01</div>
                <div class="pcos-panel-title">Enter clinical features</div>
                <div class="pcos-small">Profile, cycle, hormones, metabolic markers, symptoms, and ultrasound findings.</div>
            </div>
            <div class="pcos-panel">
                <div class="pcos-step-number">02</div>
                <div class="pcos-panel-title">Get an explained risk score</div>
                <div class="pcos-small">Low, moderate, or high risk with the top contributing factors made visible.</div>
            </div>
            <div class="pcos-panel">
                <div class="pcos-step-number">03</div>
                <div class="pcos-panel-title">Plan clinician follow-up</div>
                <div class="pcos-small">Use recommended next steps and rationale to guide the conversation.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render left navigation and return the selected page."""

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-row">
                <div class="sidebar-mark">PC</div>
                <div>
                    <div class="sidebar-title">PCOScope</div>
                    <div class="sidebar-subtitle">Risk Screening Assistant</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("**Workspace**")
    page = st.sidebar.radio(
        "Workspace",
        NAV_ITEMS,
        label_visibility="collapsed",
        format_func=lambda item: f"{NAV_ICONS[item]}  {item}",
    )
    st.sidebar.markdown(
        """
        <div class="sidebar-footer">
            Prototype only. Not a medical diagnosis.<br>
            Always consult a qualified clinician.
        </div>
        """,
        unsafe_allow_html=True,
    )
    return page


def default_patient_input(model_bundle: dict) -> dict:
    """Build default input values from training metadata."""

    defaults = {}
    for col, info in model_bundle["input_summary"].items():
        if info["type"] == "numeric":
            defaults[col] = info["median"]
        else:
            defaults[col] = info.get("mode", "")
    return defaults


def render_field(col: str, info: dict, current_value, key: str):
    """Render one typed input widget."""

    label = friendly_feature_name(col)

    if info["type"] == "numeric" and info.get("unique") == [0, 1]:
        options = [0, 1]
        index = options.index(int(current_value)) if int(current_value) in options else 0
        return st.selectbox(label, options=options, index=index, format_func=lambda x: "Yes" if x == 1 else "No", key=key)

    if info["type"] == "numeric" and info.get("unique") is not None:
        options = info["unique"]
        index = options.index(current_value) if current_value in options else 0
        return st.selectbox(label, options=options, index=index, key=key)

    if info["type"] == "numeric":
        span = float(info["max"]) - float(info["min"])
        step = max(span / 100, 0.1)
        return st.number_input(
            label,
            min_value=float(info["min"]),
            max_value=float(info["max"]),
            value=float(current_value),
            step=step,
            key=key,
        )

    options = info["values"]
    fallback = options[0] if options else ""
    value = str(current_value or fallback)
    index = options.index(value) if value in options else 0
    return st.selectbox(label, options=options, index=index, key=key)


def grouped_columns(model_bundle: dict) -> dict[str, list[str]]:
    """Return form groups, preserving any unexpected dataset columns."""

    available = set(model_bundle["training_columns"])
    grouped = {name: [col for col in cols if col in available] for name, cols in FIELD_GROUPS.items()}
    used = {col for cols in grouped.values() for col in cols}
    remaining = [col for col in model_bundle["training_columns"] if col not in used]
    if remaining:
        grouped["Other Clinical Features"] = remaining
    return grouped


def render_screening_tab(model_bundle: dict) -> None:
    """Render the clinical input workflow."""

    page_title(
        "▤",
        "Risk Screening",
        "Enter the features you have. Empty fields are not stored, and results remain non-diagnostic.",
    )

    if "patient_values" not in st.session_state:
        st.session_state["patient_values"] = default_patient_input(model_bundle)

    values = st.session_state["patient_values"].copy()
    summary = model_bundle["input_summary"]

    form_col, result_col = st.columns([2.1, 1], gap="large")
    submitted = False
    with form_col:
        with st.form("screening_form"):
            for group_name, columns in grouped_columns(model_bundle).items():
                with st.expander(group_name, expanded=group_name in {"Basic Profile", "Cycle Pattern", "Symptoms And Lifestyle"}):
                    field_cols = st.columns(2)
                    for index, col in enumerate(columns):
                        with field_cols[index % 2]:
                            values[col] = render_field(col, summary[col], values.get(col), f"field_{col}")

            submitted = st.form_submit_button("Calculate risk score", width="stretch")

    if submitted:
        patient_df = pd.DataFrame([values])[model_bundle["training_columns"]]
        probability = float(model_bundle["model"].predict_proba(patient_df)[0, 1])
        category = risk_category(probability)
        follow_up = recommended_follow_up(category)
        factors = top_prediction_factors(model_bundle, patient_df, top_n=10)
        factor_names = [friendly_feature_name(name) for name in factors["feature"].head(5).tolist()]
        explanation = generate_qwen_explanation(probability, category, factor_names, follow_up)

        st.session_state["patient_values"] = values
        st.session_state["screening_result"] = {
            "patient_df": patient_df,
            "probability": probability,
            "category": category,
            "follow_up": follow_up,
            "factors": factors,
            "factor_names": factor_names,
            "explanation": explanation,
        }
        st.markdown(
            """
            <div class="pcos-note">
                Risk score calculated. Open Results for the full follow-up summary.
            </div>
            """,
            unsafe_allow_html=True,
        )

    result = st.session_state.get("screening_result")
    with result_col:
        if result:
            st.markdown(
                f"""
                <div class="pcos-panel">
                    <div class="pcos-panel-title">Screening result</div>
                    <div class="pcos-small">Score updates after you press Calculate.</div>
                    <div style="height:1px;background:#efd8e5;margin:1rem -1.35rem;"></div>
                    <div class="pcos-step-number" style="color:#d986ac;margin-bottom:0.15rem;">{result['probability']:.1%}</div>
                    <div class="pcos-panel-title">{html.escape(result['category'])} risk category</div>
                    <div class="pcos-small">Top factor: {html.escape(result['factor_names'][0] if result['factor_names'] else 'Unavailable')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="pcos-panel" style="min-height:280px;display:flex;flex-direction:column;justify-content:center;text-align:center;">
                    <div class="pcos-icon" style="margin:0 auto 1rem auto;">▤</div>
                    <div class="pcos-panel-title">Screening result</div>
                    <div class="pcos-small">Your risk score, category, and recommended follow-up will appear here.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def require_result() -> dict | None:
    """Return the current result or show a clear empty state."""

    result = st.session_state.get("screening_result")
    if not result:
        st.markdown(
            '<div class="pcos-note">Calculate a risk score in Risk Screening first.</div>',
            unsafe_allow_html=True,
        )
        return None
    return result


def render_results_tab() -> None:
    """Render risk result and follow-up."""

    page_title("◉", "Results", "Review the screening score, category, explanation, and follow-up language.")
    result = require_result()
    if not result:
        return

    score_col, category_col, follow_col = st.columns([1, 1, 2])
    score_col.metric("PCOS risk score", f"{result['probability']:.1%}")
    category_col.metric("Risk category", result["category"])
    with follow_col:
        st.markdown('<div class="pcos-panel-title">Recommended follow-up</div>', unsafe_allow_html=True)
        st.write(result["follow_up"])

    st.markdown(
        """
        <div class="pcos-note">
            The displayed probability comes from the selected model. If a calibrated model wins medical
            ranking, the score uses calibrated probability estimation; otherwise it uses the model's native
            probability output.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Patient-friendly explanation")
    st.write(result["explanation"])

    st.markdown(
        """
        <div class="pcos-note">
            Use this result to support a follow-up conversation. It should not be used as diagnostic confirmation.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_explainability_tab(model_bundle: dict) -> None:
    """Render global and patient-level explanations."""

    page_title(
        "✧",
        "Explainability",
        "Each screening score is decomposed into the features that contributed most.",
    )

    result = require_result()
    importance = global_feature_importance(model_bundle)
    patient_factors = result["factors"].copy() if result else pd.DataFrame()
    left, right = st.columns([2, 1], gap="large")

    with left:
        if result and not patient_factors.empty:
            chart_data = patient_factors.head(9)
            rows = [
                (friendly_feature_name(row["feature"]), float(row["contribution"]))
                for _, row in chart_data.iterrows()
            ]
            feature_body = bar_rows_markup(rows)
            feature_note = (
                "Patient-level contribution for the most recent screening result. "
                "The strongest factor is set to 100% and the other bars are relative to it."
            )
            feature_title = "Patient-specific contribution"
            method = patient_factors["method"].iloc[0] if "method" in patient_factors else "model explanation"
            feature_body += f'<div class="pcos-small" style="margin-top:0.75rem;">Method: {html.escape(str(method))}</div>'
        elif importance.empty:
            feature_body = '<div class="pcos-small">Feature importance is unavailable for the selected model.</div>'
            feature_note = "Calculate a screening score to show patient-specific factors."
            feature_title = "Feature importance"
        else:
            chart_data = importance.copy().head(9)
            rows = [
                (friendly_feature_name(row["feature"]), float(row["importance"]))
                for _, row in chart_data.iterrows()
            ]
            feature_body = bar_rows_markup(rows)
            feature_note = (
                "Global model importance before a patient-specific score is calculated. "
                "The strongest feature is set to 100% and the other bars are relative to it."
            )
            feature_title = "Global feature importance"
        st.markdown(
            '<div class="pcos-panel">'
            f'<div class="pcos-panel-title">{html.escape(feature_title)}</div>'
            f'<div class="pcos-small">{html.escape(feature_note)}</div>'
            f'{feature_body}'
            '</div>',
            unsafe_allow_html=True,
        )

    with right:
        if result and not patient_factors.empty:
            factors = patient_factors.head(3)
            max_contribution = float(factors["contribution"].max()) or 1.0
            factor_items = []
            for index, (_, row) in enumerate(factors.iterrows(), start=1):
                name = friendly_feature_name(row["feature"])
                value = (float(row["contribution"]) / max_contribution) * 100
                factor_items.append(
                    '<div class="pcos-factor-card" style="margin-top:0.75rem;">'
                    f'<div class="pcos-panel-title">{index}. {html.escape(name)}</div>'
                    f'<div class="pcos-pill-row"><span class="pcos-pill">{value:.0f}% relative to strongest</span></div>'
                    '</div>'
                )
            factor_body = "".join(factor_items)
        else:
            factor_body = '<div class="pcos-small" style="margin-top:0.75rem;">Calculate a screening score to see patient-level factors.</div>'
        st.markdown(
            '<div class="pcos-panel">'
            '<div class="pcos-panel-title">Top contributing factors</div>'
            '<div class="pcos-small">The strongest patient-specific signals from the most recent score.</div>'
            f'{factor_body}'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Why These Factors May Matter")
    st.markdown('<p class="pcos-subtitle">Brief, non-diagnostic notes about each feature group.</p>', unsafe_allow_html=True)
    render_factor_cards(
        [
            ("Follicle count", "Elevated follicle counts can support polycystic ovarian morphology when interpreted with ultrasound context."),
            ("Cycle irregularity", "Irregular cycles may reflect ovulatory dysfunction, a core PCOS screening domain."),
            ("Androgen-related symptoms", "Hair growth, pimples, and hair loss can provide clinical context for possible hyperandrogenism."),
            ("AMH and LH/FSH", "Hormonal markers can support endocrine review but should not be interpreted alone."),
            ("Metabolic markers", "BMI, waist-hip ratio, glucose, and blood pressure support cardiometabolic risk assessment."),
            ("Lifestyle context", "Exercise and dietary patterns can guide practical prevention and follow-up conversations."),
        ]
    )

    artifacts = model_bundle.get("shap_artifacts", {})
    if artifacts.get("available"):
        st.markdown("### SHAP Visualizations")
        image_cols = st.columns(3)
        for col, label, path in [
            (image_cols[0], "Summary plot", artifacts.get("summary_plot")),
            (image_cols[1], "Feature importance plot", artifacts.get("feature_importance_plot")),
            (image_cols[2], "Individual prediction example", artifacts.get("individual_prediction_plot")),
        ]:
            if path and Path(path).exists():
                col.image(path, caption=label, use_container_width=True)
    else:
        reason = artifacts.get("reason") or artifacts.get("error") or "SHAP plots are unavailable for this fitted model."
        st.markdown(
            f'<div class="pcos-note">SHAP plot status: {html.escape(reason)}</div>',
            unsafe_allow_html=True,
        )


def current_assistant_context() -> tuple[float | None, str | None, list[str], str | None]:
    """Return the latest screening context for assistant prompts."""

    result = st.session_state.get("screening_result")
    risk_score = result["probability"] if result else None
    risk_cat = result["category"] if result else None
    factors = result["factor_names"] if result else []
    follow_up = result["follow_up"] if result else None
    return risk_score, risk_cat, factors, follow_up


def ensure_assistant_messages() -> None:
    """Initialize assistant chat history."""

    if "assistant_messages" not in st.session_state:
        st.session_state["assistant_messages"] = [
            {
                "role": "assistant",
                "content": "I can explain PCOScope screening results, contributing factors, and clinician follow-up questions in non-diagnostic language.",
            }
        ]
    st.session_state.setdefault("assistant_thinking", False)
    st.session_state.setdefault("assistant_current_input", "")
    st.session_state.setdefault("assistant_pending_question", None)


def queue_assistant_question(question: str) -> None:
    """Queue a user question so the UI can show a typing bubble first."""

    st.session_state["assistant_messages"].append({"role": "user", "content": question})
    st.session_state["assistant_pending_question"] = question
    st.session_state["assistant_thinking"] = True


def complete_pending_assistant_reply() -> None:
    """Generate the queued assistant reply through Qwen or the safe fallback."""

    question = st.session_state.get("assistant_pending_question")
    if not question:
        return

    risk_score, risk_cat, factors, follow_up = current_assistant_context()

    # Qwen/OpenRouter is connected inside generate_assistant_reply(). If no key
    # is configured, that helper safely returns a local fallback response.
    reply = generate_assistant_reply(question, risk_score, risk_cat, factors, follow_up)

    st.session_state["assistant_thinking"] = False
    st.session_state["assistant_pending_question"] = None
    st.session_state["assistant_messages"].append({"role": "assistant", "content": reply})


def assistant_status_markup() -> str:
    """Return assistant connection status markup."""

    if qwen_is_configured():
        provider = qwen_provider_name() or "Qwen"
        return (
            '<div class="assistant-status">'
            f'<strong>Live Qwen connected through {html.escape(provider)}.</strong><br>'
            'Responses are generated through the configured OpenAI-compatible API with non-diagnostic safety instructions.'
            '</div>'
        )
    return (
        '<div class="assistant-status">'
        '<strong>Live Qwen is not connected.</strong><br>'
        'Fallback replies are active. Add OPENROUTER_API_KEY for OpenRouter or DASHSCOPE_API_KEY for DashScope.'
        '</div>'
    )


def assistant_messages_markup() -> str:
    """Render chat history with compact custom bubbles."""

    ensure_assistant_messages()
    rows = []
    for message in st.session_state["assistant_messages"]:
        role = message["role"]
        avatar = "You" if role == "user" else "AI"
        rows.append(
            f'<div class="assistant-message {html.escape(role)}">'
            f'<div class="assistant-avatar">{html.escape(avatar)}</div>'
            f'<div class="assistant-bubble">{html.escape(message["content"])}</div>'
            '</div>'
        )
    if st.session_state.get("assistant_thinking"):
        rows.append(
            '<div class="assistant-message assistant">'
            '<div class="assistant-avatar">AI</div>'
            '<div class="assistant-bubble"><div class="assistant-typing">'
            '<span></span><span></span><span></span>'
            '</div></div>'
            '</div>'
        )
    return '<div class="assistant-messages">' + "".join(rows) + "</div>"


def render_assistant_chat(form_key: str) -> None:
    """Render a bounded assistant chat panel."""

    ensure_assistant_messages()
    st.markdown(assistant_status_markup(), unsafe_allow_html=True)
    st.markdown(
        '<div class="pcos-small">Explains screening outputs. It cannot diagnose or prescribe treatment.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="assistant-prompt-title">Suggested prompts</div>', unsafe_allow_html=True)
    quick_cols = st.columns(2)
    for index, question in enumerate(QUICK_QUESTIONS):
        if quick_cols[index % 2].button(question, key=f"{form_key}_quick_{index}", width="stretch"):
            queue_assistant_question(question)
            st.rerun()

    st.markdown(assistant_messages_markup(), unsafe_allow_html=True)

    with st.form(form_key, clear_on_submit=True):
        user_question = st.text_input("Ask a question", placeholder="Example: What should I ask my clinician?")
        submitted = st.form_submit_button("Send", width="stretch")

    if submitted and user_question.strip():
        queue_assistant_question(user_question.strip())
        st.rerun()

    if st.session_state.get("assistant_pending_question"):
        complete_pending_assistant_reply()
        st.rerun()


def render_assistant_page() -> None:
    """Render assistant page with a right-side chat panel."""

    page_title("♡", "AI Assistant", "Ask about screening results, top factors, and clinician follow-up questions.")

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown(
            """
            <div class="pcos-panel">
                <div class="pcos-panel-title">How to use the assistant</div>
                <div class="pcos-small">
                    The assistant explains PCOScope outputs in plain language. It can help prepare
                    questions for a clinician and clarify why certain factors may matter.
                    It does not diagnose PCOS or recommend medication.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        result = st.session_state.get("screening_result")
        if result:
            st.markdown("#### Current screening context")
            st.metric("PCOS risk score", f"{result['probability']:.1%}")
            st.metric("Risk category", result["category"])
            st.write("Top factors:", ", ".join(result["factor_names"][:3]))
        else:
            st.info("Calculate a score first if you want the assistant to use patient-specific context.")

    with right:
        st.markdown(
            '<div class="pcos-panel">'
            '<div class="assistant-header" style="position:static;margin:-1.35rem -1.35rem 1rem -1.35rem;border-radius:18px 18px 0 0;">'
            '<div><div class="assistant-title">AI Health Assistant</div>'
            '<div class="assistant-online">Online</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        render_assistant_chat("assistant_page_form")
        st.markdown('</div>', unsafe_allow_html=True)


def render_assistant_launcher() -> None:
    """Render a floating assistant launcher and right-side drawer."""

    with st.container(key="assistant_fab"):
        if st.button("Assistant", width="stretch", key="assistant_avatar"):
            st.session_state["assistant_open"] = not st.session_state.get("assistant_open", False)

    if st.session_state.get("assistant_open"):
        with st.container(key="assistant_drawer"):
            with st.container(key="assistant_close_wrap"):
                if st.button("×", key="assistant_close"):
                    st.session_state["assistant_open"] = False
                    st.rerun()
            st.markdown(
                """
                <div class="assistant-header">
                    <div>
                        <div class="assistant-title">AI Health Assistant</div>
                        <div class="assistant-online">Online</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_assistant_chat("assistant_drawer_form")


def render_clinical_tab() -> None:
    """Render clinical rationale in concise, judge-friendly form."""

    page_title(
        "♧",
        "Clinical Rationale",
        "PCOScope organises screening across clinical domains commonly used in PCOS assessment.",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="pcos-panel">
                <div class="pcos-icon">○</div>
                <div class="pcos-panel-title">Ovulatory dysfunction</div>
                <div class="pcos-small">
                    Captured through cycle length, irregularity, and menstrual history. Chronic
                    anovulation is a defining feature of many PCOS phenotypes.
                </div>
                <div class="pcos-pill-row">
                    <span class="pcos-pill">Cycle length</span>
                    <span class="pcos-pill">Irregular cycles</span>
                    <span class="pcos-pill">Pregnancy history</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="pcos-panel">
                <div class="pcos-icon">⌁</div>
                <div class="pcos-panel-title">Hyperandrogenism-related symptoms</div>
                <div class="pcos-small">
                    Clinical signs and biochemical markers can signal androgen excess, but they
                    require clinician interpretation and appropriate laboratory context.
                </div>
                <div class="pcos-pill-row">
                    <span class="pcos-pill">Hair growth</span>
                    <span class="pcos-pill">Pimples</span>
                    <span class="pcos-pill">LH / FSH</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(
            """
            <div class="pcos-panel">
                <div class="pcos-icon">□</div>
                <div class="pcos-panel-title">Ovarian morphology</div>
                <div class="pcos-small">
                    Follicle count, follicle size, and endometrial measurements provide ultrasound-related
                    screening context. They do not replace imaging review.
                </div>
                <div class="pcos-pill-row">
                    <span class="pcos-pill">Follicle count</span>
                    <span class="pcos-pill">Follicle size</span>
                    <span class="pcos-pill">Endometrium</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="pcos-panel">
                <div class="pcos-icon">◌</div>
                <div class="pcos-panel-title">Metabolic risk</div>
                <div class="pcos-small">
                    Metabolic markers support broader cardiometabolic screening, especially where insulin
                    resistance or weight change may influence risk.
                </div>
                <div class="pcos-pill-row">
                    <span class="pcos-pill">BMI</span>
                    <span class="pcos-pill">Glucose</span>
                    <span class="pcos-pill">Blood pressure</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### How The Score Is Composed")
    st.markdown('<p class="pcos-subtitle">Domain weights are a presentation aid for judges and clinicians. The trained model still produces the final risk score.</p>', unsafe_allow_html=True)
    render_bar_rows(
        [
            ("Ovarian morphology", 32),
            ("Ovulatory dysfunction", 28),
            ("Hyperandrogenism-related symptoms", 24),
            ("Metabolic risk", 16),
        ],
        scale=100,
    )

    st.markdown(
        """
        <div class="pcos-note">
            Limitations: the model was trained on a specific clinical dataset, requires external validation,
            and must be interpreted alongside differential diagnosis for overlapping symptoms.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_performance_tab(model_bundle: dict) -> None:
    """Render model evaluation and technical transparency."""

    page_title(
        "⌁",
        "Model Performance",
        "Held-out evaluation metrics. Recall and F1 are prioritised for screening safety.",
    )

    best = model_bundle["best_model_name"]
    metrics = model_bundle["metrics"][best]

    metric_items = [
        ("Accuracy", metrics["accuracy"], "Overall correct predictions."),
        ("Precision", metrics["precision"], "Of flagged cases, how many were truly positive."),
        ("Recall / Sensitivity", metrics["recall_sensitivity"], "Of true cases, how many were successfully flagged."),
        ("F1-score", metrics["f1"], "Harmonic mean of precision and recall."),
        ("ROC-AUC", metrics["roc_auc"], "Ranking quality across thresholds."),
    ]
    metric_markup = []
    for title, value, body in metric_items:
        metric_markup.append(
            '<div class="pcos-panel">'
            '<div class="pcos-icon">⌁</div>'
            f'<div class="pcos-step-number" style="color:#d986ac;font-size:1.45rem;">{value:.1%}</div>'
            f'<div class="pcos-panel-title">{html.escape(title)}</div>'
            f'<div class="pcos-small">{html.escape(body)}</div>'
            '</div>'
        )
    st.markdown(f'<div class="pcos-card-grid" style="grid-template-columns:repeat(5,minmax(0,1fr));">{"".join(metric_markup)}</div>', unsafe_allow_html=True)

    st.markdown("### Candidate Model Comparison")
    st.dataframe(model_bundle["metrics_table"], width="stretch", hide_index=True)

    calibration = model_bundle.get("calibration_comparison", {})
    if calibration:
        st.markdown("### Probability Calibration")
        calibration_rows = []
        for model_name, values in calibration.items():
            calibration_rows.append(
                {
                    "model": model_name,
                    "uncalibrated_roc_auc": values.get("uncalibrated_roc_auc"),
                    "calibrated_roc_auc": values.get("calibrated_roc_auc"),
                    "roc_auc_delta": values.get("roc_auc_delta"),
                    "status": values.get("error", "calibrated"),
                }
            )
        st.dataframe(pd.DataFrame(calibration_rows), width="stretch", hide_index=True)

    st.markdown("### Technical Status")
    st.markdown(
        '<div class="pcos-note">'
        f'Best selected model: <strong>{html.escape(best)}</strong><br>'
        f'{html.escape(model_bundle.get("boosting_status", "Model status unavailable."))}'
        '</div>',
        unsafe_allow_html=True,
    )

    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    st.markdown("### Confusion Matrix")
    matrix_body = (
        '<div class="pcos-step-grid" style="grid-template-columns:repeat(2,minmax(0,1fr));">'
        '<div class="pcos-panel">'
        '<div class="pcos-panel-title">Counts on the held-out validation cohort</div>'
        '<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0.75rem;margin-top:1rem;">'
        '<div class="pcos-factor-card" style="background:#fff5fa;text-align:center;">'
        f'<div class="pcos-step-number" style="color:#d986ac;">{tn}</div>'
        '<div class="pcos-small">True negative</div>'
        '</div>'
        '<div class="pcos-factor-card" style="background:#fff0f6;text-align:center;">'
        f'<div class="pcos-step-number" style="color:#eca0c2;">{fp}</div>'
        '<div class="pcos-small">False positive</div>'
        '</div>'
        '<div class="pcos-factor-card" style="background:#fde8ef;text-align:center;">'
        f'<div class="pcos-step-number" style="color:#eca0c2;">{fn}</div>'
        '<div class="pcos-small">False negative</div>'
        '</div>'
        '<div class="pcos-factor-card" style="background:#f4e1ec;text-align:center;">'
        f'<div class="pcos-step-number" style="color:#d986ac;">{tp}</div>'
        '<div class="pcos-small">True positive</div>'
        '</div>'
        '</div>'
        '</div>'
        '<div class="pcos-panel">'
        '<div class="pcos-panel-title">Why recall and F1 matter most</div>'
        '<div class="pcos-small">'
        'Screening prioritises not missing true cases. A false negative can delay review, '
        'while a false positive mainly leads to clinician follow-up that may rule PCOS out. '
        'PCOScope therefore favours recall while keeping F1 clinically reasonable.'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(
        matrix_body,
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_theme()
    page = render_sidebar()
    render_header()

    model_bundle = load_or_train_model()

    if page == "Overview":
        render_overview_tab()
    elif page == "Risk Screening":
        render_screening_tab(model_bundle)
    elif page == "Results":
        render_results_tab()
    elif page == "Explainability":
        render_explainability_tab(model_bundle)
    elif page == "AI Assistant":
        render_assistant_page()
    elif page == "Clinical Rationale":
        render_clinical_tab()
    elif page == "Model Performance":
        render_performance_tab(model_bundle)

    render_assistant_launcher()


if __name__ == "__main__":
    main()
