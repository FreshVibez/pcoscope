"""Optional Qwen explanation layer for PCOScope.

The app supports OpenRouter first because it is easy to deploy through
Streamlit secrets, and keeps DashScope as a backward-compatible provider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENROUTER_MODEL = "qwen/qwen3-next-80b-a3b-instruct:free"
DASHSCOPE_MODEL = "qwen-plus"


@dataclass(frozen=True)
class QwenProvider:
    """Runtime configuration for an OpenAI-compatible Qwen provider."""

    name: str
    api_key: str
    base_url: str
    model: str


def _read_secret(name: str) -> str:
    """Read a secret from the environment first, then Streamlit secrets."""

    value = os.getenv(name, "").strip()
    if value:
        return value

    try:
        import streamlit as st

        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def get_dashscope_api_key() -> str:
    """Read the DashScope key from environment variables or Streamlit secrets."""

    return _read_secret("DASHSCOPE_API_KEY")


def get_openrouter_api_key() -> str:
    """Read the OpenRouter key from environment variables or Streamlit secrets."""

    return _read_secret("OPENROUTER_API_KEY")


def get_qwen_provider() -> QwenProvider | None:
    """Return the configured Qwen provider.

    OpenRouter is preferred when present because the hackathon deployment is
    using an OpenRouter-hosted Qwen model. DashScope remains supported for the
    original official Qwen-compatible setup.
    """

    openrouter_key = get_openrouter_api_key()
    if openrouter_key:
        return QwenProvider(
            name="OpenRouter",
            api_key=openrouter_key,
            base_url=OPENROUTER_BASE_URL,
            model=OPENROUTER_MODEL,
        )

    dashscope_key = get_dashscope_api_key()
    if dashscope_key:
        return QwenProvider(
            name="DashScope",
            api_key=dashscope_key,
            base_url=DASHSCOPE_BASE_URL,
            model=DASHSCOPE_MODEL,
        )

    return None


def qwen_is_configured() -> bool:
    """Return whether live Qwen calls can be attempted."""

    return get_qwen_provider() is not None


def qwen_provider_name() -> str:
    """Return a display name for the configured provider."""

    provider = get_qwen_provider()
    return provider.name if provider else ""


def _qwen_client(provider: QwenProvider):
    """Create an OpenAI-compatible client for the selected provider."""

    from openai import OpenAI

    if provider.name == "OpenRouter":
        return OpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
            default_headers={
                "HTTP-Referer": "https://pcoscope.streamlit.app",
                "X-Title": "PCOScope",
            },
        )

    return OpenAI(api_key=provider.api_key, base_url=provider.base_url)


def fallback_explanation(
    risk_score: float,
    risk_category: str,
    top_factors: list[str],
    follow_up: str,
) -> str:
    """Local explanation used when Qwen is not configured or unavailable."""

    factors = ", ".join(top_factors) if top_factors else "the available clinical inputs"
    return (
        f"This screening result suggests a {risk_category.lower()} PCOS risk score "
        f"({risk_score:.1%}). The factors most associated with this score were {factors}. "
        "This is only a risk-screening result, not a medical diagnosis. "
        f"Suggested follow-up: {follow_up}"
    )


def generate_qwen_explanation(
    risk_score: float,
    risk_category: str,
    top_factors: list[str],
    follow_up: str,
    model: str | None = None,
) -> str:
    """Generate a short patient-friendly explanation through Qwen if configured."""

    provider = get_qwen_provider()
    if provider is None:
        return fallback_explanation(risk_score, risk_category, top_factors, follow_up)

    prompt = (
        "Explain this PCOS screening result in simple, calm, non-diagnostic language. "
        "State clearly that this is only a risk-screening tool and not a medical diagnosis. "
        f"Risk score: {risk_score}. "
        f"Risk category: {risk_category}. "
        f"Top factors: {top_factors}. "
        f"Suggested follow-up: {follow_up}."
    )

    try:
        client = _qwen_client(provider)
        response = client.chat.completions.create(
            model=model or provider.model,
            messages=[
                {"role": "system", "content": "You explain clinical screening outputs safely and clearly."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=220,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"Qwen explanation call failed through {provider.name}: {exc}")
        return fallback_explanation(risk_score, risk_category, top_factors, follow_up)


def fallback_assistant_reply(
    question: str,
    risk_score: float | None = None,
    risk_category: str | None = None,
    top_factors: list[str] | None = None,
    follow_up: str | None = None,
) -> str:
    """Rule-based assistant response used when Qwen is not configured."""

    lower_question = question.lower()
    factors = ", ".join(top_factors or []) or "the available screening factors"
    category_text = risk_category.lower() if risk_category else "available"
    score_text = f"{risk_score:.1%}" if risk_score is not None else "not calculated yet"

    if "diagnos" in lower_question:
        return (
            "PCOScope cannot diagnose PCOS. It estimates screening risk from structured clinical inputs. "
            "A clinician would still need to review symptoms, history, examination findings, labs, and ultrasound context."
        )

    if "factor" in lower_question or "why" in lower_question:
        return (
            f"The main factors currently highlighted are {factors}. These may relate to PCOS screening domains such as "
            "menstrual pattern, hyperandrogenism-related symptoms, ovarian morphology, and metabolic risk. "
            "They should be interpreted by a clinician rather than treated as proof of PCOS."
        )

    if "follow" in lower_question or "clinician" in lower_question or "ask" in lower_question:
        return (
            f"For a {category_text} screening result, a practical follow-up is: {follow_up or 'review symptoms and clinical context with a healthcare professional.'} "
            "Useful questions include whether further endocrine labs, metabolic screening, ultrasound review, or assessment for overlapping conditions is appropriate."
        )

    if "endometriosis" in lower_question or "overlap" in lower_question:
        return (
            "Some symptoms can overlap across reproductive health conditions, including pelvic pain, menstrual changes, and fertility concerns. "
            "PCOScope focuses on PCOS risk screening, so persistent pain or complex symptoms should be discussed with a clinician for broader differential assessment."
        )

    return (
        f"Your current PCOScope context is a {category_text} risk category with a risk score of {score_text}. "
        "This is a screening result, not a diagnosis. The safest next step is to use the highlighted factors to guide a clinician review."
    )


def generate_assistant_reply(
    question: str,
    risk_score: float | None = None,
    risk_category: str | None = None,
    top_factors: list[str] | None = None,
    follow_up: str | None = None,
    model: str | None = None,
) -> str:
    """Answer bounded PCOScope assistant questions through Qwen when configured."""

    provider = get_qwen_provider()
    if provider is None:
        return fallback_assistant_reply(question, risk_score, risk_category, top_factors, follow_up)

    context = (
        "PCOScope is a PCOS risk-screening and clinical decision-support prototype, not a diagnostic tool. "
        "Use calm, concise, non-diagnostic language. Do not prescribe medication. "
        "Always recommend clinician review for clinical decisions. "
        "Relevant clinical domains include ovulatory dysfunction, hyperandrogenism-related symptoms, ovarian morphology, and metabolic risk. "
        f"Risk score: {risk_score}. Risk category: {risk_category}. Top factors: {top_factors}. Suggested follow-up: {follow_up}."
    )

    try:
        client = _qwen_client(provider)
        response = client.chat.completions.create(
            model=model or provider.model,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            max_tokens=260,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"Qwen assistant call failed through {provider.name}: {exc}")
        return fallback_assistant_reply(question, risk_score, risk_category, top_factors, follow_up)
