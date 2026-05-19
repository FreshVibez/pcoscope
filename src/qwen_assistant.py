"""Optional Qwen/DashScope explanation layer for PCOScope."""

from __future__ import annotations

import os

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"


def get_dashscope_api_key() -> str:
    """Read the DashScope key from environment variables or Streamlit secrets."""

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if api_key:
        return api_key

    try:
        import streamlit as st

        return str(st.secrets.get("DASHSCOPE_API_KEY", "")).strip()
    except Exception:
        return ""


def qwen_is_configured() -> bool:
    """Return whether live Qwen calls can be attempted."""

    return bool(get_dashscope_api_key())


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
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate a short patient-friendly explanation through Qwen if configured."""

    api_key = get_dashscope_api_key()
    if not api_key:
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
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=BASE_URL)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You explain clinical screening outputs safely and clearly."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=220,
        )
        return response.choices[0].message.content.strip()
    except Exception:
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
    model: str = DEFAULT_MODEL,
) -> str:
    """Answer bounded PCOScope assistant questions through Qwen when configured."""

    api_key = get_dashscope_api_key()
    if not api_key:
        return fallback_assistant_reply(question, risk_score, risk_category, top_factors, follow_up)

    context = (
        "PCOScope is a PCOS risk-screening and clinical decision-support prototype, not a diagnostic tool. "
        "Use calm, concise, non-diagnostic language. Do not prescribe medication. "
        "Always recommend clinician review for clinical decisions. "
        "Relevant clinical domains include ovulatory dysfunction, hyperandrogenism-related symptoms, ovarian morphology, and metabolic risk. "
        f"Risk score: {risk_score}. Risk category: {risk_category}. Top factors: {top_factors}. Suggested follow-up: {follow_up}."
    )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=BASE_URL)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            max_tokens=260,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return fallback_assistant_reply(question, risk_score, risk_category, top_factors, follow_up)
