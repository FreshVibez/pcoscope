# PCOScope: Explainable PCOS Risk Screening Assistant

PCOScope is a hackathon MVP for explainable PCOS risk screening from tabular clinical data. It predicts a PCOS risk score, surfaces likely contributing clinical factors, and optionally uses Qwen through DashScope to generate calm patient-friendly explanations and bounded assistant responses.

This is **not a diagnosis tool**. It is a clinical decision-support and risk-screening prototype intended to support follow-up conversations and prioritisation, not replace clinician judgement.

## Problem Statement

Polycystic Ovary Syndrome (PCOS) is common, heterogeneous, and often under-recognised. Symptoms can overlap with other reproductive and metabolic conditions, which can contribute to delayed or missed assessment. The goal is to build a feasible system that helps clinicians interpret complex clinical features, reduce missed screening opportunities, and explain risk factors clearly.

## Clinical & Scientific Validity

This project is designed around accepted clinical reasoning for PCOS rather than treating the dataset as a black-box classification problem. Current international guidance describes adult PCOS assessment around three core domains: clinical or biochemical hyperandrogenism, ovulatory dysfunction, and polycystic ovarian morphology, with AMH also recognised as an adult alternative to ultrasound in the 2023 international evidence-based guideline. PCOScope does not claim to diagnose PCOS from these features alone; it estimates risk and highlights factors that may justify follow-up assessment.

The model inputs map to clinically meaningful PCOS domains:

- Ovulatory dysfunction: `Cycle(R/I)` and `Cycle length(days)` reflect menstrual irregularity and possible anovulation.
- Hyperandrogenism-related signs: `hair growth(Y/N)`, `Pimples(Y/N)`, and `Hair loss(Y/N)` reflect symptoms commonly associated with androgen excess, such as hirsutism, acne, and alopecia.
- Reproductive hormone patterns: `LH(mIU/mL)`, `FSH(mIU/mL)`, `FSH/LH`, `AMH(ng/mL)`, and `PRL(ng/mL)` provide endocrine context. These values require clinical interpretation and are not used as standalone proof of PCOS.
- Ovarian morphology: `Follicle No. (L)`, `Follicle No. (R)`, average follicle size, and endometrium measures provide ultrasound-related context.
- Metabolic and cardiometabolic risk: `BMI`, `Weight (Kg)`, `Waist(inch)`, `Hip(inch)`, `Waist:Hip Ratio`, `RBS(mg/dl)`, blood pressure, and `Weight gain(Y/N)` reflect metabolic features commonly relevant in PCOS risk assessment.
- Lifestyle and modifiable factors: `Fast food (Y/N)` and `Reg.Exercise(Y/N)` support practical counselling around metabolic risk, while avoiding blame-based interpretation.

The biological rationale is based on the team research notes in `../biohackathon - team 31.pdf`, which summarise PCOS mechanisms including hormonal imbalance, insulin resistance, chronic low-grade inflammation, hyperandrogenism, possible gut dysbiosis, lifestyle factors, and longer-term comorbidity risk. These mechanisms justify why the prototype combines reproductive, endocrine, symptom, metabolic, and lifestyle features rather than relying on one marker.

Clinical safeguards and limitations:

- PCOScope uses language such as "risk score", "possible contributing factors", and "recommended follow-up"; it does not say that a patient has PCOS.
- A high score should prompt clinical review, not automatic diagnosis. Differential diagnoses and overlapping conditions may include thyroid dysfunction, hyperprolactinaemia, adrenal disorders, pregnancy-related hormonal changes, endometriosis, and other causes of irregular bleeding or hyperandrogenic symptoms.
- The main dataset comes from hospitals in Kerala, India, so external validation is needed before use in other populations or healthcare settings.
- Model explanations are intended to support clinician review. They should be checked against patient history, examination, lab reliability, ultrasound context, and local diagnostic guidelines.
- The recommended next steps are intentionally conservative: follow-up assessment, targeted history, endocrine/metabolic labs, ultrasound review, or referral where clinically appropriate.

## Judging Criteria Mapping

- Clinical validity: maps model inputs to accepted PCOS assessment domains, including ovulatory dysfunction, hyperandrogenism-related symptoms, reproductive hormones, ovarian morphology, and metabolic risk.
- Diagnostic accuracy: compares multiple tabular models and prioritises recall, F1-score, and ROC-AUC instead of raw accuracy alone.
- Innovation: combines model comparison, ensemble learning, explainability, and optional Qwen-generated explanations.
- Public health impact: frames output as early risk screening and recommended follow-up, which can support timely assessment.
- Feasibility: uses a lightweight Streamlit dashboard and standard CSV/model files that can run locally.
- Technical quality: modular code separates preprocessing, training, evaluation, explanations, and Qwen integration.
- Explainability: shows feature importance and top prediction-level contributing factors.
- Presentation clarity: dashboard separates screening, results, explainability, assistant, clinical rationale, and model performance into focused tabs.

## Dashboard Design

The Streamlit interface is organized as a practical healthcare dashboard rather than a single crowded page:

- Screening: grouped clinical inputs for profile, cycle pattern, hormones, metabolic markers, symptoms, lifestyle, and ultrasound features.
- Results: PCOS risk score, risk category, recommended follow-up, and a patient-friendly explanation.
- Explainability: top patient-level contributing factors and global feature importance.
- Assistant: bounded Qwen-powered support, with a local fallback, for explaining screening results and clinician follow-up questions.
- Clinical Rationale: feature-to-biology mapping, safety boundaries, and limitations.
- Model Performance: candidate model comparison, selected model metrics, confusion matrix, and technical status.

## Model Choices

- Logistic Regression is included as an interpretable baseline.
- Random Forest captures non-linear relationships between symptoms, clinical values, and risk.
- XGBoost is included for strong tabular-data performance when the local environment can load it.
- If XGBoost cannot load on macOS because OpenMP is missing, the code falls back to sklearn histogram gradient boosting. This keeps the boosted-tree comparison reproducible for the demo while making the fallback explicit.
- A stacking ensemble combines Random Forest and the available boosted-tree model with Logistic Regression as a meta-model.

The saved model is selected by F1-score, then recall/sensitivity, then ROC-AUC.

## Why Recall, F1-Score, and ROC-AUC Matter

For screening, a false negative can be risky because a patient who may need follow-up could be missed. Recall/sensitivity measures how many positive cases are caught. F1-score balances recall and precision. ROC-AUC measures how well the model separates higher-risk from lower-risk patients across thresholds.

## Why Explainability Matters

Healthcare users need to understand why a model produced a risk score. PCOScope shows global feature importance and patient-level top contributing factors so the output can be reviewed, questioned, and connected to clinical reasoning. This matters because PCOS is heterogeneous: two patients can have different combinations of menstrual, androgenic, ovarian, and metabolic features. Explainability helps avoid a misleading "one-size-fits-all" interpretation.

The current saved model uses permutation importance on the holdout test set because the selected boosted-tree fallback does not expose native feature importances in the same way as Random Forest or XGBoost. SHAP is supported as an optional path in the code when compatible with the selected model, but the dashboard labels the active method honestly instead of implying SHAP is always used.

## References

- Team research notes: `../biohackathon - team 31.pdf`
- International PCOS Network. "Recommendations From the 2023 International Evidence-based Guideline for the Assessment and Management of Polycystic Ovary Syndrome." Journal of Clinical Endocrinology & Metabolism, 2023. https://academic.oup.com/jcem/article/108/10/2447/7242360
- Review referenced in team notes: https://pmc.ncbi.nlm.nih.gov/articles/PMC9964744/

## Qwen Safety

Qwen is optional and is used only after the ML model has computed the screening result. It can explain the result or answer bounded assistant questions. It receives:

- risk score
- risk category
- top contributing factors
- recommended follow-up

The prompts explicitly state that PCOScope is a risk-screening and clinical decision-support prototype, not a medical diagnosis tool. The assistant is instructed not to prescribe medication and to recommend clinician review for clinical decisions. API keys are never hardcoded. Set `DASHSCOPE_API_KEY` in your environment or in `.streamlit/secrets.toml` to enable Qwen; otherwise, the app uses local fallback explanations.

## How to Run

```bash
cd pcos-hackathon
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

Optional Qwen setup:

```bash
export DASHSCOPE_API_KEY="your_key_here"
streamlit run app.py
```

Or create `.streamlit/secrets.toml`:

```toml
DASHSCOPE_API_KEY = "your_key_here"
```

The app shows a connection message inside the AI Assistant panel so you can confirm whether live Qwen or the fallback responder is being used.

## Project Structure

```text
pcos-hackathon/
├── data/
│   └── pcos_dataset.csv
├── models/
│   └── saved_model.pkl
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   └── qwen_assistant.py
├── app.py
├── requirements.txt
└── README.md
```
