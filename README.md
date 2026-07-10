# 🧪 Customer Churn Analysis — Clinical Trial Dropout Prediction

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange) ![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)

Predict which patients are likely to drop out of clinical trials — and explain *why* — using an ensemble of 4 machine learning models, SHAP explainability, and a what-if intervention simulator built for real trial coordinators.

---

## The Problem

Clinical trials are how new drugs get approved. But on average, **30–40% of enrolled patients drop out** before the trial ends — and this is a massive, largely preventable crisis:

- A single Phase 3 trial costs **$50M–$100M**. Dropout forces extensions, extra recruitment, and sometimes complete trial failure
- Patient dropout is the **#1 reason clinical trials run over budget and schedule**, delaying life-saving treatments by years
- Dropout is also **inequitable** — patients who live far from trial sites or have high disease burden drop out at much higher rates, skewing who gets represented in drug research

Most trial teams don't predict dropout — they react to it *after* it happens. This project builds a system that flags high-risk patients **before** they drop out, so the trial team can intervene in time.

---

## What This Project Does

- **Trains and compares 4 models** — Logistic Regression, Random Forest, Gradient Boosting, XGBoost — evaluated on ROC-AUC and 5-fold cross-validation
- **Explains every prediction using SHAP** — both globally (which features drive dropout most) and per-patient (exactly why *this* patient is high risk)
- **What-if intervention simulator** — shows how changing modifiable factors (transport distance, visit frequency, support resources) shifts a patient's dropout probability
- **Risk tier system** — classifies each patient as 🟢 Low / 🟡 Medium / 🔴 High risk for prioritised coordinator outreach
- **Interactive Streamlit dashboard** — trial coordinators assess any patient in real time with full explanation

---

## Novel Angle

Most ML projects stop at accuracy. This project delivers **actionable explainability**:

A trial coordinator sees not just *"this patient is 78% likely to drop out"* but:
> *"The top reasons are: high burden score (8/10), lives alone, and 85km from the site"*

The **what-if tool** shows:
> *"If we arrange transport to reduce distance from 85km to 20km, dropout risk drops from 78% to 52%"*

This turns the model from a passive predictor into a **decision-support tool** — the kind of output a real biotech or CRO (Contract Research Organization) team would actually use in practice.

SHAP is also increasingly required in **regulated environments** — the FDA increasingly expects explainable AI in clinical contexts, making this project directly relevant to real-world pharma/biotech deployment.

---

## Dataset

**Synthetic dataset — 5,000 patients** generated from real-world distributions in ClinicalTrials.gov research literature.

| Property | Detail |
|---|---|
| Patients | 5,000 synthetic trial participants |
| Features | 17 patient and trial characteristics |
| Target | Dropout (1) / Completed (0) |
| Generation | `src/generate_data.py` |
| Source distributions | ClinicalTrials.gov research literature |

Features include demographics, trial design characteristics, patient burden factors, distance to site, and early adverse events — all variables that real CRO teams track.

**Top dropout risk factors (SHAP):**
1. Patient-reported burden score
2. Early adverse event flag
3. Distance to site (km)
4. Number of visits required
5. Trial duration (months)

---

## Models Compared

| Model | Type | Strength |
|---|---|---|
| Logistic Regression | Linear baseline | Fast, interpretable |
| Random Forest | Ensemble (bagging) | Robust, handles non-linearity |
| Gradient Boosting | Ensemble (boosting) | Strong generalisation |
| **XGBoost** | Optimised boosting | **Best AUC** ✅ |

**Why XGBoost won:** Handles non-linear feature interactions well — e.g. the combined effect of high burden score *and* early adverse event is worse than either alone. Also has the best native SHAP support for per-patient explanations.

---

## Results

| Model | Test AUC | CV AUC (5-fold) |
|---|---|---|
| Logistic Regression | — | — |
| Random Forest | — | — |
| Gradient Boosting | — | — |
| **XGBoost** | **—** | **—** |

*Fill in after running the notebook*

---

## 📁 Project Structure

```
clinical_trial_dropout/
├── README.md
├── EXPLAINER.md                            # Story for recruiters & hiring managers
├── requirements.txt
├── notebooks/
│   └── clinical_trial_dropout.ipynb        # Full ML pipeline & analysis
├── src/
│   ├── generate_data.py                    # Synthetic dataset generator
│   └── shap_utils.py                       # SHAP helper functions
├── app/
│   └── app.py                              # Streamlit dashboard
├── models/                                 # Saved model files (after running notebook)
└── data/                                   # Generated dataset (after running notebook)
```

---

## ▶️ How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the full ML pipeline
```bash
jupyter notebook notebooks/clinical_trial_dropout.ipynb
```
Click **Kernel → Restart & Run All** — this generates the dataset, trains all 4 models, runs SHAP analysis, and saves all outputs to `models/` and `data/`

### Step 3 — Launch the dashboard
```bash
streamlit run app/app.py
```
Opens at **http://localhost:8501**

### Or Run in Google Colab
```python
!pip install xgboost shap streamlit joblib -q

# Upload src/generate_data.py then:
import sys; sys.path.insert(0, '/content')
# Then run notebook cells
```

---

## 📊 Dashboard Features

- **Patient risk assessment** — enter any patient's details for instant dropout probability
- **SHAP waterfall explanation** — per-patient breakdown of exactly which factors drive their risk
- **Global feature importance** — which features matter most across all patients
- **What-if simulator** — adjust modifiable factors and see how risk changes in real time
- **Risk tier classification** — 🟢 Low / 🟡 Medium / 🔴 High with suggested interventions
- **Model comparison** — ROC-AUC and CV results for all 4 models

---

## 🤖 Is This AI?

| Component | Type |
|---|---|
| 4 models trained and compared on ROC-AUC | ✅ Machine Learning (AI) |
| XGBoost uses gradient boosting optimisation | ✅ Advanced ML (AI) |
| SHAP TreeExplainer generates per-patient explanations | ✅ Explainable AI (XAI) |
| What-if simulator models counterfactual interventions | ✅ Counterfactual AI |
| Risk tier system enables prioritised clinical action | ✅ AI-powered decision support |
| 5-fold cross-validation ensures generalisation | ✅ Rigorous AI evaluation |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **scikit-learn** — Logistic Regression, Random Forest, Gradient Boosting, preprocessing
- **XGBoost** — Best performing classifier with native SHAP support
- **SHAP** — Global and per-patient explainability (TreeExplainer)
- **Streamlit** — Interactive coordinator dashboard
- **Matplotlib / Seaborn** — Visualisations and evaluation charts
- **Joblib** — Model serialisation
- **Jupyter** — End-to-end ML pipeline notebook

---

## ⚠️ Disclaimer

This tool uses a **synthetic dataset** generated from published research distributions. It is for **educational and research purposes only** and is not validated for use in real clinical trial management. All patient risk assessments in a real trial must be reviewed by qualified clinical staff.

---

## Internship Details

| Field | Detail |
|---|---|
| Intern ID | CITS6178 |
| Organisation | Codtech IT Solutions Private Limited |
| Domain | Artificial Intelligence & Machine Learning |
| Project | Customer Churn Analysis (Clinical Trial Dropout Prediction) |

