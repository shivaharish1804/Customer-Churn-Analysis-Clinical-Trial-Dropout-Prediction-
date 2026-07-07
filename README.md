# Clinical Trial Dropout Prediction

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange) ![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)

> Predict which patients are likely to drop out of clinical trials — and explain *why* — using machine learning and SHAP explainability.

## The Problem

Clinical trial dropout rates average 30–40%, costing the pharmaceutical industry billions annually and delaying life-saving treatments. If trial coordinators can identify high-risk patients *early*, they can intervene — reducing travel burden, adjusting visit schedules, or providing additional support.

## What This Project Does

- **Trains 4 models** (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) and compares them on ROC-AUC
- **Explains predictions** using SHAP values — both globally (which features matter most) and per-patient (why *this* patient is high risk)
- **What-if analysis** — shows how changing modifiable factors (distance to site, burden score) shifts dropout risk
- **Risk tier system** — classifies patients as Low / Medium / High risk for prioritized outreach
- **Interactive dashboard** — Streamlit app for trial coordinators to assess any patient in real time

## Novel Angle

Most ML projects stop at accuracy. This project delivers *actionable explainability*:
- A trial coordinator sees not just "this patient is 78% likely to drop out" but "the top reasons are: high burden score (8/10), lives alone, and 85km from the site"
- The what-if tool shows: "if we arrange transport to reduce distance from 85km to 20km, dropout risk drops from 78% to 52%"

This is the kind of output a real biotech/CRO team would use.

## Dataset

Synthetic dataset (5,000 patients) generated from real-world distributions in ClinicalTrials.gov research literature. Features include demographics, trial characteristics, patient burden factors, and early adverse events.

## Results

| Model | Test AUC | CV AUC (5-fold) |
|-------|----------|-----------------|
| Logistic Regression | — | — |
| Random Forest | — | — |
| Gradient Boosting | — | — |
| **XGBoost** | **—** | **—** |

*Fill in after running the notebook.*

**Top dropout risk factors (SHAP):**
1. Patient reported burden score
2. Early adverse event
3. Distance to site (km)
4. Number of visits required
5. Trial duration (months)

## Project Structure

```
clinical_trial_dropout/
├── README.md
├── EXPLAINER.md          ← Story for recruiters
├── requirements.txt
├── notebooks/
│   └── clinical_trial_dropout.ipynb   ← Main analysis
├── src/
│   ├── generate_data.py  ← Dataset generator
│   └── shap_utils.py     ← SHAP helper functions
├── app/
│   └── app.py            ← Streamlit dashboard
├── models/               ← Saved model files (after running notebook)
└── data/                 ← Generated dataset (after running notebook)
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the notebook
jupyter notebook notebooks/clinical_trial_dropout.ipynb

# 3. Launch the dashboard (after notebook completes)
streamlit run app/app.py
```

## Or Run in Google Colab

```python
!pip install xgboost shap streamlit joblib -q

# Upload src/generate_data.py then:
import sys; sys.path.insert(0, '/content')
# Then run notebook cells
```
