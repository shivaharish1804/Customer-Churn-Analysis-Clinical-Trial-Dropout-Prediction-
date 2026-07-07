# Project Explainer — Clinical Trial Dropout Prediction

*Written for recruiters, hiring managers, and anyone reviewing this project without running the code.*

---

## The Problem (Why This Matters)

Clinical trials are how new drugs get approved. But on average, 30–40% of enrolled patients drop out before the trial ends. This is a massive problem:

- A single Phase 3 trial can cost $50M–$100M. Dropout forces extensions, extra recruitment, and sometimes complete trial failure.
- Patient dropout is the #1 reason clinical trials run over budget and schedule.
- It's also inequitable — patients who live far from trial sites or have high disease burden drop out at much higher rates, skewing who gets represented in drug research.

Most trial teams don't predict dropout — they react to it after it happens. This project builds a system that flags high-risk patients *before* they drop out, so the trial team can intervene.

---

## What I Built

A machine learning pipeline that:

1. **Predicts dropout probability** for each enrolled patient using 17 features (demographics, trial design, patient burden factors)
2. **Explains each prediction** using SHAP values — not just "this patient is 74% likely to drop out" but *why*: "their burden score is high, they live alone, and they're 90km from the site"
3. **Runs what-if scenarios** — if we arrange transport to reduce travel distance by 30km, how much does their risk drop?
4. **Classifies patients into risk tiers** (Low / Medium / High) for prioritized outreach by coordinators

---

## Technical Approach

**Models compared:** Logistic Regression, Random Forest, Gradient Boosting, XGBoost

**Why XGBoost won:** Handles non-linear interactions between features well (e.g. the combined effect of high burden + early adverse event is worse than either alone). Also has the best native SHAP support.

**Evaluation:** ROC-AUC (primary), 5-fold cross-validation, precision/recall at the 50% threshold

**Explainability:** SHAP TreeExplainer — produces both global importance rankings and per-patient waterfall explanations

---

## The Novel Angle

Most beginner ML projects train a classifier and report accuracy. This project goes further in two ways:

**1. Actionable explanations:** A SHAP waterfall plot tells the trial coordinator not just the risk score but the specific factors driving it. That's the difference between a model that sits in a notebook and one that gets used in practice.

**2. What-if analysis:** For any patient, the dashboard shows how modifying controllable factors (transport distance, visit frequency, support resources) changes their predicted dropout risk. This turns the model from a passive predictor into a decision-support tool.

---

## Key Findings

*(Fill in after running the notebook)*

- Best model: XGBoost (AUC = —)
- Top 3 dropout risk factors by SHAP: burden score, early adverse event, distance to site
- High-risk tier actual dropout rate: —% (vs —% for low-risk tier)
- Reducing distance from 80km → 20km reduces dropout probability by approximately —pp for a median patient

---

## Why This Project Is Relevant to Biotech/Pharma Roles

- Shows understanding of the clinical trial pipeline and real industry pain points
- Demonstrates ML beyond accuracy — explainability and decision support
- SHAP is increasingly required in regulated environments (FDA increasingly expects explainable AI in clinical contexts)
- The what-if analysis mimics how a real CRO (Contract Research Organization) analyst would use such a model
