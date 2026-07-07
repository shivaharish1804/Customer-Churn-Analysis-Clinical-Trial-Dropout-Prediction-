"""
SHAP explainability utilities for Clinical Trial Dropout model
"""
import numpy as np
import matplotlib.pyplot as plt
import shap

def get_shap_explainer(model, X_train):
    """Create a SHAP TreeExplainer for tree-based models."""
    explainer = shap.TreeExplainer(model)
    return explainer

def plot_shap_summary(explainer, X_test, feature_names, max_display=15, save_path=None):
    """Global feature importance via SHAP summary plot."""
    shap_values = explainer.shap_values(X_test)
    # For binary classifiers shap_values may be a list
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_test, feature_names=feature_names,
                      max_display=max_display, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return sv

def plot_shap_waterfall(explainer, X_sample, feature_names, patient_idx=0, save_path=None):
    """Per-patient SHAP waterfall — explains one prediction."""
    shap_values = explainer(X_sample)
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[patient_idx], max_display=12, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def whatif_analysis(model, patient_row, feature_name, values, feature_names):
    """
    What-if analysis: vary one feature and plot dropout probability.
    Shows how changing e.g. distance_to_site affects risk.
    """
    import pandas as pd
    results = []
    for v in values:
        row = patient_row.copy()
        row[feature_name] = v
        prob = model.predict_proba(pd.DataFrame([row], columns=feature_names))[:, 1][0]
        results.append({'value': v, 'dropout_prob': prob})

    df = pd.DataFrame(results)
    plt.figure(figsize=(8, 4))
    plt.plot(df['value'], df['dropout_prob'] * 100, marker='o', color='#e74c3c', linewidth=2)
    plt.xlabel(feature_name.replace('_', ' ').title())
    plt.ylabel('Dropout Risk (%)')
    plt.title(f'What-If: How {feature_name.replace("_"," ").title()} Affects Dropout Risk')
    plt.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return df
