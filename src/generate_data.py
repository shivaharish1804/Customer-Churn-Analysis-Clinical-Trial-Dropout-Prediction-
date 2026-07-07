"""
Synthetic Clinical Trial Dataset Generator
Based on real-world distributions from ClinicalTrials.gov research literature.
References: Fogel (2018), Carlisle et al. (2015), Schandelmaier et al. (2017)
"""
import numpy as np
import pandas as pd

def generate_trial_data(n_samples=5000, random_state=42):
    np.random.seed(random_state)
    n = n_samples

    # Demographics
    age = np.random.normal(52, 14, n).clip(18, 85).astype(int)
    sex = np.random.choice(['Male', 'Female'], n, p=[0.48, 0.52])
    bmi = np.random.normal(27.5, 5.5, n).clip(16, 50).round(1)

    # Trial characteristics
    trial_phase = np.random.choice(['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4'],
                                    n, p=[0.15, 0.30, 0.40, 0.15])
    trial_duration_months = np.where(
        trial_phase == 'Phase 1', np.random.randint(3, 12, n),
        np.where(trial_phase == 'Phase 2', np.random.randint(6, 24, n),
        np.where(trial_phase == 'Phase 3', np.random.randint(12, 48, n),
                 np.random.randint(6, 24, n)))
    )
    therapeutic_area = np.random.choice(
        ['Oncology', 'Cardiology', 'Neurology', 'Infectious Disease', 'Metabolic'],
        n, p=[0.30, 0.20, 0.20, 0.15, 0.15]
    )
    num_visits_required = np.random.randint(3, 25, n)
    has_placebo_arm = np.random.choice([0, 1], n, p=[0.35, 0.65])
    num_procedures_per_visit = np.random.randint(1, 8, n)

    # Patient burden factors
    distance_to_site_km = np.random.exponential(35, n).clip(1, 300).round(1)
    num_comorbidities = np.random.poisson(1.8, n).clip(0, 6)
    prior_trial_participation = np.random.choice([0, 1], n, p=[0.65, 0.35])
    employment_status = np.random.choice(
        ['Employed', 'Unemployed', 'Retired', 'Student'],
        n, p=[0.45, 0.15, 0.30, 0.10]
    )
    lives_alone = np.random.choice([0, 1], n, p=[0.70, 0.30])
    disease_severity = np.random.choice(['Mild', 'Moderate', 'Severe'],
                                         n, p=[0.30, 0.45, 0.25])

    # Side effect / tolerability
    early_adverse_event = np.random.choice([0, 1], n, p=[0.75, 0.25])
    patient_reported_burden_score = np.random.randint(1, 11, n)  # 1-10

    # Build dropout probability from known risk factors
    logit = (
        -1.5
        + 0.015 * (age - 52)
        + 0.03 * (distance_to_site_km - 35)
        + 0.08 * num_visits_required
        + 0.12 * num_procedures_per_visit
        + 0.25 * (trial_duration_months / 12)
        + 0.40 * early_adverse_event
        + 0.10 * patient_reported_burden_score
        + 0.30 * lives_alone
        - 0.35 * prior_trial_participation
        + 0.20 * (disease_severity == 'Severe').astype(int)
        + 0.15 * (therapeutic_area == 'Oncology').astype(int)
        - 0.20 * (therapeutic_area == 'Infectious Disease').astype(int)
        + 0.10 * has_placebo_arm
        + 0.15 * num_comorbidities
        - 0.15 * (employment_status == 'Retired').astype(int)
        + np.random.normal(0, 0.3, n)  # noise
    )
    dropout_prob = 1 / (1 + np.exp(-logit))
    dropped_out = (np.random.uniform(0, 1, n) < dropout_prob).astype(int)

    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'trial_phase': trial_phase,
        'trial_duration_months': trial_duration_months,
        'therapeutic_area': therapeutic_area,
        'num_visits_required': num_visits_required,
        'has_placebo_arm': has_placebo_arm,
        'num_procedures_per_visit': num_procedures_per_visit,
        'distance_to_site_km': distance_to_site_km,
        'num_comorbidities': num_comorbidities,
        'prior_trial_participation': prior_trial_participation,
        'employment_status': employment_status,
        'lives_alone': lives_alone,
        'disease_severity': disease_severity,
        'early_adverse_event': early_adverse_event,
        'patient_reported_burden_score': patient_reported_burden_score,
        'dropped_out': dropped_out
    })

    print(f"Generated {n} patient records")
    print(f"Dropout rate: {dropped_out.mean():.1%}")
    return df

if __name__ == "__main__":
    df = generate_trial_data()
    df.to_csv("data/clinical_trial_data.csv", index=False)
    print("Saved to data/clinical_trial_data.csv")
