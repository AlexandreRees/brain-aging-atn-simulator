import numpy as np
import pandas as pd

np.random.seed(42)

n_subjects = 200
n_timepoints = 6

rows = []

for subject in range(n_subjects):

    age0 = np.random.normal(70, 5)

    # initial state
    state = np.random.choice(["CN", "MCI", "AD"], p=[0.5, 0.3, 0.2])

    progression = 0

    for t in range(n_timepoints):

        age = age0 + t

        # disease progression dynamics
        if state == "CN":
            progression += np.random.normal(0.05, 0.02)
        elif state == "MCI":
            progression += np.random.normal(0.25, 0.08)
        else:  # AD
            progression += np.random.normal(0.45, 0.1)

        # brain biomarkers evolve over time
        hippocampus = 4200 - 30*age - 600*progression + np.random.normal(0, 120)
        ventricles = 3000 + 35*age + 700*progression + np.random.normal(0, 150)
        cortical_thickness = 2.8 - 0.01*age - 0.25*progression + np.random.normal(0, 0.03)

        mmse = 30 - 0.18*(age - 60) - 3.5*progression + np.random.normal(0, 1)
        mmse = np.clip(mmse, 0, 30)

        rows.append({
            "subject_id": subject,
            "timepoint": t,
            "age": age,
            "state": state,
            "progression": progression,
            "hippocampus": hippocampus,
            "ventricles": ventricles,
            "cortical_thickness": cortical_thickness,
            "mmse": mmse
        })

df = pd.DataFrame(rows)

df.to_csv("data/longitudinal_brain.csv", index=False)

print("Saved:", df.shape)
print(df.head())
