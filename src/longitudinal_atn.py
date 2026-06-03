import numpy as np
import pandas as pd

np.random.seed(42)

n_subjects = 200
n_timepoints = 6

rows = []

for subject in range(n_subjects):

    age0 = np.random.normal(70, 5)

    state = np.random.choice(["CN", "MCI", "AD"], p=[0.5, 0.3, 0.2])

    amyloid = 0
    tau = 0
    neurodegeneration = 0

    for t in range(n_timepoints):

        age = age0 + t

        # -------------------------
        # 1. Amyloid (A)
        # slow accumulation
        # -------------------------
        amyloid += np.random.normal(0.2, 0.05)
        amyloid = max(amyloid, 0)

        # -------------------------
        # 2. Tau (T)
        # driven by amyloid + state
        # -------------------------
        if state == "CN":
            tau += 0.1 * amyloid + np.random.normal(0.05, 0.02)
        elif state == "MCI":
            tau += 0.3 * amyloid + np.random.normal(0.2, 0.05)
        else:  # AD
            tau += 0.6 * amyloid + np.random.normal(0.4, 0.1)

        tau = max(tau, 0)

        # -------------------------
        # 3. Neurodegeneration (N)
        # structural damage driven by tau
        # -------------------------
        neurodegeneration += 0.5 * tau + np.random.normal(0.1, 0.05)

        # -------------------------
        # Brain biomarkers
        # -------------------------
        hippocampus = 4200 - 25*age - 400*neurodegeneration + np.random.normal(0, 120)

        ventricles = 3000 + 30*age + 500*neurodegeneration + np.random.normal(0, 150)

        cortical_thickness = 2.8 - 0.01*age - 0.2*neurodegeneration + np.random.normal(0, 0.03)

        # cognitive score
        mmse = (
            30
            - 0.15*(age - 60)
            - 2.5*neurodegeneration
            - 1.5*tau
            + np.random.normal(0, 1)
        )

        mmse = np.clip(mmse, 0, 30)

        rows.append({
            "subject_id": subject,
            "timepoint": t,
            "age": age,
            "state": state,
            "amyloid": amyloid,
            "tau": tau,
            "neurodegeneration": neurodegeneration,
            "hippocampus": hippocampus,
            "ventricles": ventricles,
            "cortical_thickness": cortical_thickness,
            "mmse": mmse
        })

df = pd.DataFrame(rows)

df.to_csv("data/longitudinal_atn.csv", index=False)

print("Saved:", df.shape)
print(df.head())
