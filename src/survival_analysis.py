import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("data/longitudinal_atn.csv")

# -------------------------
# Build survival dataset at subject level
# -------------------------
df_subject = df.groupby("subject_id").agg({
    "amyloid": "mean",
    "tau": "mean",
    "neurodegeneration": "mean",
    "age": "mean",
    "mmse": "mean",
    "timepoint": "max"
}).reset_index()

# -------------------------
# Simulate event: conversion to AD
# (higher tau + neurodegeneration → faster conversion)
# -------------------------
risk_score = (
    0.4 * df_subject["tau"]
    + 0.6 * df_subject["neurodegeneration"]
    + np.random.normal(0, 0.2, len(df_subject))
)

df_subject["event"] = (risk_score > np.median(risk_score)).astype(int)

# time-to-event = last observed timepoint
df_subject["duration"] = df_subject["timepoint"] + np.random.randint(1, 5, len(df_subject))

# -------------------------
# Survival model (Cox Proportional Hazards)
# -------------------------
cph = CoxPHFitter()

features = ["amyloid", "tau", "neurodegeneration", "age"]

cph.fit(df_subject[features + ["duration", "event"]],
        duration_col="duration",
        event_col="event")

# -------------------------
# Print summary
# -------------------------
print(cph.summary)

# -------------------------
# Plot survival curves
# -------------------------
plt.figure()
cph.plot()
plt.title("Cox Model Hazard Ratios")
plt.tight_layout()
plt.savefig("results/survival_hazard_ratios.png")
plt.show()
