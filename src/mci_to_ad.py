import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# -------------------------
# Load longitudinal data
# -------------------------
df = pd.read_csv("data/longitudinal_brain.csv")

# -------------------------
# Keep only MCI subjects (baseline condition)
# -------------------------
df_mci = df[df["state"] == "MCI"].copy()

# -------------------------
# Define conversion target
# (simulate: if progression is high → convert to AD)
# -------------------------
df_mci["converted_to_ad"] = (df_mci["progression"] > df_mci["progression"].median()).astype(int)

# -------------------------
# Features (brain biomarkers)
# -------------------------
features = [
    "age",
    "hippocampus",
    "ventricles",
    "cortical_thickness",
    "mmse"
]

X = df_mci[features]
y = df_mci["converted_to_ad"]

# -------------------------
# Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Model
# -------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# -------------------------
# Predictions
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# Evaluation
# -------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

plt.figure()
plt.hist(model.predict_proba(X_test)[:,1], bins=20)
plt.title("Probability of Conversion (MCI → AD)")
plt.xlabel("Risk score")
plt.ylabel("Count")
plt.savefig("results/mci_conversion_risk.png")
plt.show()
