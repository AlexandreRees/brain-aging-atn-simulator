import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/longitudinal_atn.csv")

# -------------------------
# Amyloid → Tau
# -------------------------
plt.figure()
plt.scatter(df["amyloid"], df["tau"], alpha=0.3)
plt.title("Amyloid → Tau relationship")
plt.xlabel("Amyloid")
plt.ylabel("Tau")
plt.savefig("results/amyloid_tau.png")
plt.show()

# -------------------------
# Tau → Neurodegeneration
# -------------------------
plt.figure()
plt.scatter(df["tau"], df["neurodegeneration"], alpha=0.3)
plt.title("Tau → Neurodegeneration")
plt.xlabel("Tau")
plt.ylabel("Neurodegeneration")
plt.savefig("results/tau_neurodegeneration.png")
plt.show()

# -------------------------
# Neurodegeneration → MMSE
# -------------------------
plt.figure()
plt.scatter(df["neurodegeneration"], df["mmse"], alpha=0.3)
plt.title("Neurodegeneration → MMSE decline")
plt.xlabel("Neurodegeneration")
plt.ylabel("MMSE")
plt.savefig("results/neurodegeneration_mmse.png")
plt.show()

# -------------------------
# Temporal progression
# -------------------------
mean_progression = df.groupby("timepoint")[["amyloid","tau","neurodegeneration"]].mean()

plt.figure()
plt.plot(mean_progression.index, mean_progression["amyloid"], label="Amyloid")
plt.plot(mean_progression.index, mean_progression["tau"], label="Tau")
plt.plot(mean_progression.index, mean_progression["neurodegeneration"], label="Neurodegeneration")

plt.legend()
plt.title("AT(N) progression over time")
plt.xlabel("Timepoint")
plt.ylabel("Level")
plt.savefig("results/atn_progression.png")
plt.show()
