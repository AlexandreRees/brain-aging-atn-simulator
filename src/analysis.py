import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/synthetic_dataset.csv")

# -------------------------
# 1. Correlation matrix
# -------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("results/correlation_matrix.png")
plt.show()

# -------------------------
# 2. Hippocampus vs diagnosis
# -------------------------
plt.figure()
plt.scatter(df["hippocampus"], df["diagnosis"])
plt.title("Hippocampus vs AD diagnosis")
plt.savefig("results/hipp_vs_ad.png")
plt.show()

# -------------------------
# 3. MMSE distribution
# -------------------------
plt.figure()
plt.hist(df["mmse"], bins=30)
plt.title("MMSE distribution")
plt.savefig("results/mmse.png")
plt.show()
