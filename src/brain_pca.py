import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------
# Load data
# -----------------------
df = pd.read_csv("data/synthetic_dataset.csv")

features = ["hippocampus", "ventricles", "cortical_thickness", "mmse"]

X = df[features]

# -----------------------
# Standardization (IMPORTANT)
# -----------------------
X_scaled = StandardScaler().fit_transform(X)

# -----------------------
# PCA projection
# -----------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# -----------------------
# Plot
# -----------------------
plt.figure()
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df["diagnosis"],
    cmap="coolwarm",
    alpha=0.6
)

plt.title("2D Brain Space (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Diagnosis (0=CN, 1=AD)")
plt.tight_layout()

plt.savefig("results/brain_pca_2d.png")
plt.show()
