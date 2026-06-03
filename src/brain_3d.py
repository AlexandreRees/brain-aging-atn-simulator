import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("data/synthetic_dataset.csv")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# color = AD vs CN
colors = df["diagnosis"]

ax.scatter(
    df["hippocampus"],
    df["ventricles"],
    df["cortical_thickness"],
    c=colors,
    cmap="coolwarm",
    alpha=0.6
)

ax.set_xlabel("Hippocampus")
ax.set_ylabel("Ventricles")
ax.set_zlabel("Cortical Thickness")
ax.set_title("3D Brain Feature Space (CN vs AD)")

plt.savefig("results/brain_3d.png")
plt.show()
