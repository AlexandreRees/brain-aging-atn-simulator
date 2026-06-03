import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -----------------------
# Load longitudinal data
# -----------------------
df = pd.read_csv("data/longitudinal_brain.csv")

fig, ax = plt.subplots()

def update(t):
    ax.clear()

    df_t = df[df["timepoint"] == t]

    scatter = ax.scatter(
        df_t["hippocampus"],
        df_t["ventricles"],
        c=df_t["progression"],
        cmap="viridis",
        alpha=0.7
    )

    ax.set_title(f"Brain Evolution - Timepoint {t}")
    ax.set_xlabel("Hippocampus")
    ax.set_ylabel("Ventricles")

# -----------------------
# Create animation
# -----------------------
ani = animation.FuncAnimation(
    fig,
    update,
    frames=df["timepoint"].max() + 1,
    interval=800
)

# Save animation
ani.save("results/brain_evolution.gif", writer="pillow")

plt.show()
