import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# --------------------
# Load data
# --------------------
df = pd.read_csv("data/synthetic_dataset.csv")

X = df.drop(columns=["mmse", "diagnosis"])
y = df["mmse"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

# --------------------
# Train models
# --------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    results[name] = mae
    print(name, "MAE:", mae)

# --------------------
# Plot results
# --------------------
plt.figure()
plt.bar(results.keys(), results.values())
plt.title("Model comparison (MAE on MMSE prediction)")
plt.ylabel("MAE")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("results/mae_comparison.png")
plt.show()
