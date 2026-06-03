import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/synthetic_dataset.csv")

X = df.drop(columns=["age", "diagnosis"])
y = df["age"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
print("Brain Age MAE:", mae)

plt.scatter(y_test, preds)
plt.xlabel("True Age")
plt.ylabel("Predicted Brain Age")
plt.title("Brain Age Model")
plt.savefig("results/brain_age.png")
plt.show()
