import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from xgboost import XGBRegressor

df = pd.read_csv("data/tg_dataset.csv")
X = df.drop(columns=["Tg"])
y = df["Tg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, n_jobs=-1, random_state=42, verbosity=0),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = root_mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results[name] = {"mae": mae, "rmse": rmse, "r2": r2, "model": model}
    print(f"{name}: MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

best_name = min(results, key=lambda k: results[k]["mae"])
best_model = results[best_name]["model"]
print(f"\nBest: {best_name}")

with open("model/tg_regressor.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model/tg_features.json", "w") as f:
    json.dump(X.columns.tolist(), f)

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
else:
    importances = np.abs(best_model.coef_)

feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 6))
feat_imp.plot(kind="barh", ax=ax)
ax.set_title(f"Feature Importance — {best_name}")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("model/tg_feature_importance.png", dpi=150)
print("Saved model/tg_feature_importance.png")
