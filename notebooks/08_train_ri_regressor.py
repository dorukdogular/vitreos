import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import pickle

import glasspy.data.load as L
from glasspy.data import SciGlass

FEATURES = ["SiO2","P2O5","ZrO2","Na2O","Al2O3","Fe2O3","CaO","MgO","K2O","B2O3","BaO","ZnO","Li2O","SrO","La2O3","TiO2","Nb2O5","PbO","Sb2O3","Bi2O3","TeO2","Se"]

sg = SciGlass(
    properties_cfg={
        "path": L._sciglass_path_dict()["properties"][1],
        "translate": L.SciGK_translation,
        "keep": ["RefractiveIndex"],
    }
)

df = sg.data
df.columns = ["_".join(c) if isinstance(c, tuple) else c for c in df.columns]

ri_col = None
if "property_RefractiveIndex" in df.columns:
    ri_col = "property_RefractiveIndex"
else:
    candidates = [c for c in df.columns if "refract" in c.lower() or c.upper().startswith("RI")]
    if candidates:
        ri_col = candidates[0]

print(f"RI column found: {ri_col}")

compound_cols_present = [f"compounds_{f}" for f in FEATURES if f"compounds_{f}" in df.columns]
missing = [f for f in FEATURES if f"compounds_{f}" not in df.columns]
if missing:
    print(f"Missing compound cols: {missing}")

df_ri = df[[c for c in compound_cols_present] + [ri_col]].copy()
df_ri = df_ri[df_ri[ri_col].notna()]
df_ri = df_ri[df_ri[compound_cols_present].gt(0).any(axis=1)]

df_ri.columns = [c.replace("compounds_", "") if c.startswith("compounds_") else c for c in df_ri.columns]
ri_col_clean = ri_col.replace("property_", "")
df_ri = df_ri.rename(columns={ri_col: ri_col_clean})

print(f"Dataset shape: {df_ri.shape}")

X = df_ri[FEATURES].fillna(0)
y = df_ri[ri_col_clean]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)
print(f"RF  MAE={rf_mae:.4f}  RMSE={rf_rmse:.4f}  R2={rf_r2:.4f}")

xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, n_jobs=-1, random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)
print(f"XGB MAE={xgb_mae:.4f}  RMSE={xgb_rmse:.4f}  R2={xgb_r2:.4f}")

if rf_mae <= xgb_mae:
    best_model = rf
    best_name = "RF"
    best_mae = rf_mae
    importances = rf.feature_importances_
else:
    best_model = xgb
    best_name = "XGB"
    best_mae = xgb_mae
    importances = xgb.feature_importances_

print(f"Best: {best_name}  MAE={best_mae:.4f}")

with open("model/ri_regressor.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model/ri_features.json", "w") as f:
    json.dump(FEATURES, f)

fi = pd.Series(importances, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(8, 7))
fi.plot.barh(ax=ax)
ax.set_xlabel("Importance")
ax.set_title("RI Regressor Feature Importance")
plt.tight_layout()
fig.savefig("model/ri_feature_importance.png", dpi=150)
plt.close()
