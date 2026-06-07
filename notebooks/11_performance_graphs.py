import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import glasspy.data.load as L
from glasspy.data import SciGlass

OXIDES = ["SiO2","P2O5","ZrO2","Na2O","Al2O3","Fe2O3","CaO","MgO","K2O","B2O3",
          "BaO","ZnO","Li2O","SrO","La2O3","TiO2","Nb2O5","PbO","Sb2O3","Bi2O3","TeO2","Se"]

# --- Load models ---
m_tg   = joblib.load("model/tg_regressor.pkl")
m_dens = joblib.load("model/density_regressor.pkl")
m_ri   = joblib.load("model/ri_regressor.pkl")

# --- Tg ---
df_tg = pd.read_csv("data/tg_dataset.csv")
X_tg = df_tg[OXIDES]
y_tg = df_tg["Tg"]
_, X_te_tg, _, y_te_tg = train_test_split(X_tg, y_tg, test_size=0.2, random_state=42)
p_tg = m_tg.predict(X_te_tg)
mae_tg = mean_absolute_error(y_te_tg, p_tg)
r2_tg  = r2_score(y_te_tg, p_tg)

# --- Density ---
df_dens = pd.read_csv("data/tg_density_dataset.csv")
X_dens = df_dens[OXIDES]
y_dens = df_dens["density"]
_, X_te_dens, _, y_te_dens = train_test_split(X_dens, y_dens, test_size=0.2, random_state=42)
p_dens = m_dens.predict(X_te_dens)
mae_dens = mean_absolute_error(y_te_dens, p_dens)
r2_dens  = r2_score(y_te_dens, p_dens)

# --- RI ---
sg = SciGlass(properties_cfg={
    "path": L._sciglass_path_dict()["properties"][1],
    "translate": L.SciGK_translation,
    "keep": ["RefractiveIndex"],
})
df_ri = sg.data.copy()
df_ri.columns = ["_".join(c) if isinstance(c, tuple) else c for c in df_ri.columns]
ri_col = "property_RefractiveIndex"
compound_cols = [f"compounds_{ox}" for ox in OXIDES if f"compounds_{ox}" in df_ri.columns]
df_ri = df_ri[compound_cols + [ri_col]].dropna(subset=[ri_col])
df_ri.columns = [c.replace("compounds_", "") if c.startswith("compounds_") else c for c in df_ri.columns]
df_ri = df_ri.rename(columns={ri_col: "RI"})
for col in OXIDES:
    if col not in df_ri.columns:
        df_ri[col] = 0.0
df_ri = df_ri[df_ri["RI"].between(1.3, 2.8)]
X_ri = df_ri[OXIDES].fillna(0)
y_ri = df_ri["RI"]
_, X_te_ri, _, y_te_ri = train_test_split(X_ri, y_ri, test_size=0.2, random_state=42)
p_ri = m_ri.predict(X_te_ri)
mae_ri = mean_absolute_error(y_te_ri, p_ri)
r2_ri  = r2_score(y_te_ri, p_ri)

print(f"Tg:      MAE={mae_tg:.2f} K   R²={r2_tg:.4f}")
print(f"Density: MAE={mae_dens:.4f} g/cm³  R²={r2_dens:.4f}")
print(f"RI:      MAE={mae_ri:.4f}   R²={r2_ri:.4f}")

# ── 1. Predicted vs Actual — 3-panel ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Predicted vs Actual — Test Set", fontsize=14, fontweight="bold")

panels = [
    (axes[0], y_te_tg,   p_tg,   "Tg (K)",        f"R²={r2_tg:.3f}  MAE={mae_tg:.1f} K"),
    (axes[1], y_te_dens, p_dens, "Density (g/cm³)", f"R²={r2_dens:.3f}  MAE={mae_dens:.3f}"),
    (axes[2], y_te_ri,   p_ri,  "Refractive Index", f"R²={r2_ri:.3f}  MAE={mae_ri:.4f}"),
]
for ax, y_true, y_pred, label, stats in panels:
    ax.scatter(y_true, y_pred, alpha=0.25, s=4, rasterized=True, color="#2c7bb6")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel(f"Actual {label}", fontsize=10)
    ax.set_ylabel(f"Predicted {label}", fontsize=10)
    ax.set_title(label, fontsize=11)
    ax.text(0.05, 0.93, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig("model/pred_vs_actual_all.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved model/pred_vs_actual_all.png")

# ── 2. Metrics Summary Bar Chart ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Model Performance Summary", fontsize=13, fontweight="bold")

props   = ["Tg", "Density", "Refractive Index"]
r2s     = [r2_tg, r2_dens, r2_ri]
maes    = [mae_tg, mae_dens, mae_ri]
mae_labels = [f"{mae_tg:.1f} K", f"{mae_dens:.3f} g/cm³", f"{mae_ri:.4f}"]
colors  = ["#2c7bb6", "#d7191c", "#1a9641"]

ax = axes[0]
bars = ax.bar(props, r2s, color=colors, edgecolor="white", linewidth=0.5)
ax.set_ylim(0, 1)
ax.set_ylabel("R²", fontsize=11)
ax.set_title("R² Score (higher is better)", fontsize=10)
for bar, val in zip(bars, r2s):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.tick_params(axis="x", labelsize=9)

# Normalize MAE to property range for comparable display
ranges = {"Tg": 1200, "Density": 9, "Refractive Index": 1.5}
mae_norm = [mae_tg/1200, mae_dens/9, mae_ri/1.5]

ax = axes[1]
bars = ax.bar(props, mae_norm, color=colors, edgecolor="white", linewidth=0.5)
ax.set_ylabel("Normalized MAE (fraction of range)", fontsize=9)
ax.set_title("Normalized MAE (lower is better)", fontsize=10)
for bar, lbl in zip(bars, mae_labels):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            lbl, ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.tick_params(axis="x", labelsize=9)

plt.tight_layout()
plt.savefig("model/metrics_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved model/metrics_summary.png")

# ── 3. Feature Importance — 3-panel ──────────────────────────────────────────
fi_tg   = pd.Series(m_tg.feature_importances_,   index=OXIDES)
fi_dens = pd.Series(m_dens.feature_importances_, index=OXIDES)
fi_ri   = pd.Series(m_ri.feature_importances_,   index=OXIDES)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Feature Importance by Oxide", fontsize=13, fontweight="bold")

for ax, fi, title, color in [
    (axes[0], fi_tg,   "Tg",              "#2c7bb6"),
    (axes[1], fi_dens, "Density",         "#d7191c"),
    (axes[2], fi_ri,   "Refractive Index","#1a9641"),
]:
    fi_sorted = fi.sort_values()
    ax.barh(fi_sorted.index, fi_sorted.values, color=color, edgecolor="none")
    ax.set_xlabel("Importance", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()
plt.savefig("model/feature_importance_all.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved model/feature_importance_all.png")
