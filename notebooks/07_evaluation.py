import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def eval_model(df, target, model_path, prefix, extra_drop=None):
    drop_cols = [target] + (extra_drop or [])
    X = df.drop(columns=drop_cols)
    y = df[target]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    pred = model.predict(X_test)
    residuals = y_test.values - pred

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, pred, alpha=0.3, s=5, rasterized=True)
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel(f"Actual {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(f"{target} — Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(f"model/{prefix}_pred_vs_actual.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=80, edgecolor="none")
    ax.axvline(0, color="r", linewidth=1)
    ax.set_xlabel("Residual")
    ax.set_title(f"{target} — Residuals Distribution")
    plt.tight_layout()
    plt.savefig(f"model/{prefix}_residuals.png", dpi=150)
    plt.close()

    abs_err = np.abs(residuals)
    worst_idx = np.argsort(abs_err)[-10:][::-1]
    compound_cols = X_test.columns.tolist()

    print(f"\n=== {target} — Top 10 worst predictions ===")
    for i in worst_idx:
        row = X_test.iloc[i]
        composition = {c: round(v, 3) for c, v in row.items() if v > 0}
        print(f"  actual={y_test.iloc[i]:.2f}  predicted={pred[i]:.2f}  err={abs_err[i]:.2f}  | {composition}")


tg_df = pd.read_csv("data/tg_dataset.csv")
eval_model(tg_df, "Tg", "model/tg_regressor.pkl", "tg")

dens_df = pd.read_csv("data/tg_density_dataset.csv")
eval_model(dens_df, "density", "model/density_regressor.pkl", "density", extra_drop=["Tg"])

print("\nSaved: tg_pred_vs_actual.png, tg_residuals.png, density_pred_vs_actual.png, density_residuals.png")
