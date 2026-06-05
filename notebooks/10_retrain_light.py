import os, json
import pandas as pd
import numpy as np
import joblib
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score

OXIDES = ["SiO2","P2O5","ZrO2","Na2O","Al2O3","Fe2O3","CaO","MgO","K2O","B2O3",
          "BaO","ZnO","Li2O","SrO","La2O3","TiO2","Nb2O5","PbO","Sb2O3","Bi2O3","TeO2","Se"]

RF_PARAMS = dict(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)

print("=== Tg ===")
df_tg = pd.read_csv("data/tg_dataset.csv")
X_tg = df_tg[OXIDES]
y_tg = df_tg["Tg"]
X_tr, X_te, y_tr, y_te = train_test_split(X_tg, y_tg, test_size=0.2, random_state=42)
m_tg = RandomForestRegressor(**RF_PARAMS)
m_tg.fit(X_tr, y_tr)
p = m_tg.predict(X_te)
print(f"  MAE={mean_absolute_error(y_te, p):.2f} K  R²={r2_score(y_te, p):.4f}")
joblib.dump(m_tg, "model/tg_regressor.pkl")
print(f"  {os.path.getsize('model/tg_regressor.pkl')/1e6:.1f} MB")

print("=== Density ===")
df_dens = pd.read_csv("data/tg_density_dataset.csv")
X_dens = df_dens[OXIDES]
y_dens = df_dens["density"]
X_tr, X_te, y_tr, y_te = train_test_split(X_dens, y_dens, test_size=0.2, random_state=42)
m_dens = RandomForestRegressor(**RF_PARAMS)
m_dens.fit(X_tr, y_tr)
p = m_dens.predict(X_te)
print(f"  MAE={mean_absolute_error(y_te, p):.4f} g/cm³  R²={r2_score(y_te, p):.4f}")
joblib.dump(m_dens, "model/density_regressor.pkl")
print(f"  {os.path.getsize('model/density_regressor.pkl')/1e6:.1f} MB")

print("=== Refractive Index ===")
import glasspy.data.load as L
from glasspy.data import SciGlass
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
X_tr, X_te, y_tr, y_te = train_test_split(X_ri, y_ri, test_size=0.2, random_state=42)
m_ri = RandomForestRegressor(**RF_PARAMS)
m_ri.fit(X_tr, y_tr)
p = m_ri.predict(X_te)
print(f"  MAE={mean_absolute_error(y_te, p):.4f}  R²={r2_score(y_te, p):.4f}")
joblib.dump(m_ri, "model/ri_regressor.pkl")
print(f"  {os.path.getsize('model/ri_regressor.pkl')/1e6:.1f} MB")

print("=== GFA ===")
CATION_TO_OXIDE = {
    "Si":"SiO2","P":"P2O5","Zr":"ZrO2","Na":"Na2O","Al":"Al2O3",
    "Fe":"Fe2O3","Ca":"CaO","Mg":"MgO","K":"K2O","B":"B2O3",
    "Ba":"BaO","Zn":"ZnO","Li":"Li2O","Sr":"SrO","La":"La2O3",
    "Ti":"TiO2","Nb":"Nb2O5","Pb":"PbO","Sb":"Sb2O3","Bi":"Bi2O3",
    "Te":"TeO2","Se":"Se",
}

def formula_to_oxide_fracs(formula):
    try:
        comp = Composition(formula)
    except Exception:
        return None
    elem_fracs = {str(el): amt for el, amt in comp.fractional_composition.items()}
    oxide_fracs = {}
    for elem, frac in elem_fracs.items():
        if elem == "O":
            continue
        oxide = CATION_TO_OXIDE.get(elem)
        if oxide:
            oxide_fracs[oxide] = oxide_fracs.get(oxide, 0.0) + frac
    total = sum(oxide_fracs.values())
    if total == 0:
        return None
    return {ox: oxide_fracs.get(ox, 0.0) / total for ox in OXIDES}

df_gfa = pd.read_csv("data/clean.csv")
rows, labels = [], []
for _, row in df_gfa.iterrows():
    fracs = formula_to_oxide_fracs(row["formula"])
    if fracs is None:
        continue
    rows.append([fracs[ox] for ox in OXIDES])
    labels.append(row["gfa"])
X_gfa = pd.DataFrame(rows, columns=OXIDES)
y_gfa = pd.Series(labels)
X_tr, X_te, y_tr, y_te = train_test_split(X_gfa, y_gfa, test_size=0.2, random_state=42, stratify=y_gfa)
m_gfa = RandomForestClassifier(**RF_PARAMS)
m_gfa.fit(X_tr, y_tr)
p = m_gfa.predict(X_te)
print(f"  Acc={accuracy_score(y_te, p):.4f}  F1={f1_score(y_te, p, average='weighted'):.4f}")
joblib.dump(m_gfa, "model/gfa_classifier_v2.pkl")
print(f"  {os.path.getsize('model/gfa_classifier_v2.pkl')/1e6:.1f} MB")

total_mb = sum(
    os.path.getsize(f"model/{f}")/1e6
    for f in ["tg_regressor.pkl","density_regressor.pkl","ri_regressor.pkl","gfa_classifier_v2.pkl"]
)
print(f"\nTotal model size: {total_mb:.1f} MB")
