import json
import pickle
import pandas as pd
import numpy as np
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

OXIDES = ["SiO2","P2O5","ZrO2","Na2O","Al2O3","Fe2O3","CaO","MgO","K2O","B2O3",
          "BaO","ZnO","Li2O","SrO","La2O3","TiO2","Nb2O5","PbO","Sb2O3","Bi2O3","TeO2","Se"]

OXIDE_FORMULAS = {
    "SiO2": {"Si": 1, "O": 2},
    "P2O5": {"P": 2, "O": 5},
    "ZrO2": {"Zr": 1, "O": 2},
    "Na2O": {"Na": 2, "O": 1},
    "Al2O3": {"Al": 2, "O": 3},
    "Fe2O3": {"Fe": 2, "O": 3},
    "CaO": {"Ca": 1, "O": 1},
    "MgO": {"Mg": 1, "O": 1},
    "K2O": {"K": 2, "O": 1},
    "B2O3": {"B": 2, "O": 3},
    "BaO": {"Ba": 1, "O": 1},
    "ZnO": {"Zn": 1, "O": 1},
    "Li2O": {"Li": 2, "O": 1},
    "SrO": {"Sr": 1, "O": 1},
    "La2O3": {"La": 2, "O": 3},
    "TiO2": {"Ti": 1, "O": 2},
    "Nb2O5": {"Nb": 2, "O": 5},
    "PbO": {"Pb": 1, "O": 1},
    "Sb2O3": {"Sb": 2, "O": 3},
    "Bi2O3": {"Bi": 2, "O": 3},
    "TeO2": {"Te": 1, "O": 2},
    "Se": {"Se": 1},
}

OXIDE_CATIONS = {
    "SiO2": "Si", "P2O5": "P", "ZrO2": "Zr", "Na2O": "Na", "Al2O3": "Al",
    "Fe2O3": "Fe", "CaO": "Ca", "MgO": "Mg", "K2O": "K", "B2O3": "B",
    "BaO": "Ba", "ZnO": "Zn", "Li2O": "Li", "SrO": "Sr", "La2O3": "La",
    "TiO2": "Ti", "Nb2O5": "Nb", "PbO": "Pb", "Sb2O3": "Sb", "Bi2O3": "Bi",
    "TeO2": "Te", "Se": "Se",
}

CATION_TO_OXIDE = {v: k for k, v in OXIDE_CATIONS.items()}


def formula_to_oxide_fracs(formula: str) -> dict:
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


df = pd.read_csv("data/clean.csv")
print(f"clean.csv: {df.shape}")

rows = []
labels = []
skipped = 0
for _, row in df.iterrows():
    fracs = formula_to_oxide_fracs(row["formula"])
    if fracs is None:
        skipped += 1
        continue
    rows.append([fracs[ox] for ox in OXIDES])
    labels.append(row["gfa"])

print(f"Converted: {len(rows)}, skipped: {skipped}")

X = pd.DataFrame(rows, columns=OXIDES)
y = pd.Series(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
print(f"Accuracy: {acc:.4f}  F1: {f1:.4f}")

with open("model/gfa_classifier_v2.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Saved model/gfa_classifier_v2.pkl")
