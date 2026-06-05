import pandas as pd
from pymatgen.core import Composition

df = pd.read_csv("data/clean.csv")
print(f"Loaded {len(df)} rows")

def parse_fractions(formula):
    try:
        comp = Composition(formula)
        total = comp.num_atoms
        return {str(el): amt / total for el, amt in comp.items()}
    except Exception:
        return {}

records = df["formula"].map(parse_fractions)
feat_df = pd.DataFrame(list(records)).fillna(0.0)
feat_df = feat_df.loc[:, (feat_df != 0).any(axis=0)]

feat_df["gfa"] = df["gfa"].values

feat_df.to_csv("data/features.csv", index=False)
print(f"Saved to data/features.csv — shape: {feat_df.shape}")
