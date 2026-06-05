import pandas as pd
import numpy as np

df = pd.read_csv("data/raw_properties.csv")

compound_cols = [c for c in df.columns if c not in ("Tg", "density")]

sparse = [c for c in compound_cols if (df[c].isna() | (df[c] == 0)).mean() > 0.95]
df = df.drop(columns=sparse)
compound_cols = [c for c in df.columns if c not in ("Tg", "density")]
print(f"Compound columns after sparse drop: {len(compound_cols)}")

tg = df[df["Tg"].notna()].copy()
tg = tg[tg[compound_cols].gt(0).any(axis=1)]
tg = tg.drop(columns=["density"])
tg.to_csv("data/tg_dataset.csv", index=False)
print(f"\ntg_dataset: {tg.shape}")
print(f"  Tg mean={tg['Tg'].mean():.1f} std={tg['Tg'].std():.1f} min={tg['Tg'].min():.1f} max={tg['Tg'].max():.1f}")
print(f"  Compound cols: {len([c for c in tg.columns if c != 'Tg'])}")

both = df[df["Tg"].notna() & df["density"].notna()].copy()
both = both[both[compound_cols].gt(0).any(axis=1)]
both.to_csv("data/tg_density_dataset.csv", index=False)
print(f"\ntg_density_dataset: {both.shape}")
print(f"  Tg mean={both['Tg'].mean():.1f} std={both['Tg'].std():.1f} min={both['Tg'].min():.1f} max={both['Tg'].max():.1f}")
print(f"  Compound cols: {len([c for c in both.columns if c not in ('Tg','density')])}")
