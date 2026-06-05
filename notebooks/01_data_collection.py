import pandas as pd
from matminer.datasets import load_dataset

ternary = load_dataset("glass_ternary_landolt")
binary = load_dataset("glass_binary")

merged = pd.concat([ternary, binary], axis=0, ignore_index=True, sort=False)

print("Shape:", merged.shape)
print("Columns:", merged.columns.tolist())
print("Null counts:\n", merged.isnull().sum())

merged.to_csv("/Users/dorukdogular/Documents/Programming/vitreos/data/raw.csv", index=False)
print("Saved to data/raw.csv")
