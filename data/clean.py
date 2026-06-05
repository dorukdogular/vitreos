import pandas as pd

df = pd.read_csv("/Users/dorukdogular/Documents/Programming/vitreos/data/raw.csv")

df = df.loc[:, df.isnull().mean() <= 0.4]

df = df.drop_duplicates()

df.to_csv("/Users/dorukdogular/Documents/Programming/vitreos/data/clean.csv", index=False)

print(df.shape)
