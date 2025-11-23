import pandas as pd

df = pd.read_csv("data/processed/processed.csv")
print("Total rows:", len(df))
dups = df.duplicated(subset=["text"], keep=False)
print("Total duplicate rows (exact same text):", int(dups.sum()))

if dups.sum() > 0:
    print("\nSample duplicates:")
    print(df[dups].head(10))
