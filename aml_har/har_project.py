import numpy as np
import pandas as pd

df = pd.read_csv("/Users/moli/PycharmProjects/Human-Activity-Recognition/aml_har/data/raw/WISDM_ar_v1.1_raw.txt",
                 header=None,
                 names=["user", "activity", "timestamp", "x", "y", "z"],
                 sep=",",
                 engine="python",
                 on_bad_lines="skip")

print("Shape:", df.shape)
print(df.head(5))
print("\nMissing values per column:\n", df.isna().sum())
print("\nActivity counts:\n", df["activity"].value_counts())

print("Example raw z values:", df["z"].head(3).tolist())

# STEP 2: Clean
df["z"] = df["z"].astype(str).str.replace(";", "", regex=False)

df["user"] = pd.to_numeric(df["user"], errors="coerce")
df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
df["x"] = pd.to_numeric(df["x"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df["z"] = pd.to_numeric(df["z"], errors="coerce")

df = df.dropna()
df = df.drop_duplicates()
df = df.sort_values(["user", "timestamp"]).reset_index(drop=True)

print("After cleaning shape:", df.shape)
print("Missing values after cleaning:\n", df.isna().sum())

# STEP 3: Save cleaned data
df.to_csv("/Users/moli/PycharmProjects/Human-Activity-Recognition/aml_har/data/processed/WISDM_clean.csv", index=False)
print("Saved cleaned dataset to: data/processed/WISDM_clean.csv")

print("\nFirst 3 rows after cleaning:\n", df.head(3))