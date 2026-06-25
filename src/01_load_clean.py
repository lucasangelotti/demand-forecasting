import pandas as pd
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")

# ── Load raw files ─────────────────────────────────────────────────────────
print("Loading raw files...")
train = pd.read_csv(os.path.join(RAW_DIR, "train.csv"), low_memory=False)
store = pd.read_csv(os.path.join(RAW_DIR, "store.csv"))

print(f"train shape: {train.shape}")
print(f"store shape: {store.shape}")

# ── Merge train + store info ───────────────────────────────────────────────
df = train.merge(store, on="Store", how="left")
print(f"merged shape: {df.shape}")

# ── Parse dates ────────────────────────────────────────────────────────────
df["Date"] = pd.to_datetime(df["Date"])
df["Year"]  = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"]  = df["Date"].dt.isocalendar().week.astype(int)

# ── Filter: only open days with actual sales ───────────────────────────────
df = df[(df["Open"] == 1) & (df["Sales"] > 0)]
print(f"after filtering closed days: {df.shape}")

# ── Check missing values ───────────────────────────────────────────────────
print("\nMissing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ── Fill known nulls ───────────────────────────────────────────────────────
df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0)
df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0)
df["PromoInterval"] = df["PromoInterval"].fillna("None")
df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0)
df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0)

# ── Final check ────────────────────────────────────────────────────────────
print("\nMissing values after cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("\nData types:")
print(df.dtypes)
print("\nSample rows:")
print(df.head(3))

# ── Save cleaned file ──────────────────────────────────────────────────────
CLEAN_PATH = os.path.join(BASE_DIR, "data", "train_clean.csv")
df.to_csv(CLEAN_PATH, index=False)
print(f"\nCleaned file saved to: {CLEAN_PATH}")