import pandas as pd
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_PATH = os.path.join(BASE_DIR, "data", "train_clean.csv")
OUT_PATH   = os.path.join(BASE_DIR, "data", "forecast_output.csv")

# ── Load cleaned data ──────────────────────────────────────────────────────
print("Loading cleaned data...")
df = pd.read_csv(CLEAN_PATH, parse_dates=["Date"], low_memory=False)

# ── Focus on a single store for clarity ───────────────────────────────────
# Store 1 has complete data across the full date range
STORE_ID = 1
store_df = df[df["Store"] == STORE_ID].copy()

# ── Aggregate to weekly sales ──────────────────────────────────────────────
weekly = (
    store_df.groupby(["Year", "Week"])
    .agg(ActualSales=("Sales", "sum"), AvgCustomers=("Customers", "mean"))
    .reset_index()
)

# Create a sortable period label (e.g. 2013-W01)
weekly["Period"] = weekly["Year"].astype(str) + "-W" + weekly["Week"].astype(str).str.zfill(2)
weekly = weekly.sort_values(["Year", "Week"]).reset_index(drop=True)

print(f"Weekly periods available: {len(weekly)}")
print(weekly.head(5))

# ── Model 1: 4-Week Moving Average ────────────────────────────────────────
weekly["MA4"] = weekly["ActualSales"].rolling(window=4).mean().shift(1)

# ── Model 2: Exponential Smoothing (alpha = 0.3) ──────────────────────────
alpha = 0.3
es = [None]  # first value has no forecast
for i in range(1, len(weekly)):
    prev_actual   = weekly.loc[i - 1, "ActualSales"]
    prev_forecast = es[i - 1] if es[i - 1] is not None else prev_actual
    es.append(alpha * prev_actual + (1 - alpha) * prev_forecast)
weekly["ES03"] = es

# ── Model 3: Same-Week Last Year (seasonal baseline) ──────────────────────
ly = weekly[["Year", "Week", "ActualSales"]].copy()
ly["Year"] = ly["Year"] + 1
ly = ly.rename(columns={"ActualSales": "LastYearSales"})
weekly = weekly.merge(ly, on=["Year", "Week"], how="left")

# ── Calculate errors (MAE — Mean Absolute Error) ──────────────────────────
valid = weekly.dropna(subset=["MA4", "ES03", "LastYearSales"])

mae_ma4 = (valid["ActualSales"] - valid["MA4"]).abs().mean()
mae_es  = (valid["ActualSales"] - valid["ES03"]).abs().mean()
mae_ly  = (valid["ActualSales"] - valid["LastYearSales"]).abs().mean()

print(f"\nModel Accuracy (MAE — Mean Absolute Error, lower is better):")
print(f"  4-Week Moving Average : €{mae_ma4:,.0f}")
print(f"  Exponential Smoothing : €{mae_es:,.0f}")
print(f"  Same-Week Last Year   : €{mae_ly:,.0f}")

# ── Save output ────────────────────────────────────────────────────────────
weekly.to_csv(OUT_PATH, index=False)
print(f"\nForecast output saved to: {OUT_PATH}")