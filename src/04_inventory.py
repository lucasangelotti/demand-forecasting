import pandas as pd
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORECAST_PATH = os.path.join(BASE_DIR, "data", "forecast_output.csv")
OUT_PATH   = os.path.join(BASE_DIR, "data", "inventory_output.csv")

# ── Load forecast output ───────────────────────────────────────────────────
print("Loading forecast data...")
df = pd.read_csv(FORECAST_PATH)

# ── Inventory parameters ───────────────────────────────────────────────────
# These are realistic assumptions for a retail store
LEAD_TIME_WEEKS    = 2      # supplier takes 2 weeks to deliver
SERVICE_LEVEL_Z    = 1.65   # 95% service level (Z-score)
HOLDING_COST       = 0.25   # 25% of item value per year (industry standard)
ORDER_COST         = 100    # fixed cost per order in €
UNIT_COST          = 2.50   # average cost per unit sold (estimated)

# ── Use best forecast (Same-Week Last Year) as demand estimate ─────────────
df = df.dropna(subset=["LastYearSales"]).copy()
df["Forecast"] = df["LastYearSales"]

# ── Weekly demand statistics ───────────────────────────────────────────────
avg_weekly_demand = df["Forecast"].mean()
std_weekly_demand = df["Forecast"].std()

print(f"\nDemand Statistics (Store 1 — weekly):")
print(f"  Average weekly demand : €{avg_weekly_demand:,.0f} in sales")
print(f"  Std deviation         : €{std_weekly_demand:,.0f}")

# ── Safety Stock ──────────────────────────────────────────────────────────
# Safety Stock = Z * std_demand * sqrt(lead_time)
safety_stock = SERVICE_LEVEL_Z * std_weekly_demand * np.sqrt(LEAD_TIME_WEEKS)

# ── Reorder Point (ROP) ───────────────────────────────────────────────────
# ROP = avg_demand * lead_time + safety_stock
rop = avg_weekly_demand * LEAD_TIME_WEEKS + safety_stock

# ── EOQ (Economic Order Quantity) ─────────────────────────────────────────
# EOQ = sqrt(2 * annual_demand * order_cost / holding_cost_per_unit)
annual_demand_units = (avg_weekly_demand / UNIT_COST) * 52
holding_cost_per_unit = UNIT_COST * HOLDING_COST
eoq = np.sqrt((2 * annual_demand_units * ORDER_COST) / holding_cost_per_unit)

print(f"\nInventory KPIs (Key Performance Indicators):")
print(f"  Safety Stock          : €{safety_stock:,.0f} in sales value")
print(f"  Reorder Point (ROP)   : €{rop:,.0f} in sales value")
print(f"  EOQ (Economic Order Quantity) : {eoq:,.0f} units")

# ── Flag stockout risk weeks ───────────────────────────────────────────────
df["StockoutRisk"] = df["Forecast"] > rop
high_risk = df[df["StockoutRisk"] == True]
print(f"\nHigh stockout risk weeks: {len(high_risk)} out of {len(df)}")

# ── Add KPI columns to dataframe ───────────────────────────────────────────
df["SafetyStock"]   = round(safety_stock, 0)
df["ReorderPoint"]  = round(rop, 0)
df["EOQ"]           = round(eoq, 0)

# ── Save output ────────────────────────────────────────────────────────────
df.to_csv(OUT_PATH, index=False)
print(f"\nInventory output saved to: {OUT_PATH}")