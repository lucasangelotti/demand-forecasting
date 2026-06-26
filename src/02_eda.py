import pandas as pd
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_PATH = os.path.join(BASE_DIR, "data", "train_clean.csv")

# ── Load cleaned data ──────────────────────────────────────────────────────
print("Loading cleaned data...")
df = pd.read_csv(CLEAN_PATH, parse_dates=["Date"], low_memory=False)

# Focus on Store 1 — weekly aggregation
store_df = df[df["Store"] == 1].copy()
weekly = (
    store_df.groupby(["Year", "Week", "Month"])
    .agg(
        WeeklySales    = ("Sales", "sum"),
        AvgCustomers   = ("Customers", "mean"),
        PromoWeek      = ("Promo", "max")   # 1 if any promo day that week
    )
    .reset_index()
    .sort_values(["Year", "Week"])
    .reset_index(drop=True)
)

# ── 1. Basic statistics ────────────────────────────────────────────────────
print("\n── 1. BASIC WEEKLY SALES STATISTICS (Store 1) ──────────────────")
stats = weekly["WeeklySales"].describe()
print(f"  Count  : {stats['count']:.0f} weeks")
print(f"  Mean   : €{stats['mean']:,.0f}")
print(f"  Median : €{weekly['WeeklySales'].median():,.0f}")
print(f"  Min    : €{stats['min']:,.0f}")
print(f"  Max    : €{stats['max']:,.0f}")
print(f"  Std Dev: €{stats['std']:,.0f}")

# ── 2. Sales by year ───────────────────────────────────────────────────────
print("\n── 2. AVERAGE WEEKLY SALES BY YEAR ─────────────────────────────")
by_year = weekly.groupby("Year")["WeeklySales"].mean()
for year, avg in by_year.items():
    print(f"  {year}: €{avg:,.0f}")

# ── 3. Sales by month ──────────────────────────────────────────────────────
print("\n── 3. AVERAGE WEEKLY SALES BY MONTH ────────────────────────────")
month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
by_month = weekly.groupby("Month")["WeeklySales"].mean().sort_index()
for month, avg in by_month.items():
    bar = "█" * int(avg / 2000)
    print(f"  {month_names[month]:>3}: €{avg:,.0f}  {bar}")

# ── 4. Promo impact ────────────────────────────────────────────────────────
print("\n── 4. PROMO IMPACT ON WEEKLY SALES ─────────────────────────────")
promo_avg    = weekly[weekly["PromoWeek"] == 1]["WeeklySales"].mean()
no_promo_avg = weekly[weekly["PromoWeek"] == 0]["WeeklySales"].mean()
uplift       = ((promo_avg - no_promo_avg) / no_promo_avg) * 100
print(f"  Promo weeks    : €{promo_avg:,.0f} avg weekly sales")
print(f"  Non-promo weeks: €{no_promo_avg:,.0f} avg weekly sales")
print(f"  Promo uplift   : +{uplift:.1f}%")

# ── 5. Best and worst weeks ────────────────────────────────────────────────
print("\n── 5. TOP 5 BEST WEEKS ──────────────────────────────────────────")
top5 = weekly.nlargest(5, "WeeklySales")[["Year","Week","Month","WeeklySales"]]
for _, row in top5.iterrows():
    print(f"  {int(row.Year)}-W{int(row.Week):02d} ({month_names[int(row.Month)]}): €{row.WeeklySales:,.0f}")

print("\n── 6. TOP 5 WORST WEEKS ─────────────────────────────────────────")
bot5 = weekly.nsmallest(5, "WeeklySales")[["Year","Week","Month","WeeklySales"]]
for _, row in bot5.iterrows():
    print(f"  {int(row.Year)}-W{int(row.Week):02d} ({month_names[int(row.Month)]}): €{row.WeeklySales:,.0f}")

print("\nEDA complete.")