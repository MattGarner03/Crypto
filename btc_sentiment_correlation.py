#!/usr/bin/env python3
"""
Analyse correlation between Bitcoin price and sentiment indices.

Requires:
    pip install pandas matplotlib seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------- LOAD DATA -----------------------
file_path = "btc_sentiment_merged.csv"

try:
    merged = pd.read_csv(file_path, parse_dates=["date"])
except FileNotFoundError:
    raise SystemExit(f"❌ File not found: {file_path}\nRun plotBTCsentiment.py first to generate it.")

# Standardise column names
merged.columns = [c.lower() for c in merged.columns]

# Detect BTC price column
price_candidates = ["close", "btc-usd", "price", "btc_price"]
price_col = next((c for c in price_candidates if c in merged.columns), None)
if not price_col:
    raise SystemExit(f"❌ No BTC price column found in {file_path}. Columns: {merged.columns}")

# Detect sentiment column
sent_candidates = ["sentiment_score", "sentiment", "value", "fear & greed (merged)", "alt fgi", "cmc fgi"]
sent_col = next((c for c in sent_candidates if c in merged.columns), None)
if not sent_col:
    raise SystemExit(f"❌ No sentiment column found in {file_path}. Columns: {merged.columns}")

# Clean numeric data
merged[price_col] = pd.to_numeric(merged[price_col], errors="coerce")
merged[sent_col] = pd.to_numeric(merged[sent_col], errors="coerce")
merged = merged.dropna(subset=[price_col, sent_col]).sort_values("date")

print(f"[INFO] Loaded {len(merged)} rows from {file_path}")
print(f"Using sentiment column: '{sent_col}', price column: '{price_col}'")

# ----------------------- STEP 1: CORRELATION COEFFICIENTS -----------------------
print("\n📈 Pearson correlation (same-day):")
pearson_corr = merged[[sent_col, price_col]].corr(method="pearson")
print(pearson_corr)

print("\n📈 Spearman correlation (same-day):")
spearman_corr = merged[[sent_col, price_col]].corr(method="spearman")
print(spearman_corr)

# ----------------------- STEP 2: LAGGED CORRELATIONS -----------------------
print("\n🔁 Lagged correlations (does sentiment lead BTC price?)")
lag_results = []
for lag in range(1, 8):
    merged[f"{sent_col}_lag_{lag}"] = merged[sent_col].shift(lag)
    lag_corr_pearson = merged[f"{sent_col}_lag_{lag}"].corr(merged[price_col], method="pearson")
    lag_corr_spearman = merged[f"{sent_col}_lag_{lag}"].corr(merged[price_col], method="spearman")
    lag_results.append((lag, lag_corr_pearson, lag_corr_spearman))
    print(f"Lag {lag} days → Pearson: {lag_corr_pearson:.4f}, Spearman: {lag_corr_spearman:.4f}")

# Create lag correlation dataframe
lag_df = pd.DataFrame(lag_results, columns=["Lag (days)", "Pearson", "Spearman"])
lag_df.to_csv("btc_sentiment_lagged_correlations.csv", index=False)
print("\n✅ Saved lagged correlation coefficients -> btc_sentiment_lagged_correlations.csv")

# ----------------------- STEP 3: VISUALISATIONS -----------------------

# Overlay plot
plt.figure(figsize=(12, 6))
plt.plot(merged["date"], merged[sent_col], label="Sentiment", alpha=0.7)
plt.plot(merged["date"], merged[price_col] / merged[price_col].max() * 100,  # scale BTC to 0–100
         label="BTC Price (scaled)", alpha=0.7)
plt.title("Bitcoin Price vs Sentiment Over Time")
plt.xlabel("Date")
plt.ylabel("Scaled Values (0–100)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("btc_sentiment_price_overlay.png", dpi=150)
print("✅ Saved overlay plot -> btc_sentiment_price_overlay.png")

# Correlation heatmap (all columns)
corr_matrix = merged[[price_col, sent_col] +
                     [f"{sent_col}_lag_{i}" for i in range(1, 8)]].corr(method="pearson")

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap: BTC Price vs Sentiment (Lag Analysis)")
plt.tight_layout()
plt.savefig("btc_sentiment_corr_heatmap.png", dpi=150)
print("✅ Saved heatmap -> btc_sentiment_corr_heatmap.png")

# Lag correlation line chart
plt.figure(figsize=(8, 5))
plt.plot(lag_df["Lag (days)"], lag_df["Pearson"], marker="o", label="Pearson")
plt.plot(lag_df["Lag (days)"], lag_df["Spearman"], marker="s", label="Spearman")
plt.title("Lagged Correlation: Sentiment vs BTC Price")
plt.xlabel("Lag (days)")
plt.ylabel("Correlation Coefficient")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("btc_sentiment_lagged_plot.png", dpi=150)
print("✅ Saved lag correlation plot -> btc_sentiment_lagged_plot.png")

plt.show()
