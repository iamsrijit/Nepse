# ... (keep all previous imports, config, github helpers, market data loading unchanged)

# ===========================
# CONFIG (update threshold here)
# ===========================
NEAR_52W_THRESHOLD_PCT = 1.5   # only stocks ≤ this % above 52w low

# ... (keep market data loading, latest_close_map, one_year_ago unchanged)

# ===========================
# 52-WEEK LOW ANALYSIS — only stocks ≤ 1.5% from low
# ===========================
signals = []
for sym in df_market["Symbol"].unique():
    if sym in EXCLUDED_SYMBOLS:
        continue

    s = df_market[df_market["Symbol"] == sym]
    if s["Date"].min() > one_year_ago:
        continue

    s_52w = s[s["Date"] >= one_year_ago]
    if len(s_52w) < 10:  # minimum data points
        continue

    low_52w = s_52w["Close"].min()
    latest = latest_close_map.get(sym, 0)
    if latest <= 0:
        continue

    distance_pct = ((latest - low_52w) / low_52w) * 100

    # Only include if within threshold
    if distance_pct <= NEAR_52W_THRESHOLD_PCT:
        low_date = s_52w[s_52w["Close"] == low_52w]["Date"].iloc[-1]

        signals.append({
            "Symbol": sym,
            "52W_Low_Date": low_date.strftime("%Y-%m-%d"),
            "Latest_Close": round(latest, 2),
            "52W_Low": round(low_52w, 2),
            "Distance_from_52W_Low_%": round(distance_pct, 2)
        })

signals_df = pd.DataFrame(signals).sort_values("Distance_from_52W_Low_%").reset_index(drop=True)
print(f"✅ Found {len(signals_df)} stocks ≤ {NEAR_52W_THRESHOLD_PCT}% above 52-week low")

full_low_file = f"STOCKS_NEAR_52W_LOW_{latest_market_date}.csv"  # nicer name
upload_to_github(full_low_file, signals_df.to_csv(index=False))
delete_old_files("STOCKS_NEAR_52W_LOW_", full_low_file)
delete_old_files("52_WEEK_LOW_LATEST_", None)  # optional: clean up old naming if you want

# ===========================
# PORTFOLIO REPORT (keep as is — includes distance column for your holdings)
# ... (keep the entire portfolio section unchanged)
# It still generates PORTFOLIO_REPORT_*.csv with Distance_from_52W_Low_% column

print("✅ DONE")
print(f"   • Created: {full_low_file} → all stocks ≤ {NEAR_52W_THRESHOLD_PCT}% from 52w low")
print(f"   • Created: PORTFOLIO_REPORT_{latest_market_date}.csv → your holdings + distance info")
