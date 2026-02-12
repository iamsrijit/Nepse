# -*- coding: utf-8 -*-
import os
import re
import base64
import requests
import pandas as pd
from collections import deque
from io import StringIO

# ===========================
# CONFIG
# ===========================
REPO_OWNER = "iamsrijit"
REPO_NAME = "Nepse"
BRANCH = "main"
PORTFOLIO_FILE = "portfolio_trades.csv"

# Symbols to exclude from 52-week low analysis
EXCLUDED_SYMBOLS = [
    "EBLD852",
    "EBL",
    "EB89",
    "NABILD2089",
    "MBLD2085",
    "SBID89",
    "SBID2090",
    "SBLD2091",
    "NIMBD90",
    "RBBD2088",
    "CCBD88",
    "ULBSL",
    "ICFCD88",
    "EBLD91",
    "ANLB",
    "GBILD84/85",
    "GBILD86/87",
    "NICD88"
]

GH_TOKEN = os.environ.get("GH_TOKEN")
if not GH_TOKEN:
    raise RuntimeError("GH_TOKEN not set in environment")

HEADERS = {"Authorization": f"token {GH_TOKEN}"}

# ===========================
# GITHUB HELPERS
# ===========================
def github_raw(path):
    return f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{path}"

def upload_to_github(filename, content):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{filename}"
    r = requests.get(url, headers=HEADERS)
    payload = {
        "message": f"Upload {filename}",
        "content": base64.b64encode(content.encode()).decode(),
        "branch": BRANCH
    }
    if r.status_code == 200:
        payload["sha"] = r.json()["sha"]
    res = requests.put(url, headers=HEADERS, json=payload)
    if res.status_code not in (200, 201):
        raise RuntimeError(f"Upload failed: {res.text}")
    print("✅ Uploaded/Overwritten:", filename)

def delete_old_files(prefix, keep_filename):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents"
    r = requests.get(url, headers=HEADERS, params={"ref": BRANCH})
    r.raise_for_status()
    for f in r.json():
        name = f["name"]
        if name.startswith(prefix) and name.endswith(".csv") and name != keep_filename:
            del_payload = {
                "message": f"Delete old file {name}",
                "sha": f["sha"],
                "branch": BRANCH
            }
            del_url = f"{url}/{name}"
            res = requests.delete(del_url, headers=HEADERS, json=del_payload)
            if res.status_code == 200:
                print(f"🗑️ Deleted: {name}")
            else:
                print(f"⚠️ Failed to delete {name}: {res.text}")

# ===========================
# GET LATEST ESPEN CSV
# ===========================
def get_latest_espen_csv():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents"
    r = requests.get(url, headers=HEADERS, params={"ref": BRANCH})
    r.raise_for_status()
    
    # Find all espen_*.csv files
    espen_files = {}
    for f in r.json():
        name = f["name"]
        if name.startswith("espen_") and name.endswith(".csv"):
            m = re.search(r"espen_(\d{4}-\d{2}-\d{2})\.csv", name)
            if m:
                espen_files[m.group(1)] = name
    
    if not espen_files:
        raise FileNotFoundError("No espen_*.csv file found")
    
    # Get the latest date
    latest_date = max(espen_files.keys())
    latest_file = espen_files[latest_date]
    
    print(f"📂 Using market data file: {latest_file}")
    return github_raw(latest_file)

# ===========================
# LOAD MARKET DATA
# ===========================
# Read the CSV content
csv_url = get_latest_espen_csv()
response = requests.get(csv_url)
csv_content = response.text

# Find the header line
lines = csv_content.strip().split('\n')
header_line = None
data_start_index = 0

for i, line in enumerate(lines):
    if 'Symbol' in line and 'Date' in line and 'Close' in line:
        header_line = line
        data_start_index = i
        break

if header_line is None:
    raise ValueError("Could not find header line with Symbol, Date, Close columns")

# Reconstruct CSV with header at top
if data_start_index > 0:
    # Header is not at the top, move it
    data_lines = lines[:data_start_index]
    reconstructed_csv = header_line + '\n' + '\n'.join(data_lines)
else:
    # Header is already at top
    reconstructed_csv = csv_content

# Parse the CSV - try tab separator first, then comma
try:
    df = pd.read_csv(StringIO(reconstructed_csv), sep='\t')
    # Check if parsing worked (multiple columns)
    if len(df.columns) == 1:
        # Only one column, try comma separator
        df = pd.read_csv(StringIO(reconstructed_csv), sep=',')
except:
    # Fallback to comma separator
    df = pd.read_csv(StringIO(reconstructed_csv), sep=',')

# CRITICAL FIX: Clean column names IMMEDIATELY after loading
df.columns = df.columns.str.strip()

# Debug: Print actual column names
print(f"📊 Loaded {len(df)} rows")
print(f"📋 Column names: {list(df.columns)}")
print(f"📋 First column name repr: {repr(df.columns[0])}")

# Now we can safely access the columns
# Verify 'Date' column exists
if 'Date' not in df.columns:
    print("❌ ERROR: 'Date' column not found!")
    print(f"Available columns: {list(df.columns)}")
    print(f"Column name representations: {[repr(c) for c in df.columns]}")
    raise KeyError("'Date' column not found in DataFrame")

# Show sample dates for debugging
print(f"📅 Sample dates from CSV: {df['Date'].head(3).tolist()}")

# Store original date column in case we need to retry
original_dates = df["Date"].copy()

# Try multiple date formats
df["Date"] = pd.to_datetime(df["Date"], format='%m/%d/%Y', errors='coerce')

# If that didn't work, try other common formats
if df["Date"].isna().all():
    print("⚠️ Format %m/%d/%Y failed, trying %Y-%m-%d...")
    df["Date"] = pd.to_datetime(original_dates, format='%Y-%m-%d', errors='coerce')

if df["Date"].isna().all():
    print("⚠️ Format %Y-%m-%d failed, trying pandas auto-detection...")
    df["Date"] = pd.to_datetime(original_dates, errors='coerce')

# Check if parsing succeeded
parsed_count = df["Date"].notna().sum()
print(f"✅ Successfully parsed {parsed_count} out of {len(df)} dates")

if parsed_count == 0:
    print("❌ ERROR: All dates failed to parse!")
    print(f"Sample raw date values: {original_dates.head(10).tolist()}")
    raise ValueError("Unable to parse any dates from the Date column")

df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
df = df.dropna(subset=["Symbol", "Date", "Close"])
df = df.sort_values(["Symbol", "Date"])

latest_market_date = df["Date"].max().strftime("%Y-%m-%d")
latest_close_map = df.groupby("Symbol")["Close"].last().to_dict()

print(f"📅 Latest market date: {latest_market_date}")
print(f"📈 Total symbols: {df['Symbol'].nunique()}")

# ===========================
# EMA CROSSOVER ANALYSIS (20 vs 50 on Weekly Data)
# ===========================
print("\n" + "="*50)
print("📊 CALCULATING EMA CROSSOVERS ON WEEKLY DATA")
print("="*50)

ema_crossover_results = []

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

# Process each symbol
for sym in df['Symbol'].unique():
    if sym in EXCLUDED_SYMBOLS:
        continue
    
    # Get daily data for this symbol
    sym_data = df[df['Symbol'] == sym].copy()
    
    # Need at least 50 weeks * 5 trading days = 250 data points for reliable 50 EMA
    if len(sym_data) < 250:
        continue
    
    # Convert daily to weekly data
    # Set Date as index for resampling
    sym_data = sym_data.set_index('Date')
    
    # Resample to weekly data (W = week ending on Sunday)
    # We use 'W-THU' to align with Sunday-Thursday trading week
    weekly_data = sym_data.resample('W-THU').agg({
        'Close': 'last',  # Last close price of the week (Thursday)
        'Symbol': 'first'
    }).dropna()
    
    # Need at least 50 weeks of data for 50 EMA
    if len(weekly_data) < 50:
        continue
    
    # Calculate 20 and 50 period EMAs on weekly data
    weekly_data['EMA_20'] = calculate_ema(weekly_data['Close'], 20)
    weekly_data['EMA_50'] = calculate_ema(weekly_data['Close'], 50)
    
    # Drop NaN values from EMA calculation
    weekly_data = weekly_data.dropna()
    
    if len(weekly_data) < 2:
        continue
    
    # Get the latest two weeks to detect crossover
    latest_week = weekly_data.iloc[-1]
    previous_week = weekly_data.iloc[-2]
    
    latest_close = latest_week['Close']
    latest_ema_20 = latest_week['EMA_20']
    latest_ema_50 = latest_week['EMA_50']
    
    prev_ema_20 = previous_week['EMA_20']
    prev_ema_50 = previous_week['EMA_50']
    
    # Detect crossover
    crossover_signal = None
    
    # Bullish crossover: EMA 20 crosses above EMA 50
    if prev_ema_20 <= prev_ema_50 and latest_ema_20 > latest_ema_50:
        crossover_signal = "BULLISH_CROSS"
    
    # Bearish crossover: EMA 20 crosses below EMA 50
    elif prev_ema_20 >= prev_ema_50 and latest_ema_20 < latest_ema_50:
        crossover_signal = "BEARISH_CROSS"
    
    # Current position (no recent crossover)
    elif latest_ema_20 > latest_ema_50:
        crossover_signal = "BULLISH"
    else:
        crossover_signal = "BEARISH"
    
    # Only add to results if there's a crossover (BUY or SELL signal)
    if crossover_signal in ["BULLISH_CROSS", "BEARISH_CROSS"]:
        # Calculate weekly percentage change
        if len(weekly_data) >= 2:
            previous_close = previous_week['Close']
            weekly_pct_change = ((latest_close - previous_close) / previous_close) * 100
        else:
            weekly_pct_change = 0
        
        # Convert signal to Buy/Sell
        signal = "BUY" if crossover_signal == "BULLISH_CROSS" else "SELL"
        
        # Get the date of the latest week
        latest_date = weekly_data.index[-1].strftime("%Y-%m-%d")
        
        ema_crossover_results.append({
            "Symbol": sym,
            "Date": latest_date,
            "Signal": signal,
            "Weekly_Change_%": round(weekly_pct_change, 2)
        })

print(f"✅ Analyzed EMA crossovers - found {len(ema_crossover_results)} signals")

# Create DataFrame and sort
if ema_crossover_results:
    ema_df = pd.DataFrame(ema_crossover_results)
    
    # Sort by Date (most recent first), then by Signal (BUY first)
    ema_df = ema_df.sort_values(['Date', 'Signal'], ascending=[False, True]).reset_index(drop=True)
    
    # Print summary
    buy_signals = len(ema_df[ema_df['Signal'] == 'BUY'])
    sell_signals = len(ema_df[ema_df['Signal'] == 'SELL'])
    
    print(f"  🟢 BUY signals: {buy_signals}")
    print(f"  🔴 SELL signals: {sell_signals}")
else:
    ema_df = pd.DataFrame(columns=[
        "Symbol", "Date", "Signal", "Weekly_Change_%"
    ])
    print("  ℹ️ No crossovers detected this week")

# Upload EMA crossover CSV
ema_file = f"EMA_CROSSOVER_20_50_WEEKLY_{latest_market_date}.csv"
upload_to_github(ema_file, ema_df.to_csv(index=False))
delete_old_files("EMA_CROSSOVER_20_50_WEEKLY_", ema_file)

# ===========================
# 52-WEEK LOW & HIGH ANALYSIS
# ===========================
signals_threshold = []  # Stocks within 1.5% of 52-week low
all_distances = []  # All stocks with distance from low and high

# Print excluded symbols if any
if EXCLUDED_SYMBOLS:
    print(f"⚠️ Excluding {len(EXCLUDED_SYMBOLS)} symbols")

# Check if 52High and 52Low columns exist in the CSV
has_52w_columns = '52High' in df.columns and '52Low' in df.columns

if not has_52w_columns:
    print("⚠️ Warning: 52High and 52Low columns not found in CSV. Columns available:", list(df.columns))
    print("ℹ️ The CSV should have columns: Symbol, Date, Open, High, Low, Close, Percent Change, Volume, 52High, 52Low")

symbols_with_52w_data = []
symbols_without_52w_data = []

# Get the latest data for each symbol (most recent date)
latest_data = df.sort_values('Date').groupby('Symbol').last().reset_index()

for _, row in latest_data.iterrows():
    sym = row['Symbol']
    
    # Skip excluded symbols
    if sym in EXCLUDED_SYMBOLS:
        continue
    
    # Get 52-week high and low from the CSV columns
    if has_52w_columns:
        high_52w = pd.to_numeric(row['52High'], errors='coerce')
        low_52w = pd.to_numeric(row['52Low'], errors='coerce')
    else:
        # Fallback: calculate from historical data if columns don't exist
        s = df[df["Symbol"] == sym].copy()
        one_year_ago = df["Date"].max() - pd.Timedelta(days=365)
        s_52w = s[s["Date"] >= one_year_ago]
        
        if len(s_52w) < 10:
            symbols_without_52w_data.append(sym)
            continue
        
        high_52w = s_52w["Close"].max()
        low_52w = s_52w["Close"].min()
    
    # Check if we have valid 52-week data
    if pd.isna(high_52w) or pd.isna(low_52w) or high_52w == 0 or low_52w == 0:
        symbols_without_52w_data.append(sym)
        continue
    
    symbols_with_52w_data.append(sym)
    
    # Use latest_close_map to get the close price (same logic as portfolio)
    latest_close = latest_close_map.get(sym, 0)
    
    if latest_close == 0:
        continue
    
    # Calculate distances
    distance_from_low_pct = ((latest_close - low_52w) / low_52w) * 100
    distance_from_high_pct = ((latest_close - high_52w) / high_52w) * 100
    
    # Add to all_distances (for CSV 2)
    all_distances.append({
        "Symbol": sym,
        "Latest_Close": round(latest_close, 2),
        "52_Week_Low": round(low_52w, 2),
        "52_Week_High": round(high_52w, 2),
        "Distance_from_Low_%": round(distance_from_low_pct, 2),
        "Distance_from_High_%": round(distance_from_high_pct, 2)
    })
    
    # Check if latest close is within 1.5% of 52-week low (for CSV 1)
    threshold = low_52w * 1.015  # 1.5% above 52-week low
    
    if latest_close <= threshold:
        signals_threshold.append({
            "Symbol": sym,
            "Latest_Close": round(latest_close, 2),
            "52_Week_Low": round(low_52w, 2),
            "Distance_from_Low_%": round(distance_from_low_pct, 2),
            "Date_at_52W_Low": "N/A"  # Date not available when using CSV columns
        })

# Print statistics
print(f"📊 Symbols with 52-week data: {len(symbols_with_52w_data)}")
print(f"📊 Symbols without 52-week data: {len(symbols_without_52w_data)}")
print(f"✅ Found {len(signals_threshold)} stocks within 1.5% of 52-week low")
print(f"✅ Found {len(all_distances)} total stocks with 52-week data")

# ===========================
# CSV 1: Stocks within 1.5% threshold
# ===========================
if signals_threshold:
    signals_df = (
        pd.DataFrame(signals_threshold)
        .sort_values("Distance_from_Low_%")
        .reset_index(drop=True)
    )
else:
    signals_df = pd.DataFrame(columns=[
        "Symbol", 
        "Latest_Close",
        "52_Week_Low", 
        "Distance_from_Low_%",
        "Date_at_52W_Low"
    ])

low_file = f"52_WEEK_LOW_LATEST_{latest_market_date}.csv"
upload_to_github(low_file, signals_df.to_csv(index=False))
delete_old_files("52_WEEK_LOW_LATEST_", low_file)

# ===========================
# CSV 2: All stocks with distance from low and high
# ===========================
if all_distances:
    distance_df = (
        pd.DataFrame(all_distances)
        .sort_values("Distance_from_Low_%")
        .reset_index(drop=True)
    )
else:
    distance_df = pd.DataFrame(columns=[
        "Symbol",
        "Latest_Close",
        "52_Week_Low",
        "52_Week_High",
        "Distance_from_Low_%",
        "Distance_from_High_%"
    ])

distance_file = f"52_WEEK_DISTANCE_{latest_market_date}.csv"
upload_to_github(distance_file, distance_df.to_csv(index=False))
delete_old_files("52_WEEK_DISTANCE_", distance_file)

# ===========================
# PORTFOLIO REPORT
# ===========================
pt = pd.read_csv(github_raw(PORTFOLIO_FILE))
pt["Date"] = pd.to_datetime(pt["Date"])

portfolio_rows = []
for symbol in pt["Symbol"].unique():
    trades = pt[pt["Symbol"] == symbol].sort_values("Date")
    open_lots = deque()
    realized = 0

    for _, t in trades.iterrows():
        qty = t["Quantity"]
        price = t["Price"]
        action = t["Action"].upper()

        if action == "BUY":
            open_lots.append([qty, price])
        elif action == "SELL":
            sell_qty = qty
            while sell_qty > 0 and open_lots:
                lot_qty, lot_price = open_lots[0]
                used = min(sell_qty, lot_qty)
                realized += used * (price - lot_price)
                lot_qty -= used
                sell_qty -= used
                if lot_qty == 0:
                    open_lots.popleft()
                else:
                    open_lots[0][0] = lot_qty

    open_qty = sum(q for q, _ in open_lots)
    invested = sum(q * p for q, p in open_lots)
    last_close = latest_close_map.get(symbol, 0)
    unrealized = open_qty * last_close - invested
    total_pl = realized + unrealized
    pl_pct = (total_pl / invested * 100) if invested else 0

    portfolio_rows.append({
        "Symbol": symbol,
        "Open_Qty": open_qty,
        "Avg_Cost": round(invested / open_qty, 2) if open_qty else 0,
        "Latest_Close": round(last_close, 2),
        "Realized_PnL": round(realized, 2),
        "Unrealized_PnL": round(unrealized, 2),
        "Total_PnL": round(total_pl, 2),
        "Total_PnL_%": round(pl_pct, 2),
    })

portfolio_df = (
    pd.DataFrame(portfolio_rows)
    .sort_values("Total_PnL", ascending=False)
    .reset_index(drop=True)
)

portfolio_file = f"PORTFOLIO_REPORT_{latest_market_date}.csv"
upload_to_github(portfolio_file, portfolio_df.to_csv(index=False))
delete_old_files("PORTFOLIO_REPORT_", portfolio_file)

print("\n" + "="*50)
print("✅ DONE — All reports generated and uploaded")
print("="*50)
print("📁 Files created:")
print(f"  • {ema_file}")
print(f"  • {low_file}")
print(f"  • {distance_file}")
print(f"  • {portfolio_file}")
