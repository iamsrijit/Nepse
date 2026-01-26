# 52_WEEK_LOW_LATEST_WITH_PORTFOLIO_OVERWRITE.py
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

# Reconstruct CSV with header at top and data below it
if data_start_index > 0:
    # Header is not at the top, we need data that comes AFTER the header
    data_lines = lines[data_start_index + 1:]  # FIX: Get lines AFTER header, not before
    reconstructed_csv = header_line + '\n' + '\n'.join(data_lines)
else:
    # Header is already at top
    reconstructed_csv = csv_content

# Detect separator by checking the header
if '\t' in header_line:
    sep = '\t'
elif ',' in header_line:
    sep = ','
else:
    sep = ','  # default

print(f"🔍 Detected separator: {repr(sep)}")

# Parse the CSV
df = pd.read_csv(StringIO(reconstructed_csv), sep=sep)

# Clean column names IMMEDIATELY
df.columns = df.columns.str.strip()

print(f"📊 Loaded {len(df)} rows")
print(f"📋 Columns: {list(df.columns)}")

# Verify required columns
required_cols = ['Symbol', 'Date', 'Close']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Show sample dates for debugging
print(f"📅 Sample dates: {df['Date'].head(3).tolist()}")

# Parse dates - try different formats
date_col = df['Date'].astype(str).str.strip()  # Ensure string and strip whitespace

# Try format 1: M/D/YYYY or MM/DD/YYYY
df['Date'] = pd.to_datetime(date_col, format='%m/%d/%Y', errors='coerce')
parsed = df['Date'].notna().sum()

if parsed == 0:
    # Try format 2: YYYY-MM-DD
    print("⚠️ Trying format YYYY-MM-DD...")
    df['Date'] = pd.to_datetime(date_col, format='%Y-%m-%d', errors='coerce')
    parsed = df['Date'].notna().sum()

if parsed == 0:
    # Try format 3: Auto-detect
    print("⚠️ Trying auto-detection...")
    df['Date'] = pd.to_datetime(date_col, errors='coerce')
    parsed = df['Date'].notna().sum()

print(f"✅ Parsed {parsed}/{len(df)} dates")

if parsed == 0:
    print(f"❌ Failed to parse dates. Sample values: {date_col.head(10).tolist()}")
    raise ValueError("Could not parse any dates")

# Convert Close to numeric
df["Close"] = pd.to_numeric(df["Close"], errors='coerce')

# Drop rows with missing critical data
df = df.dropna(subset=["Symbol", "Date", "Close"])
df = df.sort_values(["Symbol", "Date"])

print(f"📊 After cleaning: {len(df)} rows, {df['Symbol'].nunique()} symbols")

latest_market_date = df["Date"].max().strftime("%Y-%m-%d")
latest_close_map = df.groupby("Symbol")["Close"].last().to_dict()

print(f"📅 Latest market date: {latest_market_date}")

# ===========================
# 52-WEEK LOW SIGNALS
# ===========================
signals = []
one_year_ago = df["Date"].max() - pd.Timedelta(days=365)

if EXCLUDED_SYMBOLS:
    print(f"⚠️ Excluding {len(EXCLUDED_SYMBOLS)} symbols")

symbols_with_52w_data = []
symbols_without_52w_data = []

for sym in df["Symbol"].unique():
    if sym in EXCLUDED_SYMBOLS:
        continue
    
    s = df[df["Symbol"] == sym].copy()
    earliest_date = s["Date"].min()
    
    if earliest_date > one_year_ago:
        symbols_without_52w_data.append(sym)
        continue
    
    symbols_with_52w_data.append(sym)
    s_52w = s[s["Date"] >= one_year_ago]
    
    if len(s_52w) < 10:
        continue
    
    low_52w = s_52w["Close"].min()
    latest_close = latest_close_map[sym]
    threshold = low_52w * 1.015
    
    if latest_close <= threshold:
        distance_pct = ((latest_close - low_52w) / low_52w) * 100
        low_date = s_52w[s_52w["Close"] == low_52w]["Date"].iloc[-1]
        
        signals.append({
            "Symbol": sym,
            "Date_at_52W_Low": low_date.strftime("%Y-%m-%d"),
            "Latest_Close": round(latest_close, 2),
            "52_Week_Low": round(low_52w, 2),
            "Distance_from_Low_%": round(distance_pct, 2)
        })

print(f"📊 Symbols with 52w data: {len(symbols_with_52w_data)}")
print(f"📊 Symbols without 52w data: {len(symbols_without_52w_data)}")
print(f"✅ Found {len(signals)} stocks near 52-week low")

if signals:
    signals_df = pd.DataFrame(signals).sort_values("Distance_from_Low_%").reset_index(drop=True)
else:
    signals_df = pd.DataFrame(columns=[
        "Symbol", "Date_at_52W_Low", "Latest_Close", "52_Week_Low", "Distance_from_Low_%"
    ])

low_file = f"52_WEEK_LOW_LATEST_{latest_market_date}.csv"
upload_to_github(low_file, signals_df.to_csv(index=False))
delete_old_files("52_WEEK_LOW_LATEST_", low_file)

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

print("✅ DONE — 52-Week Low & Portfolio dated, old files cleaned")
