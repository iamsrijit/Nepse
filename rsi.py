# -*- coding: utf-8 -*-
# 52W_LOW_LATEST_WITH_PORTFOLIO_OVERWRITE.py
import os
import re
import base64
from datetime import datetime
import requests
import pandas as pd
from collections import deque

# ===========================
# CONFIG
# ===========================
REPO_OWNER = "iamsrijit"
REPO_NAME = "Nepse"
BRANCH = "main"
PORTFOLIO_FILE = "portfolio_trades.csv"
GH_TOKEN = os.environ.get("GH_TOKEN")
if not GH_TOKEN:
    raise RuntimeError("GH_TOKEN not set in environment")

HEADERS = {"Authorization": f"token {GH_TOKEN}"}

# ===========================
# HELPERS
# ===========================
def github_raw(path):
    return f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{path}"

def upload_to_github(filename, content):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{filename}"
    r = requests.get(url, headers=HEADERS)
    payload = {
        "message": f"Update {filename} - latest 52W low scan",
        "content": base64.b64encode(content.encode()).decode(),
        "branch": BRANCH
    }
    if r.status_code == 200:
        payload["sha"] = r.json()["sha"]
    res = requests.put(url, headers=HEADERS, json=payload)
    if res.status_code not in (200, 201):
        raise RuntimeError(f"Upload failed: {res.text}")
    print("✅ Uploaded/Overwritten:", filename)

# ===========================
# GET LATEST MARKET CSV (still used for historical close in portfolio)
# ===========================
def get_latest_csv():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents"
    r = requests.get(url, headers=HEADERS, params={"ref": BRANCH})
    r.raise_for_status()
    
    files = {f['name']: f for f in r.json()
             if f['name'].endswith(".csv") and "RSI" not in f['name'] and "PORTFOLIO" not in f['name']}
    
    dated_files = {}
    for k in files.keys():
        m = re.search(r'\d{4}-\d{2}-\d{2}', k)
        if m:
            dated_files[m.group()] = k
    
    if not dated_files:
        raise FileNotFoundError("No dated CSV files found in repo root.")
    
    latest_date = max(dated_files.keys())
    return github_raw(dated_files[latest_date])

# ===========================
# FETCH TODAY'S DATA USING nepse-scraper
# ===========================
!pip install nepse-scraper -q --upgrade

from nepse_scraper import NepseScraper

scraper = NepseScraper(verify_ssl=False)
print("Fetching today's NEPSE data...\n")

data = scraper.get_today_price()
content = data.get('content', data) if isinstance(data, dict) else []

if not content or not isinstance(content, list):
    raise RuntimeError("No valid data received from NEPSE API.")

print(f"✓ {len(content)} stocks fetched\n")

# Compact float converter
def f(v):
    try:
        return float(str(v).replace(",", ""))
    except:
        return 0.0

# Build today's DataFrame
today_df = pd.DataFrame([
    {
        'Symbol':       item.get('symbol', ''),
        'Name':         item.get('securityName', ''),
        'Close':        f(item.get('closePrice')),
        'LTP':          f(item.get('lastTradedPrice')),
        '52W_Low':      f(item.get('fiftyTwoWeekLow')),
        '52W_High':     f(item.get('fiftyTwoWeekHigh')),
        '%_from_52WLow': round( (f(item.get('closePrice')) - f(item.get('fiftyTwoWeekLow'))) / f(item.get('fiftyTwoWeekLow')) * 100 , 2) if f(item.get('fiftyTwoWeekLow')) > 0 else 0,
        'Date':         item.get('businessDate', '')
    }
    for item in content if isinstance(item, dict)
])

# Filter stocks near 52-week low (within ~1.5% tolerance)
NEAR_LOW_THRESHOLD = 1.5   # adjust if you want stricter (0.5) or looser (3.0)
low_hits = today_df[today_df['%_from_52WLow'] <= NEAR_LOW_THRESHOLD].copy()

low_hits = low_hits.sort_values('%_from_52WLow').reset_index(drop=True)

# Rename columns to match your previous signal format
signals_df = low_hits[['Symbol', 'Name', 'Date', 'Close', '52W_Low', '%_from_52WLow']].rename(columns={
    'Close':     'Latest_Close',
    '%_from_52WLow': 'Distance_from_Low_%'
})

signals_df.insert(2, 'Signal', 'Near 52W Low')

signal_file = "RSI_LT_30_LATEST.csv"   # keeping same filename for compatibility
upload_to_github(signal_file, signals_df.to_csv(index=False))

print(f"Found {len(signals_df)} stocks near 52-week low (within {NEAR_LOW_THRESHOLD}%)\n")

# ===========================
# PORTFOLIO (kept exactly the same)
# ===========================
latest_close_map = today_df.set_index('Symbol')['Close'].to_dict()

pt = pd.read_csv(github_raw(PORTFOLIO_FILE))
pt['Date'] = pd.to_datetime(pt['Date'])

portfolio_rows = []
for symbol in pt['Symbol'].unique():
    trades = pt[pt['Symbol'] == symbol].sort_values('Date')
    open_lots = deque()
    realized = 0
    for _, t in trades.iterrows():
        qty = t['Quantity']
        price = t['Price']
        action = t['Action'].upper()
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
        "Latest_Close": last_close,
        "Realized_PnL": round(realized, 2),
        "Unrealized_PnL": round(unrealized, 2),
        "Total_PnL": round(total_pl, 2),
        "Total_PnL_%": round(pl_pct, 2)
    })

portfolio_df = pd.DataFrame(portfolio_rows)
portfolio_df = portfolio_df.sort_values('Total_PnL', ascending=False).reset_index(drop=True)

portfolio_file = "PORTFOLIO_REPORT.csv"
upload_to_github(portfolio_file, portfolio_df.to_csv(index=False))

print("✅ DONE — 52W Low signals & Portfolio updated") 
