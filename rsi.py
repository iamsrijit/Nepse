# RSI_LT_30_LATEST_fixed_close.py
# -*- coding: utf-8 -*-

import os
import re
import base64
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np

# ---------------------------
# CONFIG
# ---------------------------
REPO_OWNER = "iamsrijit"
REPO_NAME = "Nepse"
BRANCH = "main"
KEEP_DAYS = 3000
MANUAL_TRADES_FILE = "manual_trades.csv"

# ---------------------------
# HELPER: get latest CSV from repo root
# ---------------------------
def get_latest_csv_raw_url(owner, repo, branch="main"):
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    resp = requests.get(api_url, params={"ref": branch})
    resp.raise_for_status()
    items = resp.json()

    csv_candidates = {}
    for item in items:
        name = item.get("name", "")
        if name.lower().endswith(".csv"):
            m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
            if m:
                csv_candidates[m.group(1)] = name

    if not csv_candidates:
        raise FileNotFoundError("No dated CSV found.")

    latest_date = max(csv_candidates.keys())
    filename = csv_candidates[latest_date]
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filename}"
    return raw_url, filename

# ---------------------------
# LOAD MAIN MARKET DATA
# ---------------------------
raw_url, found_filename = get_latest_csv_raw_url(REPO_OWNER, REPO_NAME, BRANCH)
print("Using CSV:", found_filename)

df_raw = pd.read_csv(raw_url)

expected_cols = [
    'Symbol','Date','Open','High','Low',
    'Close','Percent Change','Volume'
]

df = df_raw[expected_cols].copy()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

df = df.dropna(subset=['Date','Close'])
df = df.sort_values(['Symbol','Date','Close'])
df = df.drop_duplicates(['Symbol','Date'], keep='last')
df = df.sort_values(['Symbol','Date']).reset_index(drop=True)

# ---------------------------
# LOAD MANUAL TRADES FROM GITHUB
# ---------------------------
manual_trades_url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{MANUAL_TRADES_FILE}"

try:
    manual_df = pd.read_csv(manual_trades_url)
    manual_df['Symbol'] = manual_df['Symbol'].astype(str)
    manual_df.set_index('Symbol', inplace=True)
    print("Loaded manual trades from GitHub")
except Exception:
    manual_df = pd.DataFrame()
    print("No manual_trades.csv found")

# ---------------------------
# INDICATORS
# ---------------------------
def compute_rsi(series, length=10):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

# ---------------------------
# COMPUTE SIGNALS + P/L
# ---------------------------
results = []

for symbol in df['Symbol'].unique():
    sub = df[df['Symbol'] == symbol].copy()
    if len(sub) < 24:
        continue

    sub['RSI_10'] = compute_rsi(sub['Close'], 10)
    sub['RSI_EMA_14'] = compute_ema(sub['RSI_10'], 14)

    latest_close = float(sub.iloc[-1]['Close'])

    mask = sub['RSI_EMA_14'] < 30
    if not mask.any():
        continue

    row = sub[mask].iloc[-1]
    signal_close = float(row['Close'])
    signal_pnl = ((latest_close - signal_close) / signal_close) * 100

    buy_price = qty = invested = current_value = pl = pl_pct = np.nan

    if symbol in manual_df.index:
        buy_price = manual_df.loc[symbol, 'Buy_Price']
        qty = manual_df.loc[symbol, 'Quantity']
        invested = buy_price * qty
        current_value = latest_close * qty
        pl = current_value - invested
        pl_pct = (pl / invested) * 100

    results.append({
        'Symbol': symbol,
        'Signal_Date': row['Date'],
        'Signal': 'Buy',
        'Signal_Close': round(signal_close,2),
        'RSI_EMA_14': round(row['RSI_EMA_14'],2),
        'Latest_Close': round(latest_close,2),
        'Signal_PnL_%': round(signal_pnl,2),
        'Buy_Price': buy_price,
        'Quantity': qty,
        'Invested': invested,
        'Current_Value': current_value,
        'P/L': pl,
        'Manual_PnL_%': pl_pct
    })

signals_df = pd.DataFrame(results)

# ---------------------------
# FILTER DATE RANGE
# ---------------------------
if not signals_df.empty:
    cutoff = datetime.today() - timedelta(days=KEEP_DAYS)
    signals_df = signals_df[signals_df['Signal_Date'] >= cutoff]
    signals_df = signals_df.sort_values('Signal_Date', ascending=False)

# ---------------------------
# SAVE / UPLOAD
# ---------------------------
out_csv = signals_df.to_csv(index=False)
out_file = f"RSI_LT_30_LATEST_{datetime.today().strftime('%Y-%m-%d')}.csv"

token = os.environ.get("GH_TOKEN")

if not token:
    with open(out_file,"w") as f:
        f.write(out_csv)
    print("Saved locally:", out_file)
else:
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{out_file}"
    headers = {"Authorization": f"token {token}"}
    payload = {
        "message": f"Update {out_file}",
        "content": base64.b64encode(out_csv.encode()).decode(),
        "branch": BRANCH
    }
    r = requests.put(url, headers=headers, json=payload)
    print("Uploaded:", out_file)

print("Done | Signals:", len(signals_df))
