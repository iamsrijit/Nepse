# RSI_Portfolio_System.py
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
# GET LATEST MARKET CSV
# ---------------------------
def get_latest_csv_raw_url(owner, repo, branch="main"):
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    r = requests.get(api_url, params={"ref": branch})
    r.raise_for_status()
    items = r.json()

    dated = {}
    for it in items:
        if it["name"].endswith(".csv"):
            m = re.search(r"(\d{4}-\d{2}-\d{2})", it["name"])
            if m:
                dated[m.group(1)] = it["name"]

    latest = max(dated.keys())
    file = dated[latest]
    raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file}"
    return raw, file

# ---------------------------
# LOAD MARKET DATA
# ---------------------------
raw_url, market_file = get_latest_csv_raw_url(REPO_OWNER, REPO_NAME, BRANCH)
print("Market file:", market_file)

df = pd.read_csv(raw_url)
df = df[['Symbol','Date','Open','High','Low','Close','Percent Change','Volume']]
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Date','Close'])
df = df.sort_values(['Symbol','Date','Close'])
df = df.drop_duplicates(['Symbol','Date'], keep='last')
df = df.sort_values(['Symbol','Date']).reset_index(drop=True)

# ---------------------------
# LOAD MANUAL TRADES
# ---------------------------
manual_url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{MANUAL_TRADES_FILE}"
manual = pd.read_csv(manual_url)

manual['Buy_Date'] = pd.to_datetime(manual['Buy_Date'], errors='coerce')
manual['Exit_Price'] = pd.to_numeric(manual['Exit_Price'], errors='coerce')
manual['Buy_Price'] = pd.to_numeric(manual['Buy_Price'], errors='coerce')
manual['Quantity'] = pd.to_numeric(manual['Quantity'], errors='coerce')

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
    return 100 - (100 / (1 + rs))

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

# ---------------------------
# SIGNALS
# ---------------------------
signals = []

for symbol in df['Symbol'].unique():
    sub = df[df['Symbol'] == symbol]
    if len(sub) < 24:
        continue

    sub = sub.copy()
    sub['RSI_10'] = compute_rsi(sub['Close'], 10)
    sub['RSI_EMA_14'] = compute_ema(sub['RSI_10'], 14)

    if (sub['RSI_EMA_14'] < 30).any():
        row = sub[sub['RSI_EMA_14'] < 30].iloc[-1]
        latest_close = sub.iloc[-1]['Close']
        signals.append({
            'Symbol': symbol,
            'Signal_Date': row['Date'],
            'Signal_Close': row['Close'],
            'Latest_Close': latest_close,
            'RSI_EMA_14': round(row['RSI_EMA_14'],2),
            'Signal_PnL_%': round(((latest_close - row['Close']) / row['Close']) * 100, 2)
        })

signals_df = pd.DataFrame(signals)

# ---------------------------
# PORTFOLIO CALCULATION
# ---------------------------
portfolio_rows = []
latest_close_map = df.groupby('Symbol')['Close'].last().to_dict()

for _, t in manual.iterrows():
    symbol = t['Symbol']
    buy_price = t['Buy_Price']
    qty = t['Quantity']
    exit_price = t['Exit_Price']

    invested = buy_price * qty

    if not pd.isna(exit_price):
        current_value = exit_price * qty
        realized_pl = current_value - invested
        unrealized_pl = 0
        status = "Closed"
    else:
        latest = latest_close_map.get(symbol, np.nan)
        current_value = latest * qty if not pd.isna(latest) else np.nan
        realized_pl = 0
        unrealized_pl = current_value - invested
        status = "Open"

    portfolio_rows.append({
        'Symbol': symbol,
        'Buy_Date': t['Buy_Date'],
        'Buy_Price': buy_price,
        'Quantity': qty,
        'Exit_Price': exit_price,
        'Status': status,
        'Invested': invested,
        'Current_Value': current_value,
        'Realized_P/L': realized_pl,
        'Unrealized_P/L': unrealized_pl
    })

portfolio_df = pd.DataFrame(portfolio_rows)

# ---------------------------
# PORTFOLIO SUMMARY
# ---------------------------
summary = {
    'Total_Invested': portfolio_df['Invested'].sum(),
    'Total_Current_Value': portfolio_df['Current_Value'].sum(),
    'Total_Realized_P/L': portfolio_df['Realized_P/L'].sum(),
    'Total_Unrealized_P/L': portfolio_df['Unrealized_P/L'].sum()
}

summary['Portfolio_PnL_%'] = (
    (summary['Total_Current_Value'] - summary['Total_Invested'])
    / summary['Total_Invested'] * 100
)

summary_df = pd.DataFrame([summary])

# ---------------------------
# SAVE OUTPUT
# ---------------------------
out = {
    "signals": signals_df,
    "portfolio": portfolio_df,
    "summary": summary_df
}

out_file = f"PORTFOLIO_REPORT_{datetime.today().strftime('%Y-%m-%d')}.csv"

final_df = pd.concat([
    portfolio_df,
    pd.DataFrame([{}]),
    summary_df
])

final_df.to_csv(out_file, index=False)
print("Saved:", out_file)

print("\nPORTFOLIO SUMMARY")
print(summary_df.to_string(index=False))
