# RSI_Portfolio_Partial_Exit_With_PnL_Percent.py
# -*- coding: utf-8 -*-

import re
from datetime import datetime
import requests
import pandas as pd
import numpy as np
from collections import deque

# ---------------------------
# CONFIG
# ---------------------------
REPO_OWNER = "iamsrijit"
REPO_NAME = "Nepse"
BRANCH = "main"
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
df = pd.read_csv(raw_url)[['Symbol','Date','Close']]
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna().sort_values(['Symbol','Date'])

latest_close_map = df.groupby('Symbol')['Close'].last().to_dict()

# ---------------------------
# LOAD MANUAL TRADES
# ---------------------------
manual_url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{MANUAL_TRADES_FILE}"
trades = pd.read_csv(manual_url)

trades['Trade_Date'] = pd.to_datetime(trades['Trade_Date'], errors='coerce')
trades['Price'] = pd.to_numeric(trades['Price'], errors='coerce')
trades['Quantity'] = pd.to_numeric(trades['Quantity'], errors='coerce')
trades = trades.sort_values(['Symbol','Trade_Date'])

# ---------------------------
# FIFO PORTFOLIO ENGINE
# ---------------------------
portfolio_rows = []

for symbol, grp in trades.groupby('Symbol'):
    inventory = deque()
    realized_pl = 0.0
    realized_cost = 0.0

    for _, t in grp.iterrows():
        price = t['Price']
        qty = int(t['Quantity'])

        # BUY
        if qty > 0:
            inventory.append([qty, price])

        # SELL
        else:
            sell_qty = abs(qty)
            while sell_qty > 0 and inventory:
                buy_qty, buy_price = inventory[0]
                matched = min(buy_qty, sell_qty)

                realized_pl += matched * (price - buy_price)
                realized_cost += matched * buy_price

                inventory[0][0] -= matched
                sell_qty -= matched

                if inventory[0][0] == 0:
                    inventory.popleft()

    # OPEN POSITIONS
    latest_price = latest_close_map.get(symbol, np.nan)
    open_qty = 0
    invested_open = 0
    current_value = 0

    for qty, buy_price in inventory:
        open_qty += qty
        invested_open += qty * buy_price
        current_value += qty * latest_price

    unrealized_pl = current_value - invested_open

    unrealized_pct = (
        (unrealized_pl / invested_open) * 100
        if invested_open > 0 else 0
    )

    total_pl = realized_pl + unrealized_pl
    total_cost = invested_open + realized_cost

    total_pct = (
        (total_pl / total_cost) * 100
        if total_cost > 0 else 0
    )

    portfolio_rows.append({
        'Symbol': symbol,
        'Open_Quantity': open_qty,
        'Invested_Open': round(invested_open, 2),
        'Current_Value': round(current_value, 2),
        'Realized_P/L': round(realized_pl, 2),
        'Unrealized_P/L': round(unrealized_pl, 2),
        'Unrealized_PnL_%': round(unrealized_pct, 2),
        'Total_P/L': round(total_pl, 2),
        'Total_PnL_%': round(total_pct, 2)
    })

portfolio_df = pd.DataFrame(portfolio_rows)

# ---------------------------
# PORTFOLIO SUMMARY
# ---------------------------
summary = {
    'Total_Invested_Open': portfolio_df['Invested_Open'].sum(),
    'Total_Current_Value': portfolio_df['Current_Value'].sum(),
    'Total_Realized_P/L': portfolio_df['Realized_P/L'].sum(),
    'Total_Unrealized_P/L': portfolio_df['Unrealized_P/L'].sum()
}

summary['Portfolio_PnL_%'] = (
    (summary['Total_Current_Value'] -
     summary['Total_Invested_Open'] +
     summary['Total_Realized_P/L'])
    / summary['Total_Invested_Open'] * 100
)

summary_df = pd.DataFrame([summary])

# ---------------------------
# SAVE OUTPUT
# ---------------------------
out_file = f"PORTFOLIO_REPORT_{datetime.today().strftime('%Y-%m-%d')}.csv"
final_df = pd.concat([portfolio_df, pd.DataFrame([{}]), summary_df])
final_df.to_csv(out_file, index=False)

print("Saved:", out_file)
print("\nPORTFOLIO SUMMARY")
print(summary_df.to_string(index=False))
