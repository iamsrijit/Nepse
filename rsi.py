# RSI_LT_30_LATEST_WITH_PORTFOLIO_OVERWRITE.py
# -*- coding: utf-8 -*-

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

def delete_old_portfolio_reports(keep_filename):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents"
    r = requests.get(url, headers=HEADERS, params={"ref": BRANCH})
    r.raise_for_status()

    for f in r.json():
        name = f["name"]
        if (
            name.startswith("PORTFOLIO_REPORT")
            and name.endswith(".csv")
            and name != keep_filename
        ):
            del_payload = {
                "message": f"Delete old portfolio file {name}",
                "sha": f["sha"],
                "branch": BRANCH
            }
            del_url = f"{url}/{name}"
            res = requests.delete(del_url, headers=HEADERS, json=del_payload)

            if res.status_code == 200:
                print(f"🗑️ Deleted old file: {name}")
            else:
                print(f"⚠️ Failed to delete {name}: {res.text}")

# ===========================
# GET LATEST MARKET CSV
# ===========================
def get_latest_csv():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents"
    r = requests.get(url, headers=HEADERS, params={"ref": BRANCH})
    r.raise_for_status()

    files = {
        f["name"]: f
        for f in r.json()
        if f["name"].endswith(".csv")
        and "RSI" not in f["name"]
        and "PORTFOLIO" not in f["name"]
    }

    dated_files = {}
    for k in files.keys():
        m = re.search(r"\d{4}-\d{2}-\d{2}", k)
        if m:
            dated_files[m.group()] = k

    if not dated_files:
        raise FileNotFoundError("No dated market CSV found in repo root")

    latest_date = max(dated_files.keys())
    return github_raw(dated_files[latest_date])

# ===========================
# LOAD MARKET DATA
# ===========================
df = pd.read_csv(get_latest_csv())
df["Date"] = pd.to_datetime(df["Date"])
df["Close"] = pd.to_numeric(df["Close"])
df = df.sort_values(["Symbol", "Date"])

latest_market_date = df["Date"].max().strftime("%Y-%m-%d")
latest_close_map = df.groupby("Symbol")["Close"].last().to_dict()

# ===========================
# RSI SIGNAL GENERATION
# ===========================
def compute_rsi(series, n=10):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.ewm(alpha=1 / n, adjust=False).mean() / loss.ewm(
        alpha=1 / n, adjust=False
    ).mean()
    return 100 - (100 / (1 + rs))

signals = []

for sym in df["Symbol"].unique():
    s = df[df["Symbol"] == sym].copy()
    if len(s) < 25:
        continue

    s["RSI"] = compute_rsi(s["Close"])
    s["RSI_EMA_14"] = s["RSI"].ewm(span=14, adjust=False).mean()

    mask = s["RSI_EMA_14"] < 30
    if mask.any():
        row = s.loc[mask].iloc[-1]
        lc = latest_close_map[sym]
        pnl = (lc - row["Close"]) / row["Close"] * 100

        signals.append(
            {
                "Symbol": sym,
                "Date": row["Date"],
                "Signal": "Buy",
                "Entry_Close": round(row["Close"], 2),
                "Latest_Close": round(lc, 2),
                "PnL_%": round(pnl, 2),
            }
        )

signals_df = (
    pd.DataFrame(signals)
    .sort_values("Date", ascending=False)
    .reset_index(drop=True)
)

signal_file = "RSI_LT_30_LATEST.csv"
upload_to_github(signal_file, signals_df.to_csv(index=False))

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

    portfolio_rows.append(
        {
            "Symbol": symbol,
            "Open_Qty": open_qty,
            "Avg_Cost": round(invested / open_qty, 2) if open_qty else 0,
            "Latest_Close": round(last_close, 2),
            "Realized_PnL": round(realized, 2),
            "Unrealized_PnL": round(unrealized, 2),
            "Total_PnL": round(total_pl, 2),
            "Total_PnL_%": round(pl_pct, 2),
        }
    )

portfolio_df = (
    pd.DataFrame(portfolio_rows)
    .sort_values("Total_PnL", ascending=False)
    .reset_index(drop=True)
)

portfolio_file = f"PORTFOLIO_REPORT_{latest_market_date}.csv"
upload_to_github(portfolio_file, portfolio_df.to_csv(index=False))

delete_old_portfolio_reports(portfolio_file)

print("✅ DONE — Signals updated, portfolio dated & old reports removed")
