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
# how far back to keep signals (days)
KEEP_DAYS = 3000

# ---------------------------
# UTIL: find latest CSV in repo root (public repo)
# ---------------------------
def get_latest_csv_raw_url(owner: str, repo: str, branch: str = "main"):
    """
    Look up repository root contents via GitHub API and return the raw.githubusercontent.com URL
    for the CSV file whose filename contains a YYYY-MM-DD date and is the latest date.
    Works for public repos.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    resp = requests.get(api_url, params={"ref": branch})
    resp.raise_for_status()
    items = resp.json()
    csv_candidates = {}
    for item in items:
        name = item.get("name", "")
        if not name.lower().endswith(".csv"):
            continue
        m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
        if m:
            csv_candidates[m.group(1)] = name

    if not csv_candidates:
        raise FileNotFoundError("No CSV files with date pattern found in repository root.")

    latest_date = max(csv_candidates.keys())
    filename = csv_candidates[latest_date]
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filename}"
    return raw_url, filename

# ---------------------------
# READ CSV
# ---------------------------
raw_url, found_filename = get_latest_csv_raw_url(REPO_OWNER, REPO_NAME, BRANCH)
print(f"Using CSV: {found_filename} -> {raw_url}")
df_raw = pd.read_csv(raw_url, encoding='utf-8')

# ---------------------------
# CLEAN & PREP
# ---------------------------
expected_cols = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Percent Change', 'Volume']
missing = set(expected_cols) - set(df_raw.columns)
if missing:
    raise ValueError(f"CSV missing expected columns: {missing}")

df = df_raw[expected_cols].copy()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close'])

# Remove duplicate rows for same symbol+date, keep the last (most recent) row for that date if present
# (We assume 'latest entry' for the date is the one that should be used.)
df = df.sort_values(['Symbol', 'Date', 'Close'], ascending=[True, True, True])
df = df.drop_duplicates(subset=['Symbol', 'Date'], keep='last')  # keep last occurrence for the date

# For time series we want oldest -> newest order per symbol
df = df.sort_values(['Symbol', 'Date'], ascending=[True, True]).reset_index(drop=True)

# ---------------------------
# INDICATORS: RSI(10) using Wilder smoothing, then EMA(14) on RSI -> RSI_EMA_14
# ---------------------------
def compute_rsi(series: pd.Series, length: int = 10) -> pd.Series:
    """
    Compute RSI using Wilder's smoothing (the commonly used method).
    Returns a Series aligned with input series (same index).
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder smoothing: exponential moving average with alpha = 1/length and adjust=False
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)  # if avg_loss == 0, RSI = 100 (no losses)
    rsi = rsi.where(avg_gain != 0, 0)    # if avg_gain == 0 and avg_loss > 0, RSI = 0
    return rsi

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

results = []

symbols = df['Symbol'].unique()
for symbol in symbols:
    sub = df[df['Symbol'] == symbol].copy().reset_index(drop=True)
    # ensure ordered oldest -> newest
    sub = sub.sort_values('Date', ascending=True).reset_index(drop=True)

    # must have enough data to compute RSI(10) and RSI-EMA(14)
    if len(sub) < 10 + 14:  # conservative minimum
        # skip if insufficient history
        continue

    close = sub['Close'].astype(float)
    rsi10 = compute_rsi(close, length=10)
    rsi_ema14 = compute_ema(rsi10, span=14)

    # attach to dataframe for safe iloc selection
    sub = sub.assign(RSI_10=rsi10.values, RSI_EMA_14=rsi_ema14.values)

    # iterate from newest to oldest and pick the first row where RSI_EMA_14 < 30
    # (this ensures we get the most recent date where the smoothed RSI is < 30)
    mask = sub['RSI_EMA_14'] < 30
    if mask.any():
        # find last True (most recent)
        idx = mask[mask].index[-1]
        row = sub.iloc[idx]
        results.append({
            'Symbol': symbol,
            'Date': pd.to_datetime(row['Date']),
            'Signal': 'Buy',
            'Close': float(row['Close']),
            'RSI_EMA_14': round(float(row['RSI_EMA_14']), 2)
        })

signals_df = pd.DataFrame(results, columns=['Symbol', 'Date', 'Signal', 'Close', 'RSI_EMA_14'])

# ---------------------------
# FILTER last KEEP_DAYS days
# ---------------------------
if not signals_df.empty:
    cutoff = datetime.today() - timedelta(days=KEEP_DAYS)
    signals_df['Date'] = pd.to_datetime(signals_df['Date'])
    signals_df = signals_df[signals_df['Date'] >= cutoff].sort_values('Date', ascending=False).reset_index(drop=True)

# ---------------------------
# OUTPUT: save locally or upload to GitHub when GH_TOKEN present
# ---------------------------
out_csv = signals_df.to_csv(index=False)
out_filename = f'RSI_LT_30_LATEST_{datetime.today().strftime("%Y-%m-%d")}.csv'
token = os.environ.get("GH_TOKEN")

if not token:
    print("GH_TOKEN not set: saving file locally.")
    with open(out_filename, "w", encoding="utf-8") as f:
        f.write(out_csv)
    print("Saved:", out_filename)
else:
    # upload via GitHub API (create or update)
    repo = f"{REPO_OWNER}/{REPO_NAME}"
    contents_url = f"https://api.github.com/repos/{repo}/contents/{out_filename}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    # check if file exists to get sha
    r = requests.get(contents_url, headers=headers)
    payload = {
        "message": f"Upload {out_filename}",
        "content": base64.b64encode(out_csv.encode()).decode(),
        "branch": BRANCH
    }
    if r.status_code == 200:
        sha = r.json().get("sha")
        payload["sha"] = sha
    put = requests.put(contents_url, headers=headers, json=payload)
    if put.status_code in (200, 201):
        print("✅ Uploaded", out_filename)
    else:
        print("❌ Upload failed:", put.status_code, put.text)
        # fallback save
        with open(out_filename, "w", encoding="utf-8") as f:
            f.write(out_csv)
        print("Saved locally as fallback:", out_filename)

print("Done. Signals found:", len(signals_df))
if not signals_df.empty:
    print(signals_df.head(50).to_string(index=False))
