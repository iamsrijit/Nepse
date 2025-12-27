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
KEEP_DAYS = 3000  # last 3000 days of signals

# ---------------------------
# HELPER: get latest CSV from repo root
# ---------------------------
def get_latest_csv_raw_url(owner: str, repo: str, branch: str = "main"):
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

df_raw = pd.read_csv(raw_url, encoding="utf-8")

# ---------------------------
# CLEAN & PREP
# ---------------------------
expected_cols = [
    'Symbol', 'Date', 'Open', 'High', 'Low',
    'Close', 'Percent Change', 'Volume'
]

missing = set(expected_cols) - set(df_raw.columns)
if missing:
    raise ValueError(f"CSV missing expected columns: {missing}")

df = df_raw[expected_cols].copy()

df['Date'] = pd.to_datetime(
    df['Date'],
    errors='coerce',
    dayfirst=False,
    infer_datetime_format=True
)

df = df.dropna(subset=['Date'])

df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close'])

# remove duplicates (keep latest close)
df = df.sort_values(['Symbol', 'Date', 'Close'])
df = df.drop_duplicates(subset=['Symbol', 'Date'], keep='last')

# sort oldest -> newest
df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

# ---------------------------
# INDICATORS
# ---------------------------
def compute_rsi(series: pd.Series, length: int = 10) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(avg_gain != 0, 0)

    return rsi

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

# ---------------------------
# COMPUTE SIGNALS
# ---------------------------
results = []

for symbol in df['Symbol'].unique():
    sub = df[df['Symbol'] == symbol].copy().reset_index(drop=True)

    if len(sub) < 24:  # RSI(10) + EMA(14)
        continue

    sub['RSI_10'] = compute_rsi(sub['Close'], 10)
    sub['RSI_EMA_14'] = compute_ema(sub['RSI_10'], 14)

    # latest trading day close
    latest_close = float(sub.iloc[-1]['Close'])
    latest_date = sub.iloc[-1]['Date']

    mask = sub['RSI_EMA_14'] < 30
    if mask.any():
        idx = mask[mask].index[-1]
        row = sub.iloc[idx]

        entry_close = float(row['Close'])
        pnl_pct = ((latest_close - entry_close) / entry_close) * 100

        results.append({
            'Symbol': symbol,
            'Date': pd.to_datetime(row['Date']),
            'Signal': 'Buy',
            'Close': round(entry_close, 2),
            'RSI_EMA_14': round(float(row['RSI_EMA_14']), 2),
            'Latest_Close': round(latest_close, 2),
            'PnL_%': round(pnl_pct, 2)
        })

signals_df = pd.DataFrame(
    results,
    columns=[
        'Symbol',
        'Date',
        'Signal',
        'Close',
        'RSI_EMA_14',
        'Latest_Close',
        'PnL_%'
    ]
)

# ---------------------------
# FILTER last KEEP_DAYS
# ---------------------------
if not signals_df.empty:
    cutoff = datetime.today() - timedelta(days=KEEP_DAYS)
    signals_df = (
        signals_df[signals_df['Date'] >= cutoff]
        .sort_values('Date', ascending=False)
        .reset_index(drop=True)
    )

# ---------------------------
# OUTPUT
# ---------------------------
out_csv = signals_df.to_csv(index=False)
out_filename = f"RSI_LT_30_LATEST_{datetime.today().strftime('%Y-%m-%d')}.csv"

token = os.environ.get("GH_TOKEN")

if not token:
    print("GH_TOKEN not set → saving locally.")
    with open(out_filename, "w", encoding="utf-8") as f:
        f.write(out_csv)
    print("Saved locally:", out_filename)
else:
    repo = f"{REPO_OWNER}/{REPO_NAME}"
    contents_url = f"https://api.github.com/repos/{repo}/contents/{out_filename}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    r = requests.get(contents_url, headers=headers)

    payload = {
        "message": f"Upload {out_filename}",
        "content": base64.b64encode(out_csv.encode()).decode(),
        "branch": BRANCH
    }

    if r.status_code == 200:
        payload["sha"] = r.json().get("sha")

    put = requests.put(contents_url, headers=headers, json=payload)

    if put.status_code in (200, 201):
        print("✅ Uploaded:", out_filename)
    else:
        print("❌ Upload failed:", put.status_code, put.text)
        with open(out_filename, "w", encoding="utf-8") as f:
            f.write(out_csv)
        print("Saved locally as fallback:", out_filename)

# ---------------------------
# DONE
# ---------------------------
print("Done. Signals found:", len(signals_df))
if not signals_df.empty:
    print(signals_df.head(50).to_string(index=False))
