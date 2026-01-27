# rsi.py
# Calculates RSI oversold signals from latest espen_*.csv and uploads to GitHub
# Cleans up old RSI_SIGNALS_* files
# -*- coding: utf-8 -*-

import os
import re
import requests
import pandas as pd
import base64
from datetime import timedelta
from io import StringIO

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
REPO_OWNER = "iamsrijit"
REPO_NAME = "Nepse"
BRANCH = "main"

GH_TOKEN = os.environ.get("GH_TOKEN")
if not GH_TOKEN:
    raise RuntimeError("GH_TOKEN environment variable not set")

HEADERS = {"Authorization": f"token {GH_TOKEN}"}

EXCLUDED_SYMBOLS = [
    "EBLD852", "EBL", "EB89", "NABILD2089", "MBLD2085", "SBID89", "SBID2090",
    "SBLD2091", "NIMBD90", "RBBD2088", "CCBD88", "ULBSL", "ICFCD88", "EBLD91",
    "ANLB", "GBILD84/85", "GBILD86/87", "NICD88"
]

# RSI settings
RSI_PERIOD = 14
RSI_OVERSOLD = 30          # only collect oversold (buy signals)
# RSI_OVERBOUGHT = 70      # commented out — add back if you want overbought too

# ────────────────────────────────────────────────
# GITHUB HELPERS
# ────────────────────────────────────────────────
def github_raw(path):
    return f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{path}"

def upload_to_github(filename, content):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{filename}"
    r = requests.get(url, headers=HEADERS)
    payload = {
        "message": f"Upload RSI signals {filename}",
        "content": base64.b64encode(content.encode()).decode(),
        "branch": BRANCH
    }
    if r.status_code == 200:
        payload["sha"] = r.json()["sha"]
    res = requests.put(url, headers=HEADERS, json=payload)
    if res.status_code not in (200, 201):
        print(f"⚠️ Upload failed for {filename}: {res.status_code} - {res.text}")
        return False
    print(f"✅ Uploaded: {filename}")
    return True

def delete_old_files(prefix, keep_filename):
    """Delete all files starting with prefix except the keep_filename"""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents"
    r = requests.get(url, headers=HEADERS, params={"ref": BRANCH})
    if r.status_code != 200:
        print(f"⚠️ Could not list repo contents: {r.status_code}")
        return

    for f in r.json():
        name = f["name"]
        if name.startswith(prefix) and name.endswith(".csv") and name != keep_filename:
            del_payload = {
                "message": f"Delete old RSI file {name}",
                "sha": f["sha"],
                "branch": BRANCH
            }
            del_url = f"{url}/{name}"
            res = requests.delete(del_url, headers=HEADERS, json=del_payload)
            if res.status_code == 200:
                print(f"🗑️ Deleted old: {name}")
            else:
                print(f"⚠️ Delete failed for {name}: {res.status_code} - {res.text}")

# ────────────────────────────────────────────────
# GET LATEST ESPEN CSV
# ────────────────────────────────────────────────
def get_latest_espen_url():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents"
    r = requests.get(url, headers=HEADERS, params={"ref": BRANCH})
    r.raise_for_status()

    espen_files = {}
    for f in r.json():
        name = f["name"]
        if name.startswith("espen_") and name.endswith(".csv"):
            m = re.search(r"espen_(\d{4}-\d{2}-\d{2})\.csv", name)
            if m:
                espen_files[m.group(1)] = name

    if not espen_files:
        raise FileNotFoundError("No espen_*.csv found in repository")

    latest_date_key = max(espen_files.keys())
    latest_file = espen_files[latest_date_key]
    print(f"📂 Using market data: {latest_file}")
    return github_raw(latest_file), latest_date_key

# ─── Load and clean market data ───────────────────────────────────────
csv_url, latest_date_str = get_latest_espen_url()
resp = requests.get(csv_url)
if resp.status_code != 200:
    raise RuntimeError(f"Failed to download CSV: {resp.status_code} - {resp.reason}")

content = resp.text
lines = content.strip().split('\n')

# Find header
header_idx = next((i for i, ln in enumerate(lines) if 'Symbol' in ln and 'Close' in ln), None)
if header_idx is None:
    raise ValueError("No header line found in espen CSV")

header = lines[header_idx]
data_lines = lines[header_idx + 1:]
clean_csv = header + '\n' + '\n'.join(data_lines)

sep = '\t' if '\t' in header else ','
print(f"🔍 Detected separator: {repr(sep)}")

df_market = pd.read_csv(StringIO(clean_csv), sep=sep)
df_market.columns = df_market.columns.str.strip()

df_market['Date'] = pd.to_datetime(df_market['Date'].astype(str).str.strip(), errors='coerce')
df_market['Close'] = pd.to_numeric(df_market['Close'], errors='coerce')
df_market = df_market.dropna(subset=['Symbol', 'Date', 'Close'])
df_market = df_market.sort_values(['Symbol', 'Date'])

print(f"📊 Loaded {len(df_market)} rows | {df_market['Symbol'].nunique()} symbols")
print(f"📅 Latest date: {df_market['Date'].max().date()}")

# ─── RSI Calculation ──────────────────────────────────────────────────
signals = []
for sym in df_market["Symbol"].unique():
    if sym in EXCLUDED_SYMBOLS:
        continue

    s = df_market[df_market["Symbol"] == sym].copy()
    if len(s) < RSI_PERIOD + 20:  # more conservative minimum data requirement
        continue

    delta = s['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=RSI_PERIOD).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=RSI_PERIOD).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    latest_rsi = rsi.iloc[-1]
    if pd.isna(latest_rsi):
        continue

    latest_close = s['Close'].iloc[-1]

    if latest_rsi <= RSI_OVERSOLD:
        signals.append({
            "Symbol": sym,
            "Latest_Close": round(latest_close, 2),
            "RSI": round(latest_rsi, 2),
            "Status": "Oversold (potential buy)"
        })

# ─── Output & Upload ──────────────────────────────────────────────────
if signals:
    df_signals = pd.DataFrame(signals).sort_values("RSI").reset_index(drop=True)
    print("\nRSI Oversold Signals:")
    print(df_signals.to_string(index=False))

    # Prepare filename with market date
    market_date = df_market['Date'].max().strftime("%Y-%m-%d")
    filename = f"RSI_SIGNALS_{market_date}.csv"

    # Upload
    upload_success = upload_to_github(filename, df_signals.to_csv(index=False))

    if upload_success:
        # Clean up old files
        delete_old_files("RSI_SIGNALS_", filename)

else:
    print("No oversold RSI signals found (RSI ≤ 30).")

print("\nDone.")
