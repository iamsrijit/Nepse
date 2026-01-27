# rsi.py
import os
import re
import requests
import pandas as pd
import base64
from datetime import timedelta
from io import StringIO

# ────────────────────────────────────────────────
# CONFIG ─ same as your main script
REPO_OWNER = "iamsrijit"
REPO_NAME  = "Nepse"
BRANCH     = "main"
GH_TOKEN   = os.environ.get("GH_TOKEN")

if not GH_TOKEN:
    raise RuntimeError("GH_TOKEN environment variable not set")

HEADERS = {"Authorization": f"token {GH_TOKEN}"}

EXCLUDED_SYMBOLS = ["EBLD852", "EBL", ...]  # copy your full list

# RSI settings
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD   = 30
# ────────────────────────────────────────────────

def github_raw(path):
    return f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{path}"

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
        raise FileNotFoundError("No espen_*.csv found")
    
    latest_date = max(espen_files)
    return github_raw(espen_files[latest_date]), latest_date

# ─── Load market data ───────────────────────────────────────
csv_url, latest_date_str = get_latest_espen_url()
resp = requests.get(csv_url)
if resp.status_code != 200:
    raise RuntimeError(f"Failed to download CSV: {resp.status_code}")

content = resp.text
lines = content.strip().split('\n')

# Find header line
header_idx = next((i for i, ln in enumerate(lines) if 'Symbol' in ln and 'Close' in ln), None)
if header_idx is None:
    raise ValueError("No header line found in espen CSV")

header = lines[header_idx]
data = '\n'.join(lines[header_idx+1:])

sep = '\t' if '\t' in header else ','
df_market = pd.read_csv(StringIO(header + '\n' + data), sep=sep)
df_market.columns = df_market.columns.str.strip()

df_market['Date']  = pd.to_datetime(df_market['Date'].astype(str).str.strip(), errors='coerce')
df_market['Close'] = pd.to_numeric(df_market['Close'], errors='coerce')
df_market = df_market.dropna(subset=['Symbol', 'Date', 'Close'])
df_market = df_market.sort_values(['Symbol', 'Date'])

print(f"Loaded {len(df_market)} rows | {df_market['Symbol'].nunique()} symbols")
print(f"Latest date: {df_market['Date'].max().date()}")

# ─── Now your RSI calculation ─────────────────────────────────
signals = []

for sym in df_market["Symbol"].unique():
    if sym in EXCLUDED_SYMBOLS:
        continue
        
    s = df_market[df_market["Symbol"] == sym].copy()
    if len(s) < RSI_PERIOD + 10:   # need enough data
        continue
        
    # Calculate RSI (simple version)
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
    elif latest_rsi >= RSI_OVERBOUGHT:
        signals.append({
            "Symbol": sym,
            "Latest_Close": round(latest_close, 2),
            "RSI": round(latest_rsi, 2),
            "Status": "Overbought (potential sell)"
        })

if signals:
    df_signals = pd.DataFrame(signals).sort_values("RSI")
    print("\nRSI Signals:")
    print(df_signals.to_string(index=False))
    
    # Optional: save to GitHub like your other script
    # upload_to_github(f"RSI_SIGNALS_{latest_date_str}.csv", df_signals.to_csv(index=False))
else:
    print("No strong RSI signals (oversold/overbought) found.")
