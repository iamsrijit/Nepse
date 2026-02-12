# -*- coding: utf-8 -*-
"""
MULTI-FACTOR SCORING STRATEGY
Combines multiple proven factors to identify high-probability trades
"""
import os
import re
import base64
import requests
import pandas as pd
import numpy as np
from collections import deque
from io import StringIO

# ===========================
# CONFIG
# ===========================
REPO_OWNER = "iamsrijit"
REPO_NAME = "Nepse"
BRANCH = "main"
PORTFOLIO_FILE = "portfolio_trades.csv"

EXCLUDED_SYMBOLS = [
    "EBLD852", "EBL", "EB89", "NABILD2089", "MBLD2085", "SBID89",
    "SBID2090", "SBLD2091", "NIMBD90", "RBBD2088", "CCBD88", "ULBSL",
    "ICFCD88", "EBLD91", "ANLB", "GBILD84/85", "GBILD86/87", "NICD88"
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
    print("✅ Uploaded:", filename)

def delete_old_files(prefix, keep_filename):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents"
    r = requests.get(url, headers=HEADERS, params={"ref": BRANCH})
    r.raise_for_status()
    for f in r.json():
        name = f["name"]
        if name.startswith(prefix) and name.endswith(".csv") and name != keep_filename:
            del_payload = {"message": f"Delete old file {name}", "sha": f["sha"], "branch": BRANCH}
            del_url = f"{url}/{name}"
            res = requests.delete(del_url, headers=HEADERS, json=del_payload)
            if res.status_code == 200:
                print(f"🗑️ Deleted: {name}")

def get_latest_espen_csv():
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
        raise FileNotFoundError("No espen_*.csv file found")
    latest_date = max(espen_files.keys())
    latest_file = espen_files[latest_date]
    print(f"📂 Using: {latest_file}")
    return github_raw(latest_file)

# ===========================
# LOAD & PREPARE DATA
# ===========================
csv_url = get_latest_espen_csv()
response = requests.get(csv_url)
csv_content = response.text

lines = csv_content.strip().split('\n')
header_line = None
data_start_index = 0

for i, line in enumerate(lines):
    if 'Symbol' in line and 'Date' in line and 'Close' in line:
        header_line = line
        data_start_index = i
        break

if header_line is None:
    raise ValueError("Header not found")

if data_start_index > 0:
    reconstructed_csv = header_line + '\n' + '\n'.join(lines[:data_start_index])
else:
    reconstructed_csv = csv_content

try:
    df = pd.read_csv(StringIO(reconstructed_csv), sep='\t')
    if len(df.columns) == 1:
        df = pd.read_csv(StringIO(reconstructed_csv), sep=',')
except:
    df = pd.read_csv(StringIO(reconstructed_csv), sep=',')

df.columns = df.columns.str.strip()

# Parse dates
original_dates = df["Date"].copy()
df["Date"] = pd.to_datetime(df["Date"], format='%m/%d/%Y', errors='coerce')
if df["Date"].isna().all():
    df["Date"] = pd.to_datetime(original_dates, format='%Y-%m-%d', errors='coerce')
if df["Date"].isna().all():
    df["Date"] = pd.to_datetime(original_dates, errors='coerce')

df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
if 'Volume' in df.columns:
    df["Volume"] = pd.to_numeric(df["Volume"], errors='coerce')
if 'High' in df.columns:
    df["High"] = pd.to_numeric(df["High"], errors='coerce')
if 'Low' in df.columns:
    df["Low"] = pd.to_numeric(df["Low"], errors='coerce')

df = df.dropna(subset=["Symbol", "Date", "Close"])
df = df.sort_values(["Symbol", "Date"])

latest_market_date = df["Date"].max().strftime("%Y-%m-%d")
print(f"📅 Latest date: {latest_market_date}")
print(f"📈 Symbols: {df['Symbol'].nunique()}")

# ===========================
# MULTI-FACTOR SCORING SYSTEM
# ===========================
print("\n" + "="*70)
print("🎯 MULTI-FACTOR SCORING SYSTEM")
print("="*70)

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(series, period):
    """Calculate EMA"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

results = []

for sym in df['Symbol'].unique():
    if sym in EXCLUDED_SYMBOLS:
        continue
    
    sym_data = df[df['Symbol'] == sym].copy().sort_values('Date')
    
    # Need sufficient data
    if len(sym_data) < 200:
        continue
    
    # Calculate indicators
    sym_data['RSI'] = calculate_rsi(sym_data['Close'], 14)
    sym_data['EMA_9'] = calculate_ema(sym_data['Close'], 9)
    sym_data['EMA_21'] = calculate_ema(sym_data['Close'], 21)
    sym_data['EMA_50'] = calculate_ema(sym_data['Close'], 50)
    sym_data['SMA_200'] = sym_data['Close'].rolling(window=200).mean()
    
    # Bollinger Bands
    sym_data['BB_Upper'], sym_data['BB_Middle'], sym_data['BB_Lower'] = calculate_bollinger_bands(sym_data['Close'])
    
    # Volume analysis
    if 'Volume' in sym_data.columns:
        sym_data['Volume_SMA'] = sym_data['Volume'].rolling(window=20).mean()
        sym_data['Volume_Ratio'] = sym_data['Volume'] / sym_data['Volume_SMA']
    
    # 52-week high/low
    sym_data['52W_High'] = sym_data['Close'].rolling(window=252, min_periods=50).max()
    sym_data['52W_Low'] = sym_data['Close'].rolling(window=252, min_periods=50).min()
    
    # Price momentum
    sym_data['Momentum_5'] = sym_data['Close'].pct_change(5) * 100
    sym_data['Momentum_20'] = sym_data['Close'].pct_change(20) * 100
    
    # Get latest values
    latest = sym_data.iloc[-1]
    
    if pd.isna(latest['RSI']) or pd.isna(latest['EMA_50']):
        continue
    
    # ===========================
    # SCORING SYSTEM (0-100)
    # ===========================
    score = 0
    signals = []
    
    # FACTOR 1: Mean Reversion (RSI) - 15 points
    if latest['RSI'] < 30:
        score += 15
        signals.append("Oversold_RSI")
    elif latest['RSI'] < 40:
        score += 10
        signals.append("Low_RSI")
    elif latest['RSI'] > 70:
        score -= 15
        signals.append("Overbought_RSI")
    
    # FACTOR 2: Trend Following (EMAs) - 20 points
    if latest['EMA_9'] > latest['EMA_21'] > latest['EMA_50']:
        score += 20
        signals.append("Strong_Uptrend")
    elif latest['EMA_9'] > latest['EMA_21']:
        score += 10
        signals.append("Weak_Uptrend")
    elif latest['EMA_9'] < latest['EMA_21'] < latest['EMA_50']:
        score -= 20
        signals.append("Strong_Downtrend")
    
    # FACTOR 3: Long-term trend (200 SMA) - 15 points
    if not pd.isna(latest['SMA_200']):
        if latest['Close'] > latest['SMA_200']:
            score += 15
            signals.append("Above_200SMA")
        else:
            score -= 10
            signals.append("Below_200SMA")
    
    # FACTOR 4: 52-week position - 15 points
    if not pd.isna(latest['52W_Low']) and not pd.isna(latest['52W_High']):
        range_52w = latest['52W_High'] - latest['52W_Low']
        if range_52w > 0:
            position_in_range = (latest['Close'] - latest['52W_Low']) / range_52w
            if position_in_range < 0.15:  # Near 52-week low
                score += 15
                signals.append("Near_52W_Low")
            elif position_in_range < 0.30:
                score += 10
                signals.append("Lower_Half_Range")
            elif position_in_range > 0.85:  # Near 52-week high
                score -= 10
                signals.append("Near_52W_High")
    
    # FACTOR 5: Bollinger Band position - 10 points
    if not pd.isna(latest['BB_Lower']) and not pd.isna(latest['BB_Upper']):
        bb_range = latest['BB_Upper'] - latest['BB_Lower']
        if bb_range > 0:
            if latest['Close'] < latest['BB_Lower']:
                score += 10
                signals.append("Below_BB")
            elif latest['Close'] < latest['BB_Middle']:
                score += 5
                signals.append("Below_BB_Middle")
    
    # FACTOR 6: Volume surge - 10 points
    if 'Volume_Ratio' in sym_data.columns and not pd.isna(latest['Volume_Ratio']):
        if latest['Volume_Ratio'] > 1.5:
            score += 10
            signals.append("High_Volume")
        elif latest['Volume_Ratio'] > 1.2:
            score += 5
            signals.append("Above_Avg_Volume")
    
    # FACTOR 7: Recent momentum - 15 points
    if not pd.isna(latest['Momentum_5']):
        if latest['Momentum_5'] > 5:
            score += 5
            signals.append("Positive_5D_Momentum")
        elif latest['Momentum_5'] < -5:
            score += 10  # Reversal play
            signals.append("Negative_5D_Momentum_Reversal")
    
    if not pd.isna(latest['Momentum_20']):
        if latest['Momentum_20'] > 10:
            score += 10
            signals.append("Strong_20D_Momentum")
        elif latest['Momentum_20'] < -15:
            score += 5  # Deep pullback in uptrend
            signals.append("Deep_Pullback")
    
    # Generate signal
    if score >= 70:
        signal = "STRONG_BUY"
    elif score >= 50:
        signal = "BUY"
    elif score >= 30:
        signal = "WATCH"
    elif score <= -20:
        signal = "AVOID"
    else:
        signal = "NEUTRAL"
    
    results.append({
        "Symbol": sym,
        "Score": score,
        "Signal": signal,
        "Close": round(latest['Close'], 2),
        "RSI": round(latest['RSI'], 1),
        "Trend": "UP" if latest['EMA_9'] > latest['EMA_21'] else "DOWN",
        "52W_Position_%": round(position_in_range * 100, 1) if 'position_in_range' in locals() else None,
        "Volume_Ratio": round(latest['Volume_Ratio'], 2) if 'Volume_Ratio' in sym_data.columns else None,
        "Momentum_5D_%": round(latest['Momentum_5'], 1) if not pd.isna(latest['Momentum_5']) else None,
        "Factors": ", ".join(signals) if signals else "None"
    })

# Create DataFrame
results_df = pd.DataFrame(results).sort_values('Score', ascending=False).reset_index(drop=True)

print(f"\n✅ Analyzed {len(results_df)} symbols")
print(f"   🟢 STRONG_BUY: {len(results_df[results_df['Signal'] == 'STRONG_BUY'])}")
print(f"   🟡 BUY: {len(results_df[results_df['Signal'] == 'BUY'])}")
print(f"   ⚪ WATCH: {len(results_df[results_df['Signal'] == 'WATCH'])}")
print(f"   🔴 AVOID: {len(results_df[results_df['Signal'] == 'AVOID'])}")

# Show top opportunities
print("\n🏆 TOP 10 OPPORTUNITIES:")
print(results_df.head(10)[['Symbol', 'Score', 'Signal', 'Close', 'RSI', 'Trend', 'Factors']].to_string(index=False))

# Upload
filename = f"MULTI_FACTOR_SIGNALS_{latest_market_date}.csv"
upload_to_github(filename, results_df.to_csv(index=False))
delete_old_files("MULTI_FACTOR_SIGNALS_", filename)

print("\n" + "="*70)
print(f"✅ DONE - {filename}")
print("="*70)
