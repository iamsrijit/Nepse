# -*- coding: utf-8 -*-
import os
import re
import base64
import requests
import pandas as pd
from collections import deque
from io import StringIO

# ===========================
# CONFIG
# ===========================
REPO_OWNER = "iamsrijit"
REPO_NAME = "Nepse"
BRANCH = "main"
PORTFOLIO_FILE = "portfolio_trades.csv"

# Symbols to exclude from 52-week low analysis
EXCLUDED_SYMBOLS = [
    "EBLD852",
    "EB89",
    "NABILD2089",
    "MBLD2085",
    "SBID89",
    "SBID2090",
    "SBLD2091",
    "NIMBD90",
    "RBBD2088",
    "CCBD88",
    "ICFCD88",
    "EBLD91",
    "GBILD84/85",
    "GBILD86/87",
    "NICD88",
    "CMF2",
"GBIMESY2",
"GIBF1",
"GSY",
"H8020",
"HLICF",
"KDBY",
"KEF",
"KSY",
"LUK",
"LVF2",
"MBLEF",
"MMF1",
"MNMF1",
"NBF2",
"NBF3",
"NIBLGF",
"NIBLSTF",
"NIBSF2",
"NICBF",
"NICFC",
"NICGF2",
"NICSF",
"NMB50",
"NMBHF2",
"NSIF2",
"PRSF",
"PSF",
"RMF1",
"RMF2",
"RSY",
"SAGF",
"SBCF",
"SEF",
"SFEF",
"SIGS2",
"SIGS3",
"SLCF",
    "HEIP",
"HIDCLP",
"NIMBPO",
"NLICLP",
"RBCLPO",
"C30MF",
"CCBD88",
"CMF2",
"EBLD85",
"EBLD91",
"EBLEB89",
"ENL",
"GBBD85",
"GBIMESY2",
"GIBF1",
"GSY",
"GWFD83",
"H8020",
"HATHY",
"HIDCL",
"HIDCLP",
"HLICF",
"JBBD87",
"KDBY",
"KEF",
"KSY",
"LUK",
"LVF2",
"MBLD2085",
"MBLEF",
"MMF1",
"MNMF1",
"NABILD2089",
"NBF2",
"NBF3",
"NBLD87",
"NIBD2082",
"NIBLGF",
"NIBLSTF",
"NIBSF2",
"NICAD2091",
    "PCBLP",
"CZBILP",
"HBLD83",
"NLICP",
"KBLPO",
"JBLBP",
"KMCDB",
"ICFCD83",
"ADBLD83",
"GILB",
"GWFD83",
    "NIBD2082",
"RBBD83",
"SRBLD83",
"SBID83",
"RBBF40",
"PBLD84",
"GBBD85",
"HBLD86",
"SAND2085",
"PBLD86",
"NICAD2091",
"CIZBD86",
"EBLD85",
"NMBD87/88",
"PBD84",
"NICAD85/86",
"SBD87",
"NBBD2085",
"NBLD82",
"NIBD84",
"BOKD86KA",
"NCCD86",
"EBLEB89",
"SBIBD86",
"KSBBLD87",
"MLBLD89",
"NIFRAGED",
"BOKD86",
"PBLD87",
"NMBD2085",
"NBLD87",
"MBLD87",
"JBBD87",
"RBCLPO",
"SDBD87",
"NMBMF",
"SBD89",
"PBD88",
"CBLD88",
"KBLD89",
"NMBD89/90",
"LBBLD89",
"C30MF",
"NABILD87",
"CIZBD90",
"LBLD88",
"SBLD89",
"KBLD86",
"MLBLPO",
"KBLD90",
"PROFLP"



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
    print("✅ Uploaded/Overwritten:", filename)

def delete_old_files(prefix, keep_filename):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents"
    r = requests.get(url, headers=HEADERS, params={"ref": BRANCH})
    r.raise_for_status()
    for f in r.json():
        name = f["name"]
        if name.startswith(prefix) and name.endswith(".csv") and name != keep_filename:
            del_payload = {
                "message": f"Delete old file {name}",
                "sha": f["sha"],
                "branch": BRANCH
            }
            del_url = f"{url}/{name}"
            res = requests.delete(del_url, headers=HEADERS, json=del_payload)
            if res.status_code == 200:
                print(f"🗑️ Deleted: {name}")
            else:
                print(f"⚠️ Failed to delete {name}: {res.text}")

# ===========================
# GET LATEST ESPEN CSV
# ===========================
def get_latest_espen_csv():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents"
    r = requests.get(url, headers=HEADERS, params={"ref": BRANCH})
    r.raise_for_status()
    
    # Find all espen_*.csv files
    espen_files = {}
    for f in r.json():
        name = f["name"]
        if name.startswith("espen_") and name.endswith(".csv"):
            m = re.search(r"espen_(\d{4}-\d{2}-\d{2})\.csv", name)
            if m:
                espen_files[m.group(1)] = name
    
    if not espen_files:
        raise FileNotFoundError("No espen_*.csv file found")
    
    # Get the latest date
    latest_date = max(espen_files.keys())
    latest_file = espen_files[latest_date]
    
    print(f"📂 Using market data file: {latest_file}")
    return github_raw(latest_file)

# ===========================
# LOAD MARKET DATA
# ===========================
# Read the CSV content
csv_url = get_latest_espen_csv()
response = requests.get(csv_url)
csv_content = response.text

# Find the header line
lines = csv_content.strip().split('\n')
header_line = None
data_start_index = 0

for i, line in enumerate(lines):
    if 'Symbol' in line and 'Date' in line and 'Close' in line:
        header_line = line
        data_start_index = i
        break

if header_line is None:
    raise ValueError("Could not find header line with Symbol, Date, Close columns")

# Reconstruct CSV with header at top
if data_start_index > 0:
    # Header is not at the top, move it
    data_lines = lines[:data_start_index]
    reconstructed_csv = header_line + '\n' + '\n'.join(data_lines)
else:
    # Header is already at top
    reconstructed_csv = csv_content

# Parse the CSV - try tab separator first, then comma
try:
    df = pd.read_csv(StringIO(reconstructed_csv), sep='\t')
    # Check if parsing worked (multiple columns)
    if len(df.columns) == 1:
        # Only one column, try comma separator
        df = pd.read_csv(StringIO(reconstructed_csv), sep=',')
except:
    # Fallback to comma separator
    df = pd.read_csv(StringIO(reconstructed_csv), sep=',')

# CRITICAL FIX: Clean column names IMMEDIATELY after loading
df.columns = df.columns.str.strip()

# Debug: Print actual column names
print(f"📊 Loaded {len(df)} rows")
print(f"📋 Column names: {list(df.columns)}")
print(f"📋 First column name repr: {repr(df.columns[0])}")

# Now we can safely access the columns
# Verify 'Date' column exists
if 'Date' not in df.columns:
    print("❌ ERROR: 'Date' column not found!")
    print(f"Available columns: {list(df.columns)}")
    print(f"Column name representations: {[repr(c) for c in df.columns]}")
    raise KeyError("'Date' column not found in DataFrame")

# Show sample dates for debugging
print(f"📅 Sample dates from CSV: {df['Date'].head(3).tolist()}")

# Store original date column in case we need to retry
original_dates = df["Date"].copy()

# Try multiple date formats
df["Date"] = pd.to_datetime(df["Date"], format='%m/%d/%Y', errors='coerce')

# If that didn't work, try other common formats
if df["Date"].isna().all():
    print("⚠️ Format %m/%d/%Y failed, trying %Y-%m-%d...")
    df["Date"] = pd.to_datetime(original_dates, format='%Y-%m-%d', errors='coerce')

if df["Date"].isna().all():
    print("⚠️ Format %Y-%m-%d failed, trying pandas auto-detection...")
    df["Date"] = pd.to_datetime(original_dates, errors='coerce')

# Check if parsing succeeded
parsed_count = df["Date"].notna().sum()
print(f"✅ Successfully parsed {parsed_count} out of {len(df)} dates")

if parsed_count == 0:
    print("❌ ERROR: All dates failed to parse!")
    print(f"Sample raw date values: {original_dates.head(10).tolist()}")
    raise ValueError("Unable to parse any dates from the Date column")

df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
df = df.dropna(subset=["Symbol", "Date", "Close"])
df = df.sort_values(["Symbol", "Date"])

latest_market_date = df["Date"].max().strftime("%Y-%m-%d")
latest_close_map = df.groupby("Symbol")["Close"].last().to_dict()

print(f"📅 Latest market date: {latest_market_date}")
print(f"📈 Total symbols: {df['Symbol'].nunique()}")

# ===========================
# EMA CROSSOVER ANALYSIS (20 vs 50 on Weekly Data)
# ===========================
print("\n" + "="*50)
print("📊 CALCULATING EMA CROSSOVERS ON WEEKLY DATA")
print("="*50)

ema_crossover_results = []

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

# Process each symbol
for sym in df['Symbol'].unique():
    if sym in EXCLUDED_SYMBOLS:
        continue
    
    # Get daily data for this symbol
    sym_data = df[df['Symbol'] == sym].copy()
    
    # Need at least 50 weeks * 5 trading days = 250 data points for reliable 50 EMA
    if len(sym_data) < 250:
        continue
    
    # Convert daily to weekly data
    # Set Date as index for resampling
    sym_data = sym_data.set_index('Date')
    
    # Resample to weekly data (W = week ending on Sunday)
    # We use 'W-THU' to align with Sunday-Thursday trading week
    weekly_data = sym_data.resample('W-THU').agg({
        'Close': 'last',  # Last close price of the week (Thursday)
        'Symbol': 'first'
    }).dropna()
    
    # Need at least 50 weeks of data for 50 EMA
    if len(weekly_data) < 50:
        continue
    
    # Calculate 20 and 50 period EMAs on weekly data
    weekly_data['EMA_20'] = calculate_ema(weekly_data['Close'], 20)
    weekly_data['EMA_50'] = calculate_ema(weekly_data['Close'], 50)
    
    # Drop NaN values from EMA calculation
    weekly_data = weekly_data.dropna()
    
    if len(weekly_data) < 2:
        continue
    
    # Filter to only check data from 2025 onwards
    cutoff_date = pd.Timestamp('2025-01-01')
    weekly_data_2025 = weekly_data[weekly_data.index >= cutoff_date].copy()
    
    if len(weekly_data_2025) < 2:
        continue
    
    # Detect all crossovers from 2025 onwards
    for i in range(1, len(weekly_data_2025)):
        current_week = weekly_data_2025.iloc[i]
        previous_week = weekly_data_2025.iloc[i-1]
        
        current_close = current_week['Close']
        current_ema_20 = current_week['EMA_20']
        current_ema_50 = current_week['EMA_50']
        
        prev_ema_20 = previous_week['EMA_20']
        prev_ema_50 = previous_week['EMA_50']
        prev_close = previous_week['Close']
        
        crossover_signal = None
        
        # Bullish crossover: EMA 20 crosses above EMA 50
        if prev_ema_20 <= prev_ema_50 and current_ema_20 > current_ema_50:
            crossover_signal = "BULLISH_CROSS"
        
        # Bearish crossover: EMA 20 crosses below EMA 50
        elif prev_ema_20 >= prev_ema_50 and current_ema_20 < current_ema_50:
            crossover_signal = "BEARISH_CROSS"
        
        # Only add to results if there's a crossover (BUY or SELL signal)
        if crossover_signal in ["BULLISH_CROSS", "BEARISH_CROSS"]:
            # Convert signal to Buy/Sell
            signal = "BUY" if crossover_signal == "BULLISH_CROSS" else "SELL"
            
            # Get the date of this week
            crossover_date = weekly_data_2025.index[i].strftime("%Y-%m-%d")
            
            # Get crossover week close price
            crossover_close = current_close
            
            # Get latest close price from the most recent week
            latest_close = weekly_data.iloc[-1]['Close']
            
            # Calculate P/L percentage based on signal type
            if signal == "BUY":
                # For BUY signal, profit if price went up
                pl_pct = ((latest_close - crossover_close) / crossover_close) * 100
            else:
                # For SELL signal, profit if price went down (short position)
                pl_pct = ((crossover_close - latest_close) / crossover_close) * 100
            
            ema_crossover_results.append({
                "Symbol": sym,
                "Date": crossover_date,
                "Signal": signal,
                "Close_at_Crossover": round(crossover_close, 2),
                "Latest_Close": round(latest_close, 2),
                "P/L_%": round(pl_pct, 2)
            })

print(f"✅ Analyzed EMA crossovers - found {len(ema_crossover_results)} signals")

# Create DataFrame and sort
if ema_crossover_results:
    ema_df = pd.DataFrame(ema_crossover_results)
    
    # Sort by Date (most recent first), then by Signal (BUY first)
    ema_df = ema_df.sort_values(['Date', 'Signal'], ascending=[False, True]).reset_index(drop=True)
    
    # Print summary
    buy_signals = len(ema_df[ema_df['Signal'] == 'BUY'])
    sell_signals = len(ema_df[ema_df['Signal'] == 'SELL'])
    
    print(f"  🟢 BUY signals: {buy_signals}")
    print(f"  🔴 SELL signals: {sell_signals}")
else:
    ema_df = pd.DataFrame(columns=[
        "Symbol", "Date", "Signal", "Close_at_Crossover", "Latest_Close", "P/L_%"
    ])
    print("  ℹ️ No crossovers detected this week")

# Upload EMA crossover CSV
ema_file = f"EMA_CROSSOVER_20_50_WEEKLY_{latest_market_date}.csv"
upload_to_github(ema_file, ema_df.to_csv(index=False))
delete_old_files("EMA_CROSSOVER_20_50_WEEKLY_", ema_file)

# ===========================
# 52-WEEK LOW & HIGH ANALYSIS
# ===========================
signals_threshold = []  # Stocks within 1.5% of 52-week low
all_distances = []  # All stocks with distance from low and high

# Print excluded symbols if any
if EXCLUDED_SYMBOLS:
    print(f"⚠️ Excluding {len(EXCLUDED_SYMBOLS)} symbols")

# Check if 52High and 52Low columns exist in the CSV
has_52w_columns = '52High' in df.columns and '52Low' in df.columns

if not has_52w_columns:
    print("⚠️ Warning: 52High and 52Low columns not found in CSV. Columns available:", list(df.columns))
    print("ℹ️ The CSV should have columns: Symbol, Date, Open, High, Low, Close, Percent Change, Volume, 52High, 52Low")

symbols_with_52w_data = []
symbols_without_52w_data = []

# Get the latest data for each symbol (most recent date)
latest_data = df.sort_values('Date').groupby('Symbol').last().reset_index()

for _, row in latest_data.iterrows():
    sym = row['Symbol']
    
    # Skip excluded symbols
    if sym in EXCLUDED_SYMBOLS:
        continue
    
    # Get 52-week high and low from the CSV columns
    if has_52w_columns:
        high_52w = pd.to_numeric(row['52High'], errors='coerce')
        low_52w = pd.to_numeric(row['52Low'], errors='coerce')
    else:
        # Fallback: calculate from historical data if columns don't exist
        s = df[df["Symbol"] == sym].copy()
        one_year_ago = df["Date"].max() - pd.Timedelta(days=365)
        s_52w = s[s["Date"] >= one_year_ago]
        
        if len(s_52w) < 10:
            symbols_without_52w_data.append(sym)
            continue
        
        high_52w = s_52w["Close"].max()
        low_52w = s_52w["Close"].min()
    
    # Check if we have valid 52-week data
    if pd.isna(high_52w) or pd.isna(low_52w) or high_52w == 0 or low_52w == 0:
        symbols_without_52w_data.append(sym)
        continue
    
    symbols_with_52w_data.append(sym)
    
    # Use latest_close_map to get the close price (same logic as portfolio)
    latest_close = latest_close_map.get(sym, 0)
    
    if latest_close == 0:
        continue
    
    # Calculate distances
    distance_from_low_pct = ((latest_close - low_52w) / low_52w) * 100
    distance_from_high_pct = ((latest_close - high_52w) / high_52w) * 100
    
    # Add to all_distances (for CSV 2) - NOW WITH URL
    all_distances.append({
        "Symbol": sym,
        "Latest_Close": round(latest_close, 2),
        "52_Week_Low": round(low_52w, 2),
        "52_Week_High": round(high_52w, 2),
        "Distance_from_Low_%": round(distance_from_low_pct, 2),
        "Distance_from_High_%": round(distance_from_high_pct, 2),
        "Chart_URL": f"https://nepsealpha.com/nepse-chart?symbol={sym}"
    })
    
    # Check if latest close is within 1.5% of 52-week low (for CSV 1)
    threshold = low_52w * 1.015  # 1.5% above 52-week low
    
    if latest_close <= threshold:
        signals_threshold.append({
            "Symbol": sym,
            "Latest_Close": round(latest_close, 2),
            "52_Week_Low": round(low_52w, 2),
            "Distance_from_Low_%": round(distance_from_low_pct, 2),
            "Date_at_52W_Low": "N/A",  # Date not available when using CSV columns
            "Chart_URL": f"https://nepsealpha.com/nepse-chart?symbol={sym}"
        })

# Print statistics
print(f"📊 Symbols with 52-week data: {len(symbols_with_52w_data)}")
print(f"📊 Symbols without 52-week data: {len(symbols_without_52w_data)}")
print(f"✅ Found {len(signals_threshold)} stocks within 1.5% of 52-week low")
print(f"✅ Found {len(all_distances)} total stocks with 52-week data")

# ===========================
# CSV 1: Stocks within 1.5% threshold
# ===========================
if signals_threshold:
    signals_df = (
        pd.DataFrame(signals_threshold)
        .sort_values("Distance_from_Low_%")
        .reset_index(drop=True)
    )
else:
    signals_df = pd.DataFrame(columns=[
        "Symbol", 
        "Latest_Close",
        "52_Week_Low", 
        "Distance_from_Low_%",
        "Date_at_52W_Low",
        "Chart_URL"
    ])

low_file = f"52_WEEK_LOW_LATEST_{latest_market_date}.csv"
upload_to_github(low_file, signals_df.to_csv(index=False))
delete_old_files("52_WEEK_LOW_LATEST_", low_file)

# ===========================
# CSV 2: All stocks with distance from low and high
# ===========================
if all_distances:
    distance_df = (
        pd.DataFrame(all_distances)
        .sort_values("Distance_from_Low_%")
        .reset_index(drop=True)
    )
else:
    distance_df = pd.DataFrame(columns=[
        "Symbol",
        "Latest_Close",
        "52_Week_Low",
        "52_Week_High",
        "Distance_from_Low_%",
        "Distance_from_High_%",
        "Chart_URL"
    ])

distance_file = f"52_WEEK_DISTANCE_{latest_market_date}.csv"
upload_to_github(distance_file, distance_df.to_csv(index=False))
delete_old_files("52_WEEK_DISTANCE_", distance_file)

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

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

multi_factor_results = []

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
    position_in_range = None
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
    
    multi_factor_results.append({
        "Symbol": sym,
        "Score": score,
        "Signal": signal,
        "Close": round(latest['Close'], 2),
        "RSI": round(latest['RSI'], 1) if not pd.isna(latest['RSI']) else None,
        "Trend": "UP" if latest['EMA_9'] > latest['EMA_21'] else "DOWN",
        "52W_Position_%": round(position_in_range * 100, 1) if position_in_range is not None else None,
        "Volume_Ratio": round(latest['Volume_Ratio'], 2) if 'Volume_Ratio' in sym_data.columns and not pd.isna(latest['Volume_Ratio']) else None,
        "Momentum_5D_%": round(latest['Momentum_5'], 1) if not pd.isna(latest['Momentum_5']) else None,
        "Factors": ", ".join(signals) if signals else "None"
    })

# Create DataFrame
multi_factor_df = pd.DataFrame(multi_factor_results).sort_values('Score', ascending=False).reset_index(drop=True)

print(f"\n✅ Analyzed {len(multi_factor_df)} symbols")
print(f"   🟢 STRONG_BUY: {len(multi_factor_df[multi_factor_df['Signal'] == 'STRONG_BUY'])}")
print(f"   🟡 BUY: {len(multi_factor_df[multi_factor_df['Signal'] == 'BUY'])}")
print(f"   ⚪ WATCH: {len(multi_factor_df[multi_factor_df['Signal'] == 'WATCH'])}")
print(f"   🔴 AVOID: {len(multi_factor_df[multi_factor_df['Signal'] == 'AVOID'])}")

# Show top opportunities
if len(multi_factor_df) > 0:
    print("\n🏆 TOP 10 OPPORTUNITIES:")
    top_10 = multi_factor_df.head(10)[['Symbol', 'Score', 'Signal', 'Close', 'RSI', 'Trend']]
    print(top_10.to_string(index=False))

# Upload multi-factor CSV
multi_factor_file = f"MULTI_FACTOR_SIGNALS_{latest_market_date}.csv"
upload_to_github(multi_factor_file, multi_factor_df.to_csv(index=False))
delete_old_files("MULTI_FACTOR_SIGNALS_", multi_factor_file)

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

    portfolio_rows.append({
        "Symbol": symbol,
        "Open_Qty": open_qty,
        "Avg_Cost": round(invested / open_qty, 2) if open_qty else 0,
        "Latest_Close": round(last_close, 2),
        "Realized_PnL": round(realized, 2),
        "Unrealized_PnL": round(unrealized, 2),
        "Total_PnL": round(total_pl, 2),
        "Total_PnL_%": round(pl_pct, 2),
    })

portfolio_df = (
    pd.DataFrame(portfolio_rows)
    .sort_values("Total_PnL", ascending=False)
    .reset_index(drop=True)
)

portfolio_file = f"PORTFOLIO_REPORT_{latest_market_date}.csv"
upload_to_github(portfolio_file, portfolio_df.to_csv(index=False))
delete_old_files("PORTFOLIO_REPORT_", portfolio_file)

print("\n" + "="*50)
print("✅ DONE — All reports generated and uploaded")
print("="*50)
print("📁 Files created:")
print(f"  • {ema_file}")
print(f"  • {low_file}")
print(f"  • {distance_file}")
print(f"  • {multi_factor_file}")
print(f"  • {portfolio_file}")
