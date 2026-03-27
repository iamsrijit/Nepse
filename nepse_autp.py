import subprocess
import sys
import os
import requests
import base64
import re
import pandas as pd
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup

# ── Install required packages ────────────────────────────────────────────────
packages = ["nepse-scraper", "xlsxwriter", "gitpython", "pandas", "matplotlib", "joblib"]
subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages, stdout=subprocess.DEVNULL)

from nepse_scraper import Nepse_scraper
from joblib import Parallel, delayed

# ── Constants ────────────────────────────────────────────────────────────────
STANDARD_COLS   = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close',
                   'Percent Change', 'Volume', '52High', '52Low']
GITHUB_REPO     = 'iamsrijit/Nepse'
GH_TOKEN        = os.getenv("GH_TOKEN")

# ── Helper ───────────────────────────────────────────────────────────────────
def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0

def format_date(dt):
    """Return M/D/YYYY with no leading zeros."""
    return f"{dt.month}/{dt.day}/{dt.year}"

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 – Fetch today's price from NEPSE API
# ════════════════════════════════════════════════════════════════════════════
print("Fetching today's price from NEPSE …")
request_obj = Nepse_scraper(verify_ssl=False)
today_price  = request_obj.get_today_price()

content_data = today_price.get('content', []) if isinstance(today_price, dict) else today_price

filtered_data = []
for item in content_data:
    symbol      = item.get('symbol', '')
    date        = item.get('businessDate', '')
    open_price  = to_float(item.get('openPrice'))
    high_price  = to_float(item.get('highPrice'))
    low_price   = to_float(item.get('lowPrice'))
    close_price = to_float(item.get('closePrice'))
    volume      = to_float(item.get('totalTradedQuantity'))
    high52      = to_float(item.get('fiftyTwoWeekHigh'))
    low52       = to_float(item.get('fiftyTwoWeekLow'))
    pct_chg     = round(((close_price - open_price) / open_price * 100)
                        if open_price > 0 else 0, 2)

    filtered_data.append({
        'Symbol':         symbol,
        'Date':           date,
        'Open':           open_price,
        'High':           high_price,
        'Low':            low_price,
        'Close':          close_price,
        'Percent Change': pct_chg,
        'Volume':         volume,
        '52High':         high52,
        '52Low':          low52,
    })

first = pd.DataFrame(filtered_data)
first['Date'] = pd.to_datetime(first['Date'], errors='coerce')

# Build a lookup: symbol → (52High, 52Low) from today's live data
live_52 = (
    first.dropna(subset=['Date'])
         .sort_values('Date', ascending=False)
         .drop_duplicates('Symbol')
         .set_index('Symbol')[['52High', '52Low']]
)

if not first.empty:
    day_name  = datetime.now().strftime('%A')
    file_name = f"nepse_{day_name}.csv"
    first_out = first.copy()
    first_out['Date'] = first_out['Date'].apply(
        lambda d: format_date(d) if pd.notna(d) else '')
    first_out.to_csv(file_name, index=False, columns=STANDARD_COLS)
    print(f"Today's data saved to '{file_name}'")
else:
    print("No data available from NEPSE API today.")

# ════════════════════════════════════════════════════════════════════════════
# STEP 2 – Read latest historical espen_ file from GitHub
# ════════════════════════════════════════════════════════════════════════════
def get_latest_espen_url(repo_url: str) -> str:
    response = requests.get(repo_url, timeout=15)
    soup     = BeautifulSoup(response.content, 'html.parser')
    file_urls = {}
    for link in soup.find_all('a', href=True):
        href = link['href']
        if 'espen_' in href and href.endswith('.csv'):
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', href)
            if date_match:
                file_date = date_match.group(1)
                raw_url   = (
                    f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/"
                    + href.split('/')[-1]
                )
                file_urls[file_date] = raw_url
    if not file_urls:
        raise ValueError("No espen_ CSV files found in the repository.")
    latest = max(file_urls.keys())
    print(f"Latest espen_ file date: {latest}")
    print(f"URL: {file_urls[latest]}")
    return file_urls[latest]

secondss = pd.DataFrame()
try:
    latest_url = get_latest_espen_url(f'https://github.com/{GITHUB_REPO}/tree/main')
    raw        = pd.read_csv(latest_url)

    # ── FIX 1: keep only the 10 standard columns (ignore any extra columns) ──
    available = [c for c in STANDARD_COLS if c in raw.columns]
    secondss  = raw[available].copy()

    # If any standard column is missing, add it with NaN
    for col in STANDARD_COLS:
        if col not in secondss.columns:
            secondss[col] = np.nan

    secondss = secondss[STANDARD_COLS]
    secondss['Date'] = pd.to_datetime(secondss['Date'], infer_datetime_format=True,
                                       errors='coerce')
    print(f"Historical data loaded: {len(secondss)} rows")
except Exception as e:
    print(f"Could not load historical data: {e}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 3 – Merge today + history, deduplicate, fix 52H/52L
# ════════════════════════════════════════════════════════════════════════════
dfs = [df for df in [first, secondss] if not df.empty]

if not dfs:
    print("No data to process.")
    raise SystemExit(1)

combined_df = pd.concat(dfs, ignore_index=True, join='outer')
combined_df['Date'] = pd.to_datetime(combined_df['Date'], infer_datetime_format=True,
                                      errors='coerce')
combined_df.dropna(subset=['Date'], inplace=True)

finall_df = pd.DataFrame()
for symbol, grp in combined_df.groupby('Symbol'):
    grp = grp.sort_values('Date', ascending=False).drop_duplicates('Date', keep='first')

    # ── FIX 2: overwrite 52High / 52Low with today's live value ─────────────
    if symbol in live_52.index:
        grp['52High'] = live_52.loc[symbol, '52High']
        grp['52Low']  = live_52.loc[symbol, '52Low']

    finall_df = pd.concat([finall_df, grp], ignore_index=True)

# ── FIX 3: format Date as M/D/YYYY ──────────────────────────────────────────
finall_df['Date'] = finall_df['Date'].apply(format_date)

# Ensure column order
finall_df = finall_df[STANDARD_COLS]

output_file = 'combined_data.csv'
finall_df.to_csv(output_file, index=False)
print(f"\nCombined data saved to '{output_file}' ({len(finall_df)} rows)")
print(finall_df.head(10).to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════
# STEP 4 – Upload merged file to GitHub as espen_YYYY-MM-DD.csv
# ════════════════════════════════════════════════════════════════════════════
def github_put(file_name: str, df: pd.DataFrame):
    csv_b64  = base64.b64encode(df.to_csv(index=False).encode()).decode()
    url      = f'https://api.github.com/repos/{GITHUB_REPO}/contents/{file_name}'
    headers  = {'Authorization': f'token {GH_TOKEN}'}

    # Check if file already exists (need its SHA to update)
    r   = requests.get(url, headers=headers, timeout=15)
    sha = r.json().get('sha') if r.status_code == 200 else None

    payload = {'message': f'Add {file_name}', 'content': csv_b64, 'branch': 'main'}
    if sha:
        payload['sha'] = sha

    resp = requests.put(url, headers=headers, json=payload, timeout=30)
    if resp.status_code in (200, 201):
        print(f"Uploaded '{file_name}' to GitHub successfully.")
    else:
        print(f"Upload failed ({resp.status_code}): {resp.text}")

today_str  = datetime.today().strftime('%Y-%m-%d')
github_put(f'espen_{today_str}.csv', finall_df)

# ════════════════════════════════════════════════════════════════════════════
# STEP 5 – EMA / RSI crossover analysis
# ════════════════════════════════════════════════════════════════════════════
data = finall_df.copy()
data['Date']   = pd.to_datetime(data['Date'], infer_datetime_format=True, errors='coerce')
data           = data.sort_values(['Symbol', 'Date']).reset_index(drop=True)
data['Close']  = pd.to_numeric(data['Close'].astype(str).str.replace(',', ''), errors='coerce').astype('float32')
data['Volume'] = pd.to_numeric(
    data['Volume'].astype(str).str.replace(',', '').replace('-', np.nan),
    errors='coerce').astype('float32')

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(period, min_periods=1).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(period, min_periods=1).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def process_symbol(symbol, group):
    group = group.copy()
    group['EMA_20']        = group['Close'].ewm(span=20, adjust=False).mean()
    group['EMA_50']        = group['Close'].ewm(span=50, adjust=False).mean()
    group['RSI']           = calculate_rsi(group['Close'])
    group['30D_Avg_Volume']= group['Volume'].rolling(30, min_periods=1).mean()
    group['Slope_20']      = group['EMA_20'].diff(5) / 5
    group['Slope_50']      = group['EMA_50'].diff(5) / 5
    group['Crossover']     = (
        (group['EMA_20'] > group['EMA_50']) &
        (group['EMA_20'].shift(1) <= group['EMA_50'].shift(1))
    )
    valid = group[
        group['Crossover'] &
        (group['Slope_20'] > group['Slope_50']) &
        (group['RSI'].between(30, 70)) &
        (group['Volume'] >= 0.3 * group['30D_Avg_Volume']) &
        (group['Close'] > group['Close'].rolling(60, min_periods=1).max().shift(1) * 0.95)
    ]
    return valid

print("\nRunning EMA crossover analysis …")
results = Parallel(n_jobs=-1)(
    delayed(process_symbol)(sym, grp)
    for sym, grp in data.groupby('Symbol')
)
all_valid = pd.concat(results).reset_index(drop=True)
all_valid.to_csv('valid_ema_crossovers.csv', index=False)

combinedd_df = all_valid.sort_values('Date', ascending=False).drop_duplicates('Symbol')
combinedd_df.to_csv('latest_valid_ema_crossovers.csv', index=False)
print("\nLatest EMA crossover signals:")
print(combinedd_df[['Symbol', 'Date', 'Close']].to_string(index=False))

# ── Upload EMA file ──────────────────────────────────────────────────────────
github_put(f'EMA_Cross_for_{today_str}.csv', combinedd_df)

print("\nDone.")
