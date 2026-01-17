# Install necessary libraries if not available (run this cell first)
!pip install pandas numpy requests

import pandas as pd
import numpy as np
import requests
from datetime import datetime

# GitHub repo details
repo_owner = 'iamsrijit'
repo_name = 'Nepse'
repo_path = 'main'  # or branch name

# Fetch list of files from GitHub API
api_url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/contents/'
response = requests.get(api_url)
if response.status_code != 200:
    raise ValueError(f"Failed to fetch repo contents: {response.text}")

data = response.json()

# Filter files starting with 'espen_2026'
matching_files = []
for item in data:
    if item['type'] == 'file' and item['name'].startswith('espen_2026'):
        # Extract date from filename, e.g., 'espen_2026-01-15.csv' -> '2026-01-15'
        try:
            date_str = item['name'].split('_')[1].replace('.csv', '')
            file_date = datetime.strptime(date_str, '%Y-%m-%d')
            matching_files.append((item['name'], file_date))
        except (IndexError, ValueError):
            continue  # Skip if filename doesn't match expected format

if not matching_files:
    raise ValueError("No files found starting with 'espen_2026' in the repo.")

# Sort by date descending and pick the latest
matching_files.sort(key=lambda x: x[1], reverse=True)
latest_filename = matching_files[0][0]

# Construct raw URL
raw_url = f'https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{repo_path}/{latest_filename}'

# Read the CSV from raw URL
df = pd.read_csv(raw_url)

# Convert Date to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Ensure columns are present (case-insensitive, but assume standard names: open, high, low, close, volume)
df.columns = [col.strip().lower() for col in df.columns]
required_cols = ['open', 'high', 'low', 'close', 'volume']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col.upper()}")

# Function to calculate ATR (Average True Range)
def calculate_atr(df, period=14):
    # True Range = max[(High - Low), abs(High - Close_prev), abs(Low - Close_prev)]
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift())
    df['low_close'] = np.abs(df['low'] - df['close'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    return df

# Calculate indicators
df = calculate_atr(df, period=14)
df['sma50'] = df['close'].rolling(window=50).mean()  # 50-day Simple Moving Average for trend
df['avg_vol20'] = df['volume'].rolling(window=20).mean()  # 20-day Average Volume

# Define buy signal based on strategy (Trend + Momentum Hybrid):
# - Close > 50-day SMA (uptrend)
# - Volume > 1.2 * 20-day Avg Volume (sentiment/momentum spike)
# - ATR > previous ATR (increasing volatility, potential for asymmetric wins)
# Also, avoid buying if recent drawdown (simple check: not after 5% drop in last 5 days)
df['recent_drawdown'] = df['close'] / df['close'].rolling(5).max() - 1  # Max drawdown in last 5 days
df['buy_signal'] = (
    (df['close'] > df['sma50']) &
    (df['volume'] > df['avg_vol20'] * 1.2) &
    (df['atr'] > df['atr'].shift(1)) &
    (df['recent_drawdown'] > -0.05)  # No more than 5% drawdown recently
)

# Drop NaN rows (due to rolling windows)
df = df.dropna()

# Get buy dates
buy_dates = df[df['buy_signal']].index

# Output the buy dates
print(f"Buy dates for the stock based on the strategy (from {df.index.min().date()} to {df.index.max().date()}):")
if len(buy_dates) == 0:
    print("No buy signals found in the provided data.")
else:
    for date in buy_dates:
        print(date.strftime('%Y-%m-%d'))

# Optional: To simulate position sizing for each buy date (example with starting capital ₹5,00,000)
# Adjust alpha (risk fraction) as per model: 0.8% to 1.2%
alpha = 0.01  # 1% risk per trade
capital = 500000  # Starting capital

print("\nSimulated position sizes for each buy date (example with ₹5L capital, 1% risk, 10% stop distance assumption):")
for date in buy_dates:
    entry_price = df.at[date, 'close']
    atr = df.at[date, 'atr']
    # Stop-loss example: Entry - (2 * ATR) for volatility-adjusted stop
    stop_loss = entry_price - (2 * atr)
    stop_distance = (entry_price - stop_loss) / entry_price  # As fraction
    if stop_distance <= 0:
        continue
    risk_per_trade = alpha * capital
    quantity = risk_per_trade / (entry_price * stop_distance)
    position_value = quantity * entry_price
    print(f"{date.strftime('%Y-%m-%d')}: Entry @ {entry_price:.2f}, Stop @ {stop_loss:.2f}, Qty: {int(quantity)}, Position Value: ₹{position_value:.2f}")