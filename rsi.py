# 3_vs_9.py  (RSI < 30 CROSS version – NO duplicates)
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import base64
from datetime import datetime, timedelta
import os

# ===============================
# STEP 1: Get latest CSV from repo
# ===============================
def get_latest_file_url(repo_url):
    response = requests.get(repo_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    file_links = soup.find_all('a', href=True)

    file_urls = {}
    for link in file_links:
        file_name = link['href']
        if file_name.endswith('.csv'):
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', file_name)
            if date_match:
                file_date = date_match.group(1)
                file_urls[file_date] = repo_url.replace('/tree/', '/raw/') + '/' + file_name

    if not file_urls:
        raise ValueError("No CSV files found in the repository.")

    return file_urls[max(file_urls.keys())]

repo_url = 'https://github.com/iamsrijit/Nepse/tree/main'
latest_file_url = get_latest_file_url(repo_url)
latest_file_url = latest_file_url.replace('/iamsrijit/Nepse/blob/main/', '/')
secondss = pd.read_csv(latest_file_url)

# ===============================
# STEP 2: Clean & preprocess data
# ===============================
expected_columns = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Percent Change', 'Volume']
secondss = secondss[expected_columns]

secondss['Date'] = pd.to_datetime(secondss['Date'], errors='coerce')
secondss.dropna(subset=['Date'], inplace=True)

finall_df = pd.DataFrame()
for symbol in secondss['Symbol'].unique():
    df = secondss[secondss['Symbol'] == symbol]
    df = df.sort_values('Date', ascending=False)
    df = df.drop_duplicates('Date', keep='first')
    finall_df = pd.concat([finall_df, df], ignore_index=True)

# ===============================
# STEP 3: RSI < 30 CROSS Strategy
# ===============================
finall_df = finall_df.sort_values(by=['Symbol', 'Date'])

results = []

def calculate_rsi(prices, length=10):
    prices = np.array(prices)
    deltas = np.diff(prices)

    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros(len(prices))
    avg_loss = np.zeros(len(prices))

    avg_gain[length] = gains[:length].mean()
    avg_loss[length] = losses[:length].mean()

    for i in range(length + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (length - 1) + gains[i - 1]) / length
        avg_loss[i] = (avg_loss[i - 1] * (length - 1) + losses[i - 1]) / length

    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi[:length] = np.nan
    return rsi

def ema(series, span):
    ema_vals = np.full(len(series), np.nan)
    multiplier = 2 / (span + 1)
    for i in range(len(series)):
        if i == span - 1:
            ema_vals[i] = np.nanmean(series[:span])
        elif i > span - 1:
            ema_vals[i] = (series[i] - ema_vals[i - 1]) * multiplier + ema_vals[i - 1]
    return ema_vals

for symbol in finall_df['Symbol'].unique():
    df = finall_df[finall_df['Symbol'] == symbol].reset_index(drop=True)
    prices = df['Close'].tolist()

    if len(prices) < 25:
        continue

    rsi_10 = calculate_rsi(prices, 10)
    rsi_ema_14 = ema(rsi_10, 14)

    # 🔥 CROSS BELOW 30 (NO duplicates)
    for i in range(1, len(df)):
        if (
            not np.isnan(rsi_ema_14[i]) and
            not np.isnan(rsi_ema_14[i - 1]) and
            rsi_ema_14[i] < 30 and
            rsi_ema_14[i - 1] >= 30
        ):
            results.append([
                symbol,
                df.loc[i, 'Date'],
                "Buy",
                df.loc[i, 'Close'],
                round(rsi_ema_14[i], 2)
            ])

cross_signals_df = pd.DataFrame(
    results,
    columns=['Symbol', 'Date', 'Signal', 'Close', 'RSI_EMA_14']
)

# ===============================
# STEP 4: Filter last 300 days
# ===============================
cutoff = datetime.today() - timedelta(days=300)
filtered_df = cross_signals_df[cross_signals_df['Date'] >= cutoff]
filtered_df = filtered_df.sort_values('Date', ascending=False)

# ===============================
# STEP 5: Upload new file & delete old ones
# ===============================
csv_data = filtered_df.to_csv(index=False)
csv_data_base64 = base64.b64encode(csv_data.encode()).decode()

repo = "iamsrijit/Nepse"
file_name = f'RSI_LT_30_for_{datetime.today().strftime("%Y-%m-%d")}.csv'
token = os.environ.get("GH_TOKEN")

headers = {
    'Authorization': f'token {token}',
    'Accept': 'application/vnd.github.v3+json'
}

contents = requests.get(
    f'https://api.github.com/repos/{repo}/contents',
    headers=headers
).json()

for item in contents:
    if item['name'].startswith('RSI_LT_30_for_') and item['name'] != file_name:
        requests.delete(
            f"https://api.github.com/repos/{repo}/contents/{item['name']}",
            headers=headers,
            json={"message": "Delete old file", "sha": item['sha'], "branch": "main"}
        )

payload = {
    'message': f'Upload {file_name}',
    'content': csv_data_base64,
    'branch': 'main'
}

requests.put(
    f'https://api.github.com/repos/{repo}/contents/{file_name}',
    headers=headers,
    json=payload
)

print("✅ Duplicate-free RSI signals uploaded")
