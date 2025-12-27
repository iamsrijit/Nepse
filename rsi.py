# RSI_LT_30_LATEST.py
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
        href = link['href']
        if href.endswith('.csv'):
            m = re.search(r'(\d{4}-\d{2}-\d{2})', href)
            if m:
                file_urls[m.group(1)] = repo_url.replace('/tree/', '/raw/') + '/' + href

    if not file_urls:
        raise ValueError("No CSV files found")

    return file_urls[max(file_urls.keys())]

repo_url = 'https://github.com/iamsrijit/Nepse/tree/main'
latest_file_url = get_latest_file_url(repo_url)
latest_file_url = latest_file_url.replace('/iamsrijit/Nepse/blob/main/', '/')
df_raw = pd.read_csv(latest_file_url)

# ===============================
# STEP 2: Clean & preprocess data
# ===============================
cols = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Percent Change', 'Volume']
df_raw = df_raw[cols]

df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
df_raw.dropna(subset=['Date'], inplace=True)

final_df = pd.DataFrame()

for sym in df_raw['Symbol'].unique():
    d = df_raw[df_raw['Symbol'] == sym]
    d = d.sort_values('Date', ascending=False)
    d = d.drop_duplicates('Date', keep='first')
    final_df = pd.concat([final_df, d], ignore_index=True)

final_df = final_df.sort_values(['Symbol', 'Date'])

# ===============================
# STEP 3: RSI(10) + EMA(14) → LATEST < 30
# ===============================
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
        avg_gain[i] = (avg_gain[i-1]*(length-1) + gains[i-1]) / length
        avg_loss[i] = (avg_loss[i-1]*(length-1) + losses[i-1]) / length

    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi[:length] = np.nan
    return rsi

def ema(series, span):
    out = np.full(len(series), np.nan)
    k = 2 / (span + 1)
    for i in range(len(series)):
        if i == span - 1:
            out[i] = np.nanmean(series[:span])
        elif i >= span:
            out[i] = (series[i] - out[i-1]) * k + out[i-1]
    return out

results = []

for symbol in final_df['Symbol'].unique():
    df = final_df[final_df['Symbol'] == symbol].reset_index(drop=True)
    prices = df['Close'].tolist()

    if len(prices) < 25:
        continue

    rsi10 = calculate_rsi(prices, 10)
    rsi_ema14 = ema(rsi10, 14)

    # 🔥 LATEST date where RSI EMA < 30
    for i in range(len(df)-1, -1, -1):
        if not np.isnan(rsi_ema14[i]) and rsi_ema14[i] < 30:
            results.append([
                symbol,
                df.loc[i, 'Date'],
                "Buy",
                df.loc[i, 'Close'],
                round(rsi_ema14[i], 2)
            ])
            break   # only ONE row per stock

signals_df = pd.DataFrame(
    results,
    columns=['Symbol', 'Date', 'Signal', 'Close', 'RSI_EMA_14']
)

# ===============================
# STEP 4: Filter last 300 days
# ===============================
cutoff = datetime.today() - timedelta(days=300)
signals_df = signals_df[signals_df['Date'] >= cutoff]
signals_df = signals_df.sort_values('Date', ascending=False)

# ===============================
# STEP 5: Upload & delete old files
# ===============================
csv_data = signals_df.to_csv(index=False)
encoded = base64.b64encode(csv_data.encode()).decode()

repo = "iamsrijit/Nepse"
file_name = f'RSI_LT_30_LATEST_{datetime.today().strftime("%Y-%m-%d")}.csv'
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
    if item['name'].startswith('RSI_LT_30_LATEST_') and item['name'] != file_name:
        requests.delete(
            f"https://api.github.com/repos/{repo}/contents/{item['name']}",
            headers=headers,
            json={
                "message": "Delete old file",
                "sha": item['sha'],
                "branch": "main"
            }
        )

payload = {
    'message': f'Upload {file_name}',
    'content': encoded,
    'branch': 'main'
}

requests.put(
    f'https://api.github.com/repos/{repo}/contents/{file_name}',
    headers=headers,
    json=payload
)

print("✅ Latest RSI<30 signals uploaded (one per stock)")
