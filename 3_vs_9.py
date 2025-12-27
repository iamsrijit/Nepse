# 3_vs_9.py  (RSI < 30 version)
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

    latest_file_date = max(file_urls.keys())
    return file_urls[latest_file_date]

repo_url = 'https://github.com/iamsrijit/Nepse/tree/main'
latest_file_url = get_latest_file_url(repo_url)
latest_file_url = latest_file_url.replace('/iamsrijit/Nepse/blob/main/', '/')
secondss = pd.read_csv(latest_file_url)

# ===============================
# STEP 2: Clean & preprocess data
# ===============================
expected_columns = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Percent Change', 'Volume']
missing_cols = set(expected_columns) - set(secondss.columns)
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

secondss = secondss[expected_columns]

secondss['Date'] = pd.to_datetime(secondss['Date'], errors='coerce')
secondss.dropna(subset=['Date'], inplace=True)

combined_df = secondss.copy()
combined_df['Date'] = pd.to_datetime(combined_df['Date'])

finall_df = pd.DataFrame()

for symbol in combined_df['Symbol'].unique():
    symbol_df = combined_df[combined_df['Symbol'] == symbol]
    symbol_df = symbol_df.sort_values(by='Date', ascending=False)
    symbol_df = symbol_df.drop_duplicates(subset=['Date'], keep='first')
    finall_df = pd.concat([finall_df, symbol_df], ignore_index=True)

# ===============================
# STEP 3: RSI < 30 Strategy
# ===============================
finall_df = finall_df.sort_values(by=['Symbol', 'Date'], ascending=[True, True])

results = []
insufficient_data = []

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
        elif i >= span:
            ema_vals[i] = (series[i] - ema_vals[i - 1]) * multiplier + ema_vals[i - 1]
    return ema_vals

for symbol in finall_df['Symbol'].unique():
    symbol_df = finall_df[finall_df['Symbol'] == symbol].reset_index(drop=True)
    prices = symbol_df['Close'].tolist()

    if len(prices) < 25:
        insufficient_data.append(symbol)
        continue

    rsi_10 = calculate_rsi(prices, length=10)
    rsi_ema_14 = ema(rsi_10, span=14)

    for i in range(len(symbol_df)):
        if not np.isnan(rsi_ema_14[i]) and rsi_ema_14[i] < 30:
            results.append([
                symbol,
                symbol_df['Date'][i],
                "Buy",
                symbol_df['Close'][i],
                round(rsi_ema_14[i], 2)
            ])

cross_signals_df = pd.DataFrame(
    results,
    columns=['Symbol', 'Date', 'Signal', 'Close', 'RSI_EMA_14']
)

# ===============================
# STEP 4: Filter last 300 days
# ===============================
three_hundred_days_ago = datetime.today() - timedelta(days=300)
cross_signals_df['Date'] = pd.to_datetime(cross_signals_df['Date'])
filtered_df = cross_signals_df[cross_signals_df['Date'] >= three_hundred_days_ago]
filtered_df = filtered_df.sort_values(by='Date', ascending=False)

# ===============================
# STEP 5: Upload new file & delete old ones
# ===============================
try:
    csv_data = filtered_df.to_csv(index=False)
    csv_data_base64 = base64.b64encode(csv_data.encode()).decode()

    repo = "iamsrijit/Nepse"
    file_name = f'RSI_LT_30_for_{datetime.today().strftime("%Y-%m-%d")}.csv'
    token = os.environ.get("GH_TOKEN")

    if not token:
        raise ValueError("GH_TOKEN not found")

    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }

    contents_url = f'https://api.github.com/repos/{repo}/contents'
    contents = requests.get(contents_url, headers=headers).json()

    for item in contents:
        if item['name'].startswith('RSI_LT_30_for_') and item['name'] != file_name:
            delete_url = f"https://api.github.com/repos/{repo}/contents/{item['name']}"
            requests.delete(
                delete_url,
                headers=headers,
                json={
                    "message": f"Delete old file {item['name']}",
                    "sha": item['sha'],
                    "branch": "main"
                }
            )

    upload_url = f'https://api.github.com/repos/{repo}/contents/{file_name}'
    check_response = requests.get(upload_url, headers=headers)
    sha = check_response.json().get('sha') if check_response.status_code == 200 else None

    payload = {
        'message': f'Upload {file_name}',
        'content': csv_data_base64,
        'branch': 'main'
    }
    if sha:
        payload['sha'] = sha

    upload_response = requests.put(upload_url, headers=headers, json=payload)

    if upload_response.status_code in [200, 201]:
        print(f'✅ File {file_name} uploaded successfully')
    else:
        print('❌ Upload failed', upload_response.json())

except Exception as e:
    print("Error:", e)
