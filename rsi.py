# RSI_LT_30_LATEST_fixed_close.py
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
    response.raise_for_status()
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
        raise ValueError("No CSV files found in the repository.")

    return file_urls[max(file_urls.keys())]

repo_url = 'https://github.com/iamsrijit/Nepse/tree/main'
latest_file_url = get_latest_file_url(repo_url)
latest_file_url = latest_file_url.replace('/iamsrijit/Nepse/blob/main/', '/')
df_raw = pd.read_csv(latest_file_url)

# ===============================
# STEP 2: Clean & preprocess data
# ===============================
expected_cols = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Percent Change', 'Volume']
missing_cols = set(expected_cols) - set(df_raw.columns)
if missing_cols:
    raise ValueError(f"Missing columns from CSV: {missing_cols}")

df_raw = df_raw[expected_cols].copy()
df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
df_raw.dropna(subset=['Date'], inplace=True)

# Normalize Close column to numeric (coerce errors)
df_raw['Close'] = pd.to_numeric(df_raw['Close'], errors='coerce')
df_raw.dropna(subset=['Close'], inplace=True)

# Remove duplicate dates per symbol keeping latest entry for each date
final_df = pd.DataFrame()
for sym in df_raw['Symbol'].unique():
    d = df_raw[df_raw['Symbol'] == sym].copy()
    d = d.sort_values('Date', ascending=False)  # latest first
    d = d.drop_duplicates('Date', keep='first')
    final_df = pd.concat([final_df, d], ignore_index=True)

final_df = final_df.sort_values(['Symbol', 'Date']).reset_index(drop=True)  # oldest -> newest per symbol

# ===============================
# STEP 3: RSI(10) + EMA(14) → LATEST < 30
# ===============================
def calculate_rsi(prices, length=10):
    prices = np.asarray(prices, dtype=float)
    if len(prices) < length + 1:
        return np.full(len(prices), np.nan)

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    rsi = np.full(len(prices), np.nan)
    # initial average gain/loss (Wilder smoothing seed)
    avg_gain = np.zeros(len(prices))
    avg_loss = np.zeros(len(prices))

    avg_gain[length] = gains[:length].mean()
    avg_loss[length] = losses[:length].mean()

    # fill forward
    for i in range(length + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (length - 1) + gains[i - 1]) / length
        avg_loss[i] = (avg_loss[i - 1] * (length - 1) + losses[i - 1]) / length

    # avoid divide-by-zero
    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi[:length] = np.nan
    return rsi

def ema(series, span):
    series = np.asarray(series, dtype=float)
    out = np.full(len(series), np.nan)
    if len(series) < span:
        return out
    k = 2 / (span + 1)
    # seed with simple mean
    out[span - 1] = np.nanmean(series[:span])
    for i in range(span, len(series)):
        out[i] = (series[i] - out[i - 1]) * k + out[i - 1]
    return out

results = []

for symbol in final_df['Symbol'].unique():
    df = final_df[final_df['Symbol'] == symbol].copy().reset_index(drop=True)
    # Ensure sorted oldest -> newest
    df = df.sort_values('Date', ascending=True).reset_index(drop=True)

    prices = df['Close'].astype(float).values
    if len(prices) < 25:
        # not enough data to compute RSI(10) and EMA(14) reliably
        continue

    rsi10 = calculate_rsi(prices, length=10)
    rsi_ema14 = ema(rsi10, span=14)

    # iterate from latest backwards and pick first index where rsi_ema14 < 30
    for i in range(len(df) - 1, -1, -1):
        val = rsi_ema14[i]
        if not np.isnan(val) and val < 30:
            # use iloc to guarantee we fetch the Close and Date from the same row
            row_date = df.iloc[i]['Date']
            row_close = df.iloc[i]['Close']
            results.append([
                symbol,
                pd.to_datetime(row_date),
                "Buy",
                float(row_close),
                round(float(val), 2)
            ])
            break  # only one entry per symbol (latest)
    # if loop finishes without break => no RSI<30 for this symbol, nothing appended

signals_df = pd.DataFrame(results, columns=['Symbol', 'Date', 'Signal', 'Close', 'RSI_EMA_14'])

# ===============================
# STEP 4: Filter last 300 days
# ===============================
signals_df['Date'] = pd.to_datetime(signals_df['Date'])
cutoff = datetime.today() - timedelta(days=900)
signals_df = signals_df[signals_df['Date'] >= cutoff]
signals_df = signals_df.sort_values('Date', ascending=False).reset_index(drop=True)

# ===============================
# STEP 5: Upload & delete old files
# ===============================
csv_data = signals_df.to_csv(index=False)
encoded = base64.b64encode(csv_data.encode()).decode()

repo = "iamsrijit/Nepse"
file_name = f'RSI_LT_30_LATEST_{datetime.today().strftime("%Y-%m-%d")}.csv'
token = os.environ.get("GH_TOKEN")

if not token:
    # don't fail silently; inform the user
    print("Warning: GH_TOKEN not set. Skipping upload. You can save the CSV locally instead.")
    # Save locally as fallback
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(csv_data)
    print(f"Saved output to local file: {file_name}")
else:
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }

    # list contents
    contents_resp = requests.get(f'https://api.github.com/repos/{repo}/contents', headers=headers)
    if contents_resp.status_code == 200:
        contents = contents_resp.json()
        for item in contents:
            if item.get('name', '').startswith('RSI_LT_30_LATEST_') and item.get('name') != file_name:
                requests.delete(
                    f"https://api.github.com/repos/{repo}/contents/{item['name']}",
                    headers=headers,
                    json={
                        "message": f"Delete old file {item['name']}",
                        "sha": item['sha'],
                        "branch": "main"
                    }
                )
    else:
        print("Warning: could not list repository contents. Status:", contents_resp.status_code)

    # upload new file (create/update)
    upload_url = f'https://api.github.com/repos/{repo}/contents/{file_name}'
    check_resp = requests.get(upload_url, headers=headers)
    sha = None
    if check_resp.status_code == 200:
        sha = check_resp.json().get('sha')

    payload = {
        'message': f'Upload {file_name}',
        'content': encoded,
        'branch': 'main'
    }
    if sha:
        payload['sha'] = sha

    put_resp = requests.put(upload_url, headers=headers, json=payload)
    if put_resp.status_code in (200, 201):
        print(f'✅ File {file_name} uploaded successfully')
    else:
        print('❌ Upload failed', put_resp.status_code, put_resp.text)

print("Done.")
