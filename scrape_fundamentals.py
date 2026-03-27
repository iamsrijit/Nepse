# scrape_fundamentals.py

import os
import time
import pandas as pd
import re
import requests
import base64
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# ===========================
# GITHUB CONFIG
# ===========================
REPO_OWNER = "iamsrijit"
REPO_NAME = "Nepse"
BRANCH = "main"
FUNDAMENTAL_FILE = f"Fundamental/Fundamental_{datetime.now().strftime('%Y-%m-%d')}.csv"

GH_TOKEN = os.environ.get("GH_TOKEN")
if not GH_TOKEN:
    raise RuntimeError("GH_TOKEN not set")

HEADERS = {"Authorization": f"token {GH_TOKEN}"}

def upload_to_github(filename, csv_content):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{filename}"

    r = requests.get(url, headers=HEADERS)

    payload = {
        "message": f"Daily fundamental update {datetime.now().strftime('%Y-%m-%d')}",
        "content": base64.b64encode(csv_content.encode('utf-8')).decode('utf-8'),
        "branch": BRANCH
    }

    if r.status_code == 200:
        payload["sha"] = r.json()["sha"]

    res = requests.put(url, headers=HEADERS, json=payload)

    if res.status_code in (200, 201):
        print(f"Uploaded: {filename}")
    else:
        raise RuntimeError(f"Upload failed: {res.status_code} {res.text}")

# ===========================
# NUMERIC CLEANING
# ===========================
def clean_numeric(value):
    if value is None or value == '' or pd.isna(value):
        return None

    text = str(value).strip()
    cleaned = re.sub(r'[Rs\s,]+', '', text)

    multiplier = 1
    if 'Arba' in cleaned:
        cleaned = cleaned.replace('Arba', '')
        multiplier = 1_000_000_000
    elif 'Crore' in cleaned:
        cleaned = cleaned.replace('Crore', '')
        multiplier = 10_000_000

    try:
        return float(cleaned) * multiplier
    except:
        return None

# ===========================
# SELENIUM SETUP
# ===========================
def get_driver():
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def extract_data_selenium(driver, ticker, url, xpath_dict):
    data = {"Ticker": ticker}

    try:
        driver.get(url)
        time.sleep(3)

        for name, xpath in xpath_dict.items():
            try:
                el = driver.find_element(By.XPATH, xpath)
                txt = el.text.strip()
                data[name] = txt if txt else None
            except NoSuchElementException:
                data[name] = None

    except Exception as e:
        print(f"Error scraping {ticker}: {e}")

    return data

# ===========================
# FETCH TICKERS (FIXED)
# ===========================
def fetch_with_retry(session, url, headers, retries=3):
    for i in range(retries):
        res = session.get(url, headers=headers)
        if res.status_code == 200:
            return res
        print(f"Retry {i+1}...")
        time.sleep(2)
    raise RuntimeError("Failed after retries")

def get_tickers():
    print("Fetching tickers...")

    session = requests.Session()

    session.get("https://www.nepalstock.com", headers={
        "User-Agent": "Mozilla/5.0"
    })

    url = "https://www.nepalstock.com/api/nots/nepse-data/today-price?page=0&size=500"

    response = fetch_with_retry(session, url, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nepalstock.com",
        "Origin": "https://www.nepalstock.com"
    })

    data = response.json()

    return sorted({
        item.get("symbol", "").strip()
        for item in data.get("content", [])
        if item.get("symbol")
    })

# ===========================
# RATIOS
# ===========================
def calc_roe(row):
    eps = row.get("EPS (Trailing)")
    bvps = row.get("Book Value per Share (Latest)")
    if pd.notna(eps) and pd.notna(bvps) and bvps != 0:
        return round(eps / bvps, 4)
    return None

def calc_de(row):
    liab = row.get("Total Liabilities (Latest)")
    assets = row.get("Total Assets (Latest)")
    if pd.notna(liab) and pd.notna(assets):
        equity = assets - liab
        if equity != 0:
            return round(liab / equity, 4)
    return None

# ===========================
# MAIN
# ===========================
ticker_list = get_tickers()
print(f"Found {len(ticker_list)} stocks")

driver = get_driver()

url_prefix = "https://www.onlinekhabar.com/markets/ticker/"

xpath_dict = {
    "Stock Name": "/html/body/div[1]/div/section/main/div/div/section[3]/article/div/p",
    "Sector": '//*[@id="sector"]',
    "Today's Price": "/html/body/div/div/section/main/div/div/section[4]/article/div/div[1]/p[2]",
    "Market Cap": "/html/body/div[1]/div/section/main/div/div/section[4]/article/div/div[3]/p[2]",
    "EPS (Trailing)": "/html/body/div[1]/div/section/main/div/section[2]/div[1]/section[1]/div[1]/article/table/tbody/tr[2]/td[2]",
    "P/E Ratio": "/html/body/div[1]/div/section/main/div/section[2]/div[1]/section[1]/div[1]/article/table/tbody/tr[4]/td[2]",
    "P/B Ratio": "/html/body/div[1]/div/section/main/div/section[2]/div[1]/section[1]/div[1]/article/table/tbody/tr[5]/td[2]",
    "Book Value per Share (Latest)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[6]/td[2]/span",
    "Total Assets (Latest)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[7]/td[2]/span",
    "Total Liabilities (Latest)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[8]/td[2]/span",
}

results = []

for ticker in ticker_list:
    print(f"Scraping {ticker}...")
    url = f"{url_prefix}{ticker}"

    data = extract_data_selenium(driver, ticker, url, xpath_dict)
    results.append(data)

    time.sleep(1)

driver.quit()

df = pd.DataFrame(results)

# CLEAN NUMERIC
numeric_cols = [
    "EPS (Trailing)",
    "P/E Ratio",
    "P/B Ratio",
    "Book Value per Share (Latest)",
    "Total Assets (Latest)",
    "Total Liabilities (Latest)"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

# RATIOS
df["ROE"] = df.apply(calc_roe, axis=1)
df["D/E Ratio"] = df.apply(calc_de, axis=1)

print("ROE computed:", df["ROE"].notna().sum())
print("D/E computed:", df["D/E Ratio"].notna().sum())

# SAVE
csv_content = df.to_csv(index=False)

with open(FUNDAMENTAL_FILE, "w", encoding="utf-8") as f:
    f.write(csv_content)

upload_to_github(FUNDAMENTAL_FILE, csv_content)

print("✅ DONE:", FUNDAMENTAL_FILE)
