# scrape_fundamentals.py
# Place in repository root

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
from nepse_scraper import Nepse_scraper

# ===========================
# GITHUB CONFIG
# ===========================
REPO_OWNER = "iamsrijit"
REPO_NAME = "Nepse"
BRANCH = "main"
FUNDAMENTAL_FILE = f"Fundamental_{datetime.now().strftime('%Y-%m-%d')}.csv"

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
        cleaned = cleaned.replace('Arba', '').strip()
        multiplier = 1_000_000_000
    elif 'Crore' in cleaned:
        cleaned = cleaned.replace('Crore', '').strip()
        multiplier = 10_000_000
    try:
        return float(cleaned) * multiplier
    except ValueError:
        return None

# ===========================
# SELENIUM SETUP (Standard for GitHub Actions)
# ===========================
def get_driver():
    options = Options()
    options.add_argument('--headless=new')  # Modern headless mode
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1200')
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def extract_data_selenium(ticker, url, xpath_dict):
    driver = get_driver()
    results = {"Ticker": ticker}
    try:
        driver.get(url)
        time.sleep(4)
        for name, xpath in xpath_dict.items():
            try:
                element = driver.find_element(By.XPATH, xpath)
                text = element.text.strip()
                results[name] = text if text else None
            except NoSuchElementException:
                results[name] = None
    finally:
        driver.quit()
    return results

# ===========================
# MAIN
# ===========================
print("Fetching live tickers from NEPSE...")
scraper = Nepse_scraper(verify_ssl=False)
today_price = scraper.get_today_price()

if isinstance(today_price, list):
    content_data = today_price
else:
    content_data = today_price.get('content', today_price.get('data', []))

ticker_list = sorted({
    item.get('symbol', '').strip()
    for item in content_data
    if item.get('symbol') and item.get('symbol').strip()
})

print(f"Found {len(ticker_list)} active tickers.\n")

url_prefix = "https://www.onlinekhabar.com/markets/ticker/"

xpath_dict = {
       "Stock Name": "/html/body/div[1]/div/section/main/div/div/section[3]/article/div/p",
    "Ticker": "/html/body/div[1]/div/section/main/div/div/section[4]/article/div/div[1]/p[1]",
    "Sector": '//*[@id="sector"]',
    "Today's Price": "/html/body/div/div/section/main/div/div/section[4]/article/div/div[1]/p[2]",
    "Market Cap": "/html/body/div[1]/div/section/main/div/div/section[4]/article/div/div[3]/p[2]",
    "Daily Change (%)": "/html/body/div[1]/div/section/main/div/div/section[4]/article/div/div[1]/span/span[1]",
    "Weekly Change (%)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[1]/div/div/div/article[1]/div/div/p",
    "Monthly Change (%)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[1]/div/div/div/article[2]/div/div/p",
    "3-Month Change (%)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[1]/div/div/div/article[3]/div/div/p",
    "Yearly Change (%)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[1]/div/div/div/article[4]/div/div/p",
    "5-Year Change (%)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[1]/div/div/div/article[5]/div/div/p",
    "EPS (Trailing)": "/html/body/div[1]/div/section/main/div/section[2]/div[1]/section[1]/div[1]/article/table/tbody/tr[2]/td[2]",
    "P/E Ratio": "/html/body/div[1]/div/section/main/div/section[2]/div[1]/section[1]/div[1]/article/table/tbody/tr[4]/td[2]",
    "P/B Ratio": "/html/body/div[1]/div/section/main/div/section[2]/div[1]/section[1]/div[1]/article/table/tbody/tr[5]/td[2]",
    "RSI": "/html/body/div[1]/div/section/main/div/section[2]/div[1]/section[1]/div[2]/article/table/tbody/tr[1]/td[2]",
    "ROE": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[4]/td[2]/span",

    "Total Revenue (Latest Quarter)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[2]/td[2]/span",
    "Total Revenue (Previous Quarter)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[2]/td[3]/span",
    "Total Revenue % Change": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[2]/td[4]/span",

    "Gross Profit (Latest)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[3]/td[2]/span",
    "Gross Profit (Previous)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[3]/td[3]/span",
    "Gross Profit % Change": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[3]/td[4]/span",

    "Net Profit (Latest)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[4]/td[2]/span",
    "Net Profit (Previous)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[4]/td[3]/span",
    "Net Profit % Change": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[4]/td[4]/span",

    "Annualized EPS (Latest)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[5]/td[2]/span",
    "Annualized EPS (Previous)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[5]/td[3]/span",
    "Annualized EPS % Change": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[5]/td[4]/span",

    "Book Value per Share (Latest)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[6]/td[2]/span",
    "Book Value per Share (Previous)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[6]/td[3]/span",
    "Book Value per Share % Change": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[6]/td[4]/span",

    "Total Assets (Latest)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[7]/td[2]/span",
    "Total Assets (Previous)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[7]/td[3]/span",
    "Total Assets % Change": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[7]/td[4]/span",

    "Total Liabilities (Latest)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[8]/td[2]/span",
    "Total Liabilities (Previous)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[8]/td[3]/span",
    "Total Liabilities % Change": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[8]/td[4]/span",

    "Paid-up Capital (Latest)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[9]/td[2]/span",
    "Paid-up Capital (Previous)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[9]/td[3]/span",
    "Paid-up Capital % Change": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[9]/td[4]/span",

    "Reserves (Latest)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[10]/td[2]/span",
    "Reserves (Previous)": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[10]/td[3]/span",
    "Reserves % Change": "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody/tr[10]/td[4]/span",
}

results = []
for ticker in ticker_list:
    url = f"{url_prefix}{ticker}"
    print(f"Scraping {ticker}...")
    data = extract_data_selenium(ticker, url, xpath_dict)
    results.append(data)
    time.sleep(1.2)

df = pd.DataFrame(results)

numeric_cols = [
    "EPS", "PE ratio", "PB ratio", "RSI",
    "T Rev L", "T Rev P", "Gross Profit L", "Gross Profit P",
    "Net Profit L", "Net Profit P", "% change in Net Profit",
    "Eps Annualized L", "Eps Annualized P",
    "Book Value Per Share L", "Book Value Per Share P",
    "Total Asset L", "Total Asset P",
    "Total Liabilities L", "Total Liabilities P",
    "Paid Up Capital L", "Paid Up Capital P",
    "Reserves L", "Reserves P"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

csv_content = df.to_csv(index=False)
with open(FUNDAMENTAL_FILE, "w", encoding="utf-8") as f:
    f.write(csv_content)

upload_to_github(FUNDAMENTAL_FILE, csv_content)

print(f"\nCompleted! {len(df)} stocks scraped and uploaded as {FUNDAMENTAL_FILE}")
