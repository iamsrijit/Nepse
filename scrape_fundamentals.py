# scrape_fundamentals.py
# Place in repository root
#
# Extra requirement vs. before:  pip install lxml   (used by pandas.read_html)

import os
import re
import time
import base64
import threading
from io import StringIO
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from urllib import robotparser

import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from nepse_scraper import Nepse_scraper

# ===========================
# CONFIG
# ===========================
REPO_OWNER = "iamsrijit"
REPO_NAME = "Nepse"
BRANCH = "main"
TODAY = datetime.now().strftime('%Y-%m-%d')
FUNDAMENTAL_FILE = f"Fundamental/Fundamental_{TODAY}.csv"
DIVIDEND_FILE = f"Dividend/Dividend_History_{TODAY}.csv"   # full year-by-year history

WORKERS = 3             # parallel browsers (GitHub Actions runners have 2 cores; 3 is a safe max)
PER_TICKER_PAUSE = 0.4  # polite pause per worker between tickers
PAR_VALUE = 100         # NEPSE standard par value (Rs) - cash % is on this base
DEBUG_DIR = "debug_html"  # page source of first failed dividend page is saved here

ONLINEKHABAR_URL = "https://www.onlinekhabar.com/markets/ticker/"
NEPALIPAISA_URL = "https://nepalipaisa.com/company/"
USER_AGENT = "Mozilla/5.0 (compatible; NepseResearchBot/1.0; personal research use)"

GH_TOKEN = os.environ.get("GH_TOKEN")
if not GH_TOKEN:
    raise RuntimeError("GH_TOKEN not set")

HEADERS = {"Authorization": f"token {GH_TOKEN}"}


def upload_to_github(filename, csv_content):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{filename}"
    r = requests.get(url, headers=HEADERS, params={"ref": BRANCH})
    payload = {
        "message": f"Daily fundamental update {TODAY}",
        "content": base64.b64encode(csv_content.encode('utf-8')).decode('utf-8'),
        "branch": BRANCH,
    }
    if r.status_code == 200:
        payload["sha"] = r.json()["sha"]
    res = requests.put(url, headers=HEADERS, json=payload)
    if res.status_code in (200, 201):
        print(f"Uploaded: {filename}")
    else:
        raise RuntimeError(f"Upload failed: {res.status_code} {res.text}")


def save_local(path, content):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ===========================
# ROBOTS.TXT CHECK (be a good citizen)
# ===========================
def allowed_by_robots(base_url, sample_path):
    try:
        root = re.match(r"https?://[^/]+", base_url).group(0)
        rp = robotparser.RobotFileParser()
        rp.set_url(root + "/robots.txt")
        rp.read()
        return rp.can_fetch(USER_AGENT, sample_path)
    except Exception:
        return True  # robots.txt unreachable -> no explicit restriction found


# ===========================
# NUMERIC CLEANING
# ===========================
def clean_numeric(value):
    if value is None or value == '' or pd.isna(value):
        return None
    text = str(value).strip()
    cleaned = re.sub(r'Rs\.?|%|\s|,', '', text)
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


def pct_to_float(value):
    """'1,842.00 %' -> 1842.0 ; '-' / '' -> None"""
    if value is None or pd.isna(value):
        return None
    t = str(value).replace('%', '').replace(',', '').strip()
    if t in ('', '-', '--', 'nan', 'None'):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def dash_to_none(value):
    if value is None or pd.isna(value):
        return None
    t = str(value).strip()
    return None if t in ('', '-', '--', 'nan', 'None') else t


# ===========================
# SELENIUM SETUP (fast)
# ===========================
DRIVER_PATH = None
_local = threading.local()
_all_drivers = []
_drivers_lock = threading.Lock()


def build_driver():
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1200')
    options.add_argument(f'--user-agent={USER_AGENT}')
    options.page_load_strategy = 'eager'          # don't wait for images/ads
    options.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2,   # no images
    })
    driver = webdriver.Chrome(service=Service(DRIVER_PATH), options=options)
    driver.set_page_load_timeout(30)
    return driver


def thread_driver():
    """One browser per worker thread, reused for every ticker (huge speed-up)."""
    if getattr(_local, "driver", None) is None:
        _local.driver = build_driver()
        with _drivers_lock:
            _all_drivers.append(_local.driver)
    return _local.driver


def reset_thread_driver():
    drv = getattr(_local, "driver", None)
    if drv:
        try:
            drv.quit()
        except Exception:
            pass
    _local.driver = None


# ===========================
# ONLINEKHABAR (existing fundamentals)
# ===========================
def extract_data_selenium(driver, ticker, url, xpath_dict):
    results = {"Ticker": ticker}
    try:
        driver.get(url)
        try:   # wait for page content instead of sleeping a fixed 4 s
            WebDriverWait(driver, 10).until(
                lambda d: d.find_elements(By.XPATH, '//*[@id="sector"]')
                or d.find_elements(By.XPATH, "//table//td")
            )
            time.sleep(0.8)  # let the lazy financial table finish painting
        except TimeoutException:
            pass
        for name, xpath in xpath_dict.items():
            try:
                text = driver.find_element(By.XPATH, xpath).text.strip()
                results[name] = text if text else None
            except NoSuchElementException:
                results[name] = None
    except WebDriverException as e:
        print(f"  ! onlinekhabar failed for {ticker}: {e.__class__.__name__}")
    return results


# ===========================
# NEPALIPAISA (dividend + right share history)
# ===========================
# Table columns on the site:
# S.No. | Fiscal Year | Bonus (%) | Cash (%) | Total (%) | Dividend Book Close | Right Share | Right Book Close
def _map_columns(columns):
    mapping = {}
    for c in columns:
        k = str(c).strip().lower()
        if "fiscal" in k:
            mapping[c] = "Fiscal Year"
        elif "right" in k and "book" in k:
            mapping[c] = "Right Book Close (BS)"
        elif "right" in k:
            mapping[c] = "Right Share"
        elif "dividend" in k and "book" in k:
            mapping[c] = "Dividend Book Close (BS)"
        elif "bonus" in k:
            mapping[c] = "Bonus %"
        elif "cash" in k:
            mapping[c] = "Cash %"
        elif "total" in k:
            mapping[c] = "Total %"
    return mapping


def parse_dividend_table(html):
    """Return list of dict rows (newest fiscal year first) from page HTML."""
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return []
    for t in tables:
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = [" ".join(map(str, c)).strip() for c in t.columns]
        mapping = _map_columns(t.columns)
        if "Fiscal Year" in mapping.values() and "Cash %" in mapping.values():
            t = t.rename(columns=mapping)
            rows = []
            for _, r in t.iterrows():
                fy = dash_to_none(r.get("Fiscal Year"))
                if not fy:
                    continue
                fy = re.sub(r"\s*BS\s*$", "", fy).strip()
                bonus = pct_to_float(r.get("Bonus %"))
                cash = pct_to_float(r.get("Cash %"))
                total = pct_to_float(r.get("Total %"))
                if total is None and (bonus is not None or cash is not None):
                    total = (bonus or 0) + (cash or 0)
                rows.append({
                    "Fiscal Year": fy,
                    "Bonus %": bonus,
                    "Cash %": cash,
                    "Total %": total,
                    "Dividend Book Close (BS)": dash_to_none(r.get("Dividend Book Close (BS)")),
                    "Right Share": dash_to_none(r.get("Right Share")),
                    "Right Book Close (BS)": dash_to_none(r.get("Right Book Close (BS)")),
                })
            rows.sort(key=lambda x: x["Fiscal Year"], reverse=True)
            # de-duplicate (pagination can repeat rows)
            seen, uniq = set(), []
            for x in rows:
                key = (x["Fiscal Year"], x["Right Share"], x["Dividend Book Close (BS)"])
                if key not in seen:
                    seen.add(key)
                    uniq.append(x)
            return uniq
    return []


_debug_saved = threading.Event()


def scrape_dividends(driver, ticker):
    url = f"{NEPALIPAISA_URL}{ticker}"
    for attempt in (1, 2):
        try:
            driver.get(url)
            WebDriverWait(driver, 12).until(
                lambda d: d.find_elements(
                    By.XPATH, "//table[contains(normalize-space(.),'Fiscal Year')]//tr[td]")
            )
            rows = parse_dividend_table(driver.page_source)

            # Best-effort pagination (DataTables-style 'Next' button), max 10 pages
            for _ in range(10):
                nxt = driver.find_elements(
                    By.XPATH,
                    "//a[contains(@class,'next') and not(contains(@class,'disabled'))]"
                    " | //li[contains(@class,'next') and not(contains(@class,'disabled'))]/a")
                if not nxt:
                    break
                driver.execute_script("arguments[0].click();", nxt[0])
                time.sleep(0.4)
                more = parse_dividend_table(driver.page_source)
                known = {(r["Fiscal Year"], r["Right Share"]) for r in rows}
                new = [m for m in more if (m["Fiscal Year"], m["Right Share"]) not in known]
                if not new:
                    break
                rows.extend(new)
            rows.sort(key=lambda x: x["Fiscal Year"], reverse=True)
            return rows
        except TimeoutException:
            # Many companies simply have no dividend history; only retry once.
            if attempt == 2 and not _debug_saved.is_set():
                _debug_saved.set()
                try:
                    save_local(f"{DEBUG_DIR}/{ticker}.html", driver.page_source)
                except Exception:
                    pass
        except WebDriverException:
            reset_thread_driver()
            driver = thread_driver()
    return []


def summarize_dividends(rows, price):
    """One-row summary that is merged into the fundamentals file."""
    out = {
        "Latest Dividend FY": None, "Latest Bonus %": None, "Latest Cash %": None,
        "Latest Total Dividend %": None, "Latest Dividend Book Close (BS)": None,
        "Latest Cash Dividend per Share (Rs)": None, "Cash Dividend Yield (%)": None,
        "Avg Cash % (5Y)": None, "Years Dividend Paid (last 5)": None,
        "Latest Right Share": None, "Latest Right Book Close (BS)": None,
        "Dividend Records": len(rows),
    }
    if not rows:
        return out

    div_rows = [r for r in rows if r["Total %"] is not None or r["Cash %"] is not None]
    if div_rows:
        d = div_rows[0]
        out["Latest Dividend FY"] = d["Fiscal Year"]
        out["Latest Bonus %"] = d["Bonus %"]
        out["Latest Cash %"] = d["Cash %"]
        out["Latest Total Dividend %"] = d["Total %"]
        out["Latest Dividend Book Close (BS)"] = d["Dividend Book Close (BS)"]
        if d["Cash %"] is not None:
            dps = round(d["Cash %"] / 100 * PAR_VALUE, 2)
            out["Latest Cash Dividend per Share (Rs)"] = dps
            if price and price > 0:
                out["Cash Dividend Yield (%)"] = round(dps / price * 100, 2)
        last5 = div_rows[:5]
        cash5 = [r["Cash %"] or 0 for r in last5]
        out["Avg Cash % (5Y)"] = round(sum(cash5) / len(cash5), 2)
        out["Years Dividend Paid (last 5)"] = sum(1 for r in last5 if (r["Total %"] or 0) > 0)

    right_rows = [r for r in rows if r["Right Share"]]
    if right_rows:
        out["Latest Right Share"] = right_rows[0]["Right Share"]
        out["Latest Right Book Close (BS)"] = right_rows[0]["Right Book Close (BS)"]
    return out


# ===========================
# DERIVED RATIO CALCULATORS
# ===========================
def calc_roe(row):
    """ROE = EPS (Trailing) / Book Value per Share (Latest)"""
    eps = row.get("EPS (Trailing)")
    bvps = row.get("Book Value per Share (Latest)")
    if pd.notna(eps) and pd.notna(bvps) and bvps != 0:
        return round(eps / bvps, 4)
    return None


def calc_de(row):
    """D/E = Total Liabilities / (Total Assets - Total Liabilities)"""
    liab = row.get("Total Liabilities (Latest)")
    assets = row.get("Total Assets (Latest)")
    if pd.notna(liab) and pd.notna(assets):
        equity = assets - liab
        if equity != 0:
            return round(liab / equity, 4)
    return None


# ===========================
# XPATHS (onlinekhabar) - unchanged
# ===========================
_FIN = "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[4]/article/div/div[2]/div/div/div/div/div/div/div/table/tbody"
_PERF = "/html/body/div[1]/div/section/main/div/section[2]/div[3]/section[1]/div/div/div/article"
_TOP = "/html/body/div[1]/div/section/main/div/div/section[4]/article/div"

xpath_dict = {
    "Stock Name": "/html/body/div[1]/div/section/main/div/div/section[3]/article/div/p",
    "Ticker (Page)": f"{_TOP}/div[1]/p[1]",
    "Sector": '//*[@id="sector"]',
    "Today's Price": "/html/body/div/div/section/main/div/div/section[4]/article/div/div[1]/p[2]",
    "Market Cap": f"{_TOP}/div[3]/p[2]",
    "Daily Change (%)": f"{_TOP}/div[1]/span/span[1]",
    "Weekly Change (%)": f"{_PERF}[1]/div/div/p",
    "Monthly Change (%)": f"{_PERF}[2]/div/div/p",
    "3-Month Change (%)": f"{_PERF}[3]/div/div/p",
    "Yearly Change (%)": f"{_PERF}[4]/div/div/p",
    "5-Year Change (%)": f"{_PERF}[5]/div/div/p",
    "EPS (Trailing)": "/html/body/div[1]/div/section/main/div/section[2]/div[1]/section[1]/div[1]/article/table/tbody/tr[2]/td[2]",
    "P/E Ratio": "/html/body/div[1]/div/section/main/div/section[2]/div[1]/section[1]/div[1]/article/table/tbody/tr[4]/td[2]",
    "P/B Ratio": "/html/body/div[1]/div/section/main/div/section[2]/div[1]/section[1]/div[1]/article/table/tbody/tr[5]/td[2]",
}

_fin_rows = [
    (2, "Total Revenue", ("Latest Quarter", "Previous Quarter")),
    (3, "Gross Profit", ("Latest", "Previous")),
    (4, "Net Profit", ("Latest", "Previous")),
    (5, "Annualized EPS", ("Latest", "Previous")),
    (6, "Book Value per Share", ("Latest", "Previous")),
    (7, "Total Assets", ("Latest", "Previous")),
    (8, "Total Liabilities", ("Latest", "Previous")),
    (9, "Paid-up Capital", ("Latest", "Previous")),
    (10, "Reserves", ("Latest", "Previous")),
]
for _tr, _label, (_l, _p) in _fin_rows:
    xpath_dict[f"{_label} ({_l})"] = f"{_FIN}/tr[{_tr}]/td[2]/span"
    xpath_dict[f"{_label} ({_p})"] = f"{_FIN}/tr[{_tr}]/td[3]/span"
    xpath_dict[f"{_label} % Change"] = f"{_FIN}/tr[{_tr}]/td[4]/span"

numeric_cols = ["EPS (Trailing)", "P/E Ratio", "P/B Ratio"] + [
    f"{label} ({lbl})" for _, label, pair in _fin_rows for lbl in pair
]


# ===========================
# PER-TICKER JOB
# ===========================
def scrape_ticker(ticker):
    driver = thread_driver()
    data = extract_data_selenium(driver, ticker, f"{ONLINEKHABAR_URL}{ticker}", xpath_dict)
    price = clean_numeric(re.sub(r"[^\d.,]", "", str(data.get("Today's Price") or "")))
    div_rows = scrape_dividends(driver, ticker) if USE_NEPALIPAISA else []
    data.update(summarize_dividends(div_rows, price))
    time.sleep(PER_TICKER_PAUSE)
    print(f"Done {ticker}: {len(div_rows)} dividend/right records")
    return data, [{"Ticker": ticker, **r} for r in div_rows]


# ===========================
# MAIN
# ===========================
def tickers_from_nepse():
    """Primary source: live NEPSE list (retries; the API is flaky / token-based)."""
    last = None
    for attempt in range(1, 4):
        try:
            scraper = Nepse_scraper(verify_ssl=False)
            tp = scraper.get_today_price()
            data = tp if isinstance(tp, list) else tp.get('content', tp.get('data', []))
            found = sorted({i.get('symbol', '').strip() for i in data
                            if i.get('symbol') and i.get('symbol').strip()})
            if found:
                return found
        except Exception as e:
            last = e
            print(f"  NEPSE attempt {attempt} failed: {e}")
            time.sleep(3 * attempt)
    print(f"NEPSE ticker fetch failed ({last})")
    return []


def tickers_from_repo():
    """Fallback: tickers from the newest Fundamental_*.csv already in the GitHub repo."""
    try:
        r = requests.get(
            f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/Fundamental",
            headers=HEADERS, params={"ref": BRANCH}, timeout=30)
        r.raise_for_status()
        files = sorted(f["name"] for f in r.json()
                       if f["name"].startswith("Fundamental_") and f["name"].endswith(".csv"))
        if not files:
            return []
        newest = next(f for f in r.json() if f["name"] == files[-1])
        raw = requests.get(newest["download_url"], headers=HEADERS, timeout=60)
        raw.raise_for_status()
        col = pd.read_csv(StringIO(raw.text))["Ticker"].dropna().astype(str)
        found = sorted({t.strip().upper() for t in col if re.fullmatch(r"[A-Za-z0-9]{2,12}", t.strip())})
        print(f"Using {len(found)} tickers from repo file {files[-1]}")
        return found
    except Exception as e:
        print(f"Repo ticker fallback failed: {e}")
        return []


print("Fetching live tickers from NEPSE...")
ticker_list = tickers_from_nepse() or tickers_from_repo()
if not ticker_list:
    raise RuntimeError("No tickers available from NEPSE or from the repo's previous Fundamental file")
print(f"Found {len(ticker_list)} active tickers.\n")

USE_NEPALIPAISA = allowed_by_robots(NEPALIPAISA_URL, "/company/UNL")
if not USE_NEPALIPAISA:
    print("robots.txt disallows nepalipaisa company pages - skipping dividend scrape.")

DRIVER_PATH = ChromeDriverManager().install()   # install once, not once per ticker

results, dividend_rows = [], []
try:
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for data, rows in pool.map(scrape_ticker, ticker_list):
            results.append(data)
            dividend_rows.extend(rows)
finally:
    for d in _all_drivers:
        try:
            d.quit()
        except Exception:
            pass

df = pd.DataFrame(results)

# ---- clean numeric columns
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

# ---- derived ratios
df["ROE"] = df.apply(calc_roe, axis=1)
df["D/E Ratio"] = df.apply(calc_de, axis=1)
print(f"ROE computed for {df['ROE'].notna().sum()} / {len(df)} stocks")
print(f"D/E computed for {df['D/E Ratio'].notna().sum()} / {len(df)} stocks")
print(f"Dividend data found for {(df['Dividend Records'] > 0).sum()} / {len(df)} stocks")

# ---- save & upload
fund_csv = df.to_csv(index=False)
save_local(FUNDAMENTAL_FILE, fund_csv)
upload_to_github(FUNDAMENTAL_FILE, fund_csv)

if dividend_rows:
    div_csv = pd.DataFrame(dividend_rows).to_csv(index=False)
    save_local(DIVIDEND_FILE, div_csv)
    upload_to_github(DIVIDEND_FILE, div_csv)

print(f"\nCompleted! {len(df)} stocks -> {FUNDAMENTAL_FILE}; "
      f"{len(dividend_rows)} dividend/right records -> {DIVIDEND_FILE}")
