# scrape_fundamentals.py
# Place in repository root.   Requires: pip install lxml
#
# Sources
#   - NEPSE (via nepse_scraper)  : list of live tickers   (fallback: last Fundamental CSV in repo)
#   - onlinekhabar.com           : price / ratios / quarterly financials
#   - nepalipaisa.com            : year-by-year dividend + right share history

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
from selenium.common.exceptions import WebDriverException, TimeoutException
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
DIVIDEND_FILE = f"Dividend/Dividend_History_{TODAY}.csv"      # every year, every company (long format)

WORKERS = 3                 # parallel browsers. 2-4 is sensible on a GitHub runner
PER_TICKER_PAUSE = 0.3      # polite pause between tickers (per worker)
RECYCLE_EVERY = 30          # restart each browser after N tickers (prevents memory/state build-up)
OK_TIMEOUT = 14             # max seconds to wait for a fully rendered onlinekhabar page
PAR_VALUE = 100             # NEPSE par value (Rs); cash % is on this base
DEBUG_DIR = "debug_html"

ONLINEKHABAR_URL = "https://www.onlinekhabar.com/markets/ticker/"
NEPALIPAISA_URL = "https://nepalipaisa.com/company/"
USER_AGENT = "Mozilla/5.0 (compatible; NepseResearchBot/1.0; personal research use)"

GH_TOKEN = os.environ.get("GH_TOKEN")
if not GH_TOKEN:
    raise RuntimeError("GH_TOKEN not set")
HEADERS = {"Authorization": f"token {GH_TOKEN}"}


# ===========================
# GITHUB / FILE HELPERS
# ===========================
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


def allowed_by_robots(base_url, sample_path):
    try:
        root = re.match(r"https?://[^/]+", base_url).group(0)
        rp = robotparser.RobotFileParser()
        rp.set_url(root + "/robots.txt")
        rp.read()
        return rp.can_fetch(USER_AGENT, sample_path)
    except Exception:
        return True


# ===========================
# NUMBER CLEANING
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


def first_number(text):
    m = re.search(r"[\d,]+(?:\.\d+)?", str(text or ""))
    return clean_numeric(m.group(0)) if m else None


# ===========================
# BROWSER MANAGEMENT
# ===========================
DRIVER_PATH = None
_local = threading.local()
_all_drivers = []
_drivers_lock = threading.Lock()


def build_driver():
    o = Options()
    o.add_argument('--headless=new')
    o.add_argument('--no-sandbox')
    o.add_argument('--disable-dev-shm-usage')
    o.add_argument('--disable-gpu')
    o.add_argument('--window-size=1920,1200')
    o.add_argument(f'--user-agent={USER_AGENT}')
    o.page_load_strategy = 'eager'
    o.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2})
    d = webdriver.Chrome(service=Service(DRIVER_PATH), options=o)
    d.set_page_load_timeout(30)
    return d


def thread_driver():
    if getattr(_local, "driver", None) is None or getattr(_local, "count", 0) >= RECYCLE_EVERY:
        reset_thread_driver()
        _local.driver = build_driver()
        _local.count = 0
        with _drivers_lock:
            _all_drivers.append(_local.driver)
    return _local.driver


def reset_thread_driver():
    d = getattr(_local, "driver", None)
    if d is not None:
        try:
            d.quit()
        except Exception:
            pass
    _local.driver = None
    _local.count = 0


def safe_get(driver, url):
    """Always start from a blank page so nothing from the previous company can linger."""
    try:
        driver.get("about:blank")
        driver.get(url)
    except TimeoutException:
        try:
            driver.execute_script("window.stop();")
        except Exception:
            pass


# ===========================
# ONLINEKHABAR  (prices, ratios, financials)
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

# Reads ALL xpaths in ONE browser round-trip (much faster than 45 find_element calls)
JS_READ = """
const xs = arguments[0]; const out = {};
for (const k in xs) {
  try {
    const n = document.evaluate(xs[k], document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
    const t = n ? (n.innerText || n.textContent || '').trim() : '';
    out[k] = t === '' ? null : t;
  } catch (e) { out[k] = null; }
}
return out;
"""


def page_is_ticker(page_text, ticker):
    return ticker.upper() in re.findall(r"[A-Z0-9]+", (page_text or "").upper())


def read_onlinekhabar(driver, ticker):
    """
    Wait until the page for THIS ticker is fully rendered and the values stop changing.
    Returns (values|None, status) with status in ok / empty / partial / mismatch / error.
    """
    safe_get(driver, f"{ONLINEKHABAR_URL}{ticker}")
    start = time.time()
    prev, stable, cur = None, 0, None
    while time.time() - start < OK_TIMEOUT:
        cur = driver.execute_script(JS_READ, xpath_dict)
        page_t = cur.get("Ticker (Page)")
        if page_t and not page_is_ticker(page_t, ticker):
            prev, stable = None, 0                      # another company's content -> keep waiting
        else:
            if any(cur.values()) and cur == prev:
                stable += 1
                if stable >= 2:
                    return cur, "ok"
            else:
                stable = 0
            if not any(cur.values()) and time.time() - start > 5:
                return cur, "empty"                     # page genuinely has no data
        prev = cur
        time.sleep(0.5)
    page_t = (cur or {}).get("Ticker (Page)")
    if page_t and not page_is_ticker(page_t, ticker):
        return None, "mismatch"
    return cur, "partial"


# ===========================
# NEPALIPAISA  (dividend + right share, all years)
# ===========================
FY_RE = re.compile(r"^\d{4}\s*/\s*\d{2,4}")


def _map_columns(columns):
    m = {}
    for c in columns:
        k = str(c).strip().lower()
        if "fiscal" in k:
            m[c] = "Fiscal Year"
        elif "right" in k and "book" in k:
            m[c] = "Right Book Close (BS)"
        elif "right" in k:
            m[c] = "Right Share"
        elif "dividend" in k and "book" in k:
            m[c] = "Dividend Book Close (BS)"
        elif "bonus" in k:
            m[c] = "Bonus %"
        elif "cash" in k:
            m[c] = "Cash %"
        elif "total" in k:
            m[c] = "Total %"
    return m


def parse_dividend_table(html):
    """Table HTML -> list of dict rows, newest fiscal year first."""
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return []
    for t in tables:
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = [" ".join(map(str, c)).strip() for c in t.columns]
        mp = _map_columns(t.columns)
        if "Fiscal Year" not in mp.values() or "Cash %" not in mp.values():
            continue
        t = t.rename(columns=mp)
        rows = []
        for _, r in t.iterrows():
            fy = dash_to_none(r.get("Fiscal Year"))
            if not fy or not FY_RE.match(fy):          # skips "No data available in table"
                continue
            fy = re.sub(r"\s*BS\s*$", "", fy).strip()
            bonus, cash, total = (pct_to_float(r.get("Bonus %")), pct_to_float(r.get("Cash %")),
                                  pct_to_float(r.get("Total %")))
            if total is None and (bonus is not None or cash is not None):
                total = (bonus or 0) + (cash or 0)
            rows.append({
                "Fiscal Year": fy, "Bonus %": bonus, "Cash %": cash, "Total %": total,
                "Dividend Book Close (BS)": dash_to_none(r.get("Dividend Book Close (BS)")),
                "Right Share": dash_to_none(r.get("Right Share")),
                "Right Book Close (BS)": dash_to_none(r.get("Right Book Close (BS)")),
            })
        return _dedupe_sort(rows)
    return []


def _dedupe_sort(rows):
    seen, out = set(), []
    for x in rows:
        key = (x["Fiscal Year"], x["Right Share"], x["Dividend Book Close (BS)"], x["Cash %"], x["Bonus %"])
        if key not in seen:
            seen.add(key)
            out.append(x)
    out.sort(key=lambda x: x["Fiscal Year"], reverse=True)
    return out


JS_DIV_STATE = """
const pane = document.querySelector('#c-dividend');
const h1 = document.querySelector('h1');
const tbl = pane ? pane.querySelector('table') : null;
return {h1: h1 ? h1.innerText : '', hasPane: !!pane,
        text: pane ? pane.innerText.slice(0, 600) : '', html: tbl ? tbl.outerHTML : ''};
"""
JS_OPEN_TAB = "const a=document.querySelector('a[href=\"#c-dividend\"]'); if(a){a.click(); return true;} return false;"
JS_PAGE_100 = """
const pane = document.querySelector('#c-dividend'); if(!pane) return false;
const sel = pane.querySelector('select'); if(!sel) return false;
const o = Array.from(sel.options).find(x => x.text.trim()==='100' || x.value==='100');
if(!o) return false; sel.value = o.value; sel.dispatchEvent(new Event('change',{bubbles:true})); return true;
"""
JS_NEXT = """
const pane = document.querySelector('#c-dividend'); if(!pane) return false;
const n = pane.querySelector('.paginate_button.next:not(.disabled), li.next:not(.disabled) a, a.next:not(.disabled)');
if(!n) return false; n.click(); return true;
"""

_debug_count = 0
_debug_lock = threading.Lock()


def _debug_dump(driver, ticker, note):
    global _debug_count
    with _debug_lock:
        if _debug_count >= 3:
            return
        _debug_count += 1
    try:
        save_local(f"{DEBUG_DIR}/{ticker}_{note}.html", driver.page_source)
    except Exception:
        pass


def scrape_dividends(driver, ticker):
    """Returns (rows, status). status: ok / empty / page_mismatch / no_tab / error."""
    safe_get(driver, f"{NEPALIPAISA_URL}{ticker}")
    st, t0 = None, time.time()
    while time.time() - t0 < 10:                         # 1) page rendered for THIS ticker
        st = driver.execute_script(JS_DIV_STATE)
        if st["h1"] and f"({ticker.upper()})" in st["h1"].upper() and st["hasPane"]:
            break
        time.sleep(0.4)
    else:
        _debug_dump(driver, ticker, "page")
        return [], ("page_mismatch" if st and st["h1"] else "error")

    driver.execute_script(JS_OPEN_TAB)                   # 2) open "Dividend / Right" tab
    rows, t1 = [], time.time()
    while time.time() - t1 < 6:                          # 3) wait for the AJAX table
        st = driver.execute_script(JS_DIV_STATE)
        rows = parse_dividend_table(st["html"]) if st["html"] else []
        if rows:
            break
        if "no data" in st["text"].lower() and time.time() - t1 > 2.5:
            break
        time.sleep(0.4)
    if not rows:
        _debug_dump(driver, ticker, "empty")
        return [], "empty"

    if len(rows) >= 10:                                  # 4) show up to 100 rows, then page through
        if driver.execute_script(JS_PAGE_100):
            time.sleep(0.8)
            rows = _dedupe_sort(rows + parse_dividend_table(driver.execute_script(JS_DIV_STATE)["html"]))
        for _ in range(10):
            before = len(rows)
            if not driver.execute_script(JS_NEXT):
                break
            time.sleep(0.8)
            rows = _dedupe_sort(rows + parse_dividend_table(driver.execute_script(JS_DIV_STATE)["html"]))
            if len(rows) == before:
                break
    return rows, "ok"


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def summarize_dividends(rows, price):
    """Collapse the history into columns for the fundamentals sheet (all years kept as text + Y1..Y5)."""
    out = {
        "Dividend Records": len(rows),
        "Latest Dividend FY": None, "Latest Bonus %": None, "Latest Cash %": None,
        "Latest Total Dividend %": None, "Latest Dividend Book Close (BS)": None,
        "Latest Cash Dividend per Share (Rs)": None, "Cash Dividend Yield (%)": None,
        "Avg Total Dividend % (5Y)": None, "Avg Cash % (5Y)": None, "Avg Bonus % (5Y)": None,
        "Avg Cash Yield % (5Y avg, on current price)": None, "Years Dividend Paid (last 5)": None,
        "Latest Right Share": None, "Latest Right Book Close (BS)": None, "Right Share Count": 0,
        "Dividend History (All)": None, "Right Share History (All)": None,
    }
    for i in range(1, 6):
        for f in ("FY", "Bonus %", "Cash %", "Total %"):
            out[f"Div Y{i} {f}"] = None
    if not rows:
        return out

    div_rows = [r for r in rows if r["Bonus %"] is not None or r["Cash %"] is not None]
    if div_rows:
        d = div_rows[0]
        out.update({
            "Latest Dividend FY": d["Fiscal Year"], "Latest Bonus %": d["Bonus %"],
            "Latest Cash %": d["Cash %"], "Latest Total Dividend %": d["Total %"],
            "Latest Dividend Book Close (BS)": d["Dividend Book Close (BS)"],
        })
        if d["Cash %"] is not None:
            dps = round(d["Cash %"] / 100 * PAR_VALUE, 2)
            out["Latest Cash Dividend per Share (Rs)"] = dps
            if price and price > 0:
                out["Cash Dividend Yield (%)"] = round(dps / price * 100, 2)

        last5 = div_rows[:5]
        out["Avg Total Dividend % (5Y)"] = _avg([r["Total %"] for r in last5])
        out["Avg Cash % (5Y)"] = _avg([r["Cash %"] or 0 for r in last5])
        out["Avg Bonus % (5Y)"] = _avg([r["Bonus %"] or 0 for r in last5])
        avg_dps = _avg([(r["Cash %"] or 0) / 100 * PAR_VALUE for r in last5])
        if avg_dps is not None and price and price > 0:
            out["Avg Cash Yield % (5Y avg, on current price)"] = round(avg_dps / price * 100, 2)
        out["Years Dividend Paid (last 5)"] = sum(1 for r in last5 if (r["Total %"] or 0) > 0)
        for i, r in enumerate(last5, 1):
            out[f"Div Y{i} FY"] = r["Fiscal Year"]
            out[f"Div Y{i} Bonus %"] = r["Bonus %"]
            out[f"Div Y{i} Cash %"] = r["Cash %"]
            out[f"Div Y{i} Total %"] = r["Total %"]
        out["Dividend History (All)"] = "; ".join(
            f"{r['Fiscal Year']}: B{r['Bonus %'] or 0:g}|C{r['Cash %'] or 0:g}|T{r['Total %'] or 0:g}"
            + (f" @{r['Dividend Book Close (BS)']}" if r["Dividend Book Close (BS)"] else "")
            for r in div_rows)

    right_rows = [r for r in rows if r["Right Share"]]
    if right_rows:
        out["Latest Right Share"] = right_rows[0]["Right Share"]
        out["Latest Right Book Close (BS)"] = right_rows[0]["Right Book Close (BS)"]
        out["Right Share Count"] = len(right_rows)
        out["Right Share History (All)"] = "; ".join(
            f"{r['Fiscal Year']}: {r['Right Share']}"
            + (f" @{r['Right Book Close (BS)']}" if r["Right Book Close (BS)"] else "")
            for r in right_rows)
    return out


# ===========================
# DERIVED RATIOS
# ===========================
def calc_roe(row):
    eps, bvps = row.get("EPS (Trailing)"), row.get("Book Value per Share (Latest)")
    if pd.notna(eps) and pd.notna(bvps) and bvps != 0:
        return round(eps / bvps, 4)
    return None


def calc_de(row):
    liab, assets = row.get("Total Liabilities (Latest)"), row.get("Total Assets (Latest)")
    if pd.notna(liab) and pd.notna(assets):
        equity = assets - liab
        if equity != 0:
            return round(liab / equity, 4)
    return None


# ===========================
# PER-TICKER JOB
# ===========================
def scrape_ticker(ticker):
    vals, status = None, "error"
    for attempt in (1, 2):                                # 2nd attempt uses a brand-new browser
        try:
            vals, status = read_onlinekhabar(thread_driver(), ticker)
        except WebDriverException:
            vals, status = None, "error"
        if status in ("ok", "partial", "empty"):
            break
        reset_thread_driver()

    data = {"Ticker": ticker}
    if vals and status != "mismatch":
        data.update(vals)
    data["Scrape Status"] = status
    price = first_number(data.get("Today's Price"))

    div_rows, dstatus = [], "skipped"
    if USE_NEPALIPAISA:
        for attempt in (1, 2):
            try:
                div_rows, dstatus = scrape_dividends(thread_driver(), ticker)
            except WebDriverException:
                div_rows, dstatus = [], "error"
            if dstatus in ("ok", "empty"):
                break
            reset_thread_driver()
    data["Dividend Scrape Status"] = dstatus
    data.update(summarize_dividends(div_rows, price))
    data["Scraped At"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    _local.count = getattr(_local, "count", 0) + 1
    time.sleep(PER_TICKER_PAUSE)
    print(f"{ticker}: page={status} dividends={dstatus} ({len(div_rows)} rows)", flush=True)
    return data, [{"Ticker": ticker, **r} for r in div_rows]


# ===========================
# TICKER LIST
# ===========================
def tickers_from_nepse():
    last = None
    for attempt in range(1, 4):
        try:
            tp = Nepse_scraper(verify_ssl=False).get_today_price()
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
    try:
        r = requests.get(f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/Fundamental",
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


# ===========================
# MAIN
# ===========================
if __name__ == "__main__":
    print("Fetching live tickers from NEPSE...")
    ticker_list = tickers_from_nepse() or tickers_from_repo()
    if not ticker_list:
        raise RuntimeError("No tickers available from NEPSE or from the repo's previous Fundamental file")
    print(f"Found {len(ticker_list)} active tickers.\n")

    USE_NEPALIPAISA = allowed_by_robots(NEPALIPAISA_URL, "/company/UNL")
    if not USE_NEPALIPAISA:
        print("robots.txt disallows nepalipaisa company pages - skipping dividend scrape.")

    DRIVER_PATH = ChromeDriverManager().install()
    t_start = time.time()

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
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)
    df["ROE"] = df.apply(calc_roe, axis=1)
    df["D/E Ratio"] = df.apply(calc_de, axis=1)

    print(f"\nRuntime: {(time.time() - t_start) / 60:.1f} min")
    print("Page status:\n", df["Scrape Status"].value_counts().to_string())
    print("Dividend status:\n", df["Dividend Scrape Status"].value_counts().to_string())
    print(f"ROE {df['ROE'].notna().sum()}/{len(df)}  D/E {df['D/E Ratio'].notna().sum()}/{len(df)}  "
          f"with dividend data {(df['Dividend Records'] > 0).sum()}/{len(df)}")

    fund_csv = df.to_csv(index=False)
    save_local(FUNDAMENTAL_FILE, fund_csv)
    upload_to_github(FUNDAMENTAL_FILE, fund_csv)

    if dividend_rows:
        div_csv = pd.DataFrame(dividend_rows).to_csv(index=False)
        save_local(DIVIDEND_FILE, div_csv)
        upload_to_github(DIVIDEND_FILE, div_csv)

    print(f"\nCompleted! {len(df)} stocks -> {FUNDAMENTAL_FILE}; "
          f"{len(dividend_rows)} dividend/right rows -> {DIVIDEND_FILE}")
