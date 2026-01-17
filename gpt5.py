# NEPSE Trading Strategy - Buy Signal Generator
import pandas as pd
import numpy as np
import requests
import base64
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
REPO_OWNER = 'iamsrijit'
REPO_NAME = 'Nepse'
BRANCH = 'main'

# Get GitHub token from environment variable (for GitHub Actions)
GITHUB_TOKEN = os.environ.get('GH_TOKEN')

# Headers for GitHub API
HEADERS = {
    'Accept': 'application/vnd.github.v3+json'
}
if GITHUB_TOKEN:
    HEADERS['Authorization'] = f'token {GITHUB_TOKEN}'

# Trading parameters
ATR_PERIOD = 14
SMA_PERIOD = 50
VOLUME_PERIOD = 20
VOLUME_MULTIPLIER = 1.2
MAX_RECENT_DRAWDOWN = -0.05  # 5% max drawdown threshold
STOP_LOSS_ATR_MULTIPLIER = 2

# ============================================================================
# GITHUB UTILITY FUNCTIONS
# ============================================================================

def github_raw(path):
    """Generate raw GitHub URL for a file"""
    return f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{path}"

def upload_to_github(filename, content):
    """Upload or overwrite a file in GitHub repository"""
    if not GITHUB_TOKEN:
        print(f"⚠️ No GitHub token found, skipping upload of {filename}")
        return False
    
    try:
        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{filename}"
        r = requests.get(url, headers=HEADERS)

        payload = {
            "message": f"Auto-update: {filename} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "content": base64.b64encode(content.encode()).decode(),
            "branch": BRANCH
        }

        if r.status_code == 200:
            payload["sha"] = r.json()["sha"]

        res = requests.put(url, headers=HEADERS, json=payload)
        if res.status_code not in (200, 201):
            print(f"❌ Upload failed: {res.status_code} - {res.text}")
            return False

        print(f"✅ Uploaded to GitHub: {filename}")
        return True
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return False

def delete_old_files(prefix, keep_filename):
    """Delete old files with a given prefix, keeping only the specified file"""
    if not GITHUB_TOKEN:
        print(f"ℹ️ No GitHub token, skipping cleanup")
        return
    
    try:
        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents"
        r = requests.get(url, headers=HEADERS, params={"ref": BRANCH})
        r.raise_for_status()
        
        deleted_count = 0
        for f in r.json():
            name = f["name"]
            if name.startswith(prefix) and name.endswith(".csv") and name != keep_filename:
                del_payload = {
                    "message": f"Cleanup: Delete old {name}",
                    "sha": f["sha"],
                    "branch": BRANCH
                }
                del_url = f"{url}/{name}"
                res = requests.delete(del_url, headers=HEADERS, json=del_payload)

                if res.status_code == 200:
                    print(f"🗑️ Deleted: {name}")
                    deleted_count += 1
                else:
                    print(f"⚠️ Failed to delete {name}")
        
        if deleted_count == 0:
            print(f"ℹ️ No old files to delete")
        else:
            print(f"✅ Cleaned up {deleted_count} old file(s)")
            
    except Exception as e:
        print(f"⚠️ Cleanup failed: {str(e)}")

# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_latest_data_file(prefix='espen_2026'):
    """Fetch the latest data file from GitHub based on date in filename"""
    api_url = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/'
    response = requests.get(api_url, headers=HEADERS)
    
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch repo contents: {response.text}")

    data = response.json()

    # Filter files starting with prefix
    matching_files = []
    for item in data:
        if item['type'] == 'file' and item['name'].startswith(prefix):
            try:
                # Extract date from filename, e.g., 'espen_2026-01-15.csv' -> '2026-01-15'
                date_str = item['name'].split('_')[1].replace('.csv', '')
                file_date = datetime.strptime(date_str, '%Y-%m-%d')
                matching_files.append((item['name'], file_date))
            except (IndexError, ValueError):
                continue

    if not matching_files:
        raise ValueError(f"No files found starting with '{prefix}' in the repo.")

    # Sort by date descending and pick the latest
    matching_files.sort(key=lambda x: x[1], reverse=True)
    latest_filename = matching_files[0][0]
    
    print(f"📊 Latest data file: {latest_filename}")
    return latest_filename

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def calculate_atr(df, period=14):
    """Calculate Average True Range (ATR)"""
    df = df.copy()
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift())
    df['low_close'] = np.abs(df['low'] - df['close'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    return df

def calculate_indicators(df):
    """Calculate all technical indicators"""
    df = calculate_atr(df, period=ATR_PERIOD)
    df['sma50'] = df['close'].rolling(window=SMA_PERIOD).mean()
    df['avg_vol20'] = df['volume'].rolling(window=VOLUME_PERIOD).mean()
    df['recent_drawdown'] = df['close'] / df['close'].rolling(5).max() - 1
    return df

# ============================================================================
# TRADING STRATEGY
# ============================================================================

def generate_buy_signals(df):
    """
    Generate buy signals based on Trend + Momentum Hybrid strategy:
    - Close > 50-day SMA (uptrend)
    - Volume > 1.2 * 20-day Avg Volume (momentum spike)
    - ATR > previous ATR (increasing volatility)
    - Recent drawdown < 5% (avoid buying after sharp drops)
    """
    df['buy_signal'] = (
        (df['close'] > df['sma50']) &
        (df['volume'] > df['avg_vol20'] * VOLUME_MULTIPLIER) &
        (df['atr'] > df['atr'].shift(1)) &
        (df['recent_drawdown'] > MAX_RECENT_DRAWDOWN)
    )
    return df

def calculate_position_sizing(df, buy_dates):
    """Calculate position sizing for each buy signal"""
    positions = []
    
    for date in buy_dates:
        entry_price = df.at[date, 'close']
        atr = df.at[date, 'atr']
        
        # Stop-loss: Entry - (2 * ATR) for volatility-adjusted stop
        stop_loss = entry_price - (STOP_LOSS_ATR_MULTIPLIER * atr)
        stop_distance = (entry_price - stop_loss) / entry_price
        
        if stop_distance <= 0:
            continue
        
        positions.append({
            'date': date.strftime('%Y-%m-%d'),
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'atr': round(atr, 2),
            'stop_distance_pct': round(stop_distance * 100, 2),
            'target_1': round(entry_price + (1.5 * atr), 2),
            'target_2': round(entry_price + (3 * atr), 2),
            'volume': int(df.at[date, 'volume']),
            'avg_volume': int(df.at[date, 'avg_vol20']),
            'volume_ratio': round(df.at[date, 'volume'] / df.at[date, 'avg_vol20'], 2)
        })
    
    return positions

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    try:
        print("=" * 70)
        print("NEPSE TRADING STRATEGY - BUY SIGNAL GENERATOR")
        print("=" * 70)
        
        # Fetch latest data file
        latest_filename = fetch_latest_data_file()
        raw_url = github_raw(latest_filename)
        
        # Read CSV data
        print(f"\n📥 Loading data from: {raw_url}")
        df = pd.read_csv(raw_url)
        
        # Preprocess data
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col.upper()}")
        
        print(f"✅ Data loaded: {len(df)} rows from {df.index.min().date()} to {df.index.max().date()}")
        
        # Calculate indicators
        print("\n🔍 Calculating technical indicators...")
        df = calculate_indicators(df)
        df = generate_buy_signals(df)
        df = df.dropna()
        
        # Get buy dates
        buy_dates = df[df['buy_signal']].index
        
        print(f"\n📈 Buy Signals Found: {len(buy_dates)}")
        print("=" * 70)
        
        if len(buy_dates) == 0:
            print("❌ No buy signals found in the provided data.")
            
            # Create empty results file
            empty_df = pd.DataFrame(columns=['date', 'entry_price', 'stop_loss', 'atr', 'stop_distance_pct', 'target_1', 'target_2', 'volume', 'avg_volume', 'volume_ratio'])
            timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            results_filename = f"buy_signals_{timestamp}.csv"
            
            # Save locally
            empty_df.to_csv(results_filename, index=False)
            print(f"💾 Saved locally: {results_filename}")
            
            # Upload to GitHub
            print("\n" + "=" * 70)
            print("💾 UPLOADING TO GITHUB")
            print("=" * 70)
            try:
                upload_to_github(results_filename, empty_df.to_csv(index=False))
            except Exception as e:
                print(f"❌ Upload failed: {e}")
            
            print("\n✅ Analysis complete (no signals)!")
            return
        
        # Display buy dates
        print("\n🎯 BUY DATES:")
        for i, date in enumerate(buy_dates, 1):
            print(f"  {i}. {date.strftime('%Y-%m-%d')}")
        
        # Calculate position sizing
        print(f"\n📊 POSITION ANALYSIS")
        print("=" * 70)
        positions = calculate_position_sizing(df, buy_dates)
        
        # Display positions
        for pos in positions:
            print(f"\n📅 Date: {pos['date']}")
            print(f"   Entry Price: ₹{pos['entry_price']}")
            print(f"   Stop Loss: ₹{pos['stop_loss']} ({pos['stop_distance_pct']}% below entry)")
            print(f"   Target 1: ₹{pos['target_1']}")
            print(f"   Target 2: ₹{pos['target_2']}")
            print(f"   ATR: {pos['atr']}")
            print(f"   Volume: {pos['volume']:,} ({pos['volume_ratio']}x avg)")
        
        # Save results to CSV and upload to GitHub
        print("\n" + "=" * 70)
        print("💾 SAVING RESULTS")
        print("=" * 70)
        
        if positions:
            results_df = pd.DataFrame(positions)
            timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            results_filename = f"buy_signals_{timestamp}.csv"
            
            # Save locally first
            results_df.to_csv(results_filename, index=False)
            print(f"💾 Saved locally: {results_filename}")
            print(f"   File size: {os.path.getsize(results_filename)} bytes")
            print(f"   Rows: {len(results_df)}")
            
            # Upload to GitHub
            print("\n📤 UPLOADING TO GITHUB")
            print("-" * 70)
            try:
                results_csv = results_df.to_csv(index=False)
                print(f"   CSV content length: {len(results_csv)} characters")
                
                success = upload_to_github(results_filename, results_csv)
                
                if success:
                    print("\n🧹 CLEANING UP OLD FILES")
                    print("-" * 70)
                    delete_old_files('buy_signals_', results_filename)
                else:
                    print("\n⚠️ Upload failed, skipping cleanup")
                    
            except Exception as e:
                print(f"❌ Upload error: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("⚠️ No positions to save")
        
        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"📊 Total signals: {len(positions)}")
        print(f"📁 Results file: {results_filename if positions else 'N/A'}")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main() 