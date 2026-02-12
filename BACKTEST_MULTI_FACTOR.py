# -*- coding: utf-8 -*-
"""
MULTI-FACTOR STRATEGY BACKTESTER
Tests the strategy on historical data to see real performance
"""
import os
import re
import base64
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, timedelta

# ===========================
# CONFIG
# ===========================
REPO_OWNER = "iamsrijit"
REPO_NAME = "Nepse"
BRANCH = "main"

EXCLUDED_SYMBOLS = [
    "EBLD852", "EBL", "EB89", "NABILD2089", "MBLD2085", "SBID89",
    "SBID2090", "SBLD2091", "NIMBD90", "RBBD2088", "CCBD88", "ULBSL",
    "ICFCD88", "EBLD91", "ANLB", "GBILD84/85", "GBILD86/87", "NICD88"
]

# BACKTEST SETTINGS
INITIAL_CAPITAL = 100000  # Starting capital
POSITION_SIZE_PCT = 0.05  # 5% per position (for score 70-79)
MAX_POSITIONS = 5  # Maximum concurrent positions
STOP_LOSS_PCT = 0.08  # 8% stop loss
TAKE_PROFIT_1 = 0.10  # 10% first target (sell 50%)
TAKE_PROFIT_2 = 0.20  # 20% second target (sell remaining)
MIN_SCORE = 70  # Only trade signals with score >= 70

GH_TOKEN = os.environ.get("GH_TOKEN")
if not GH_TOKEN:
    raise RuntimeError("GH_TOKEN not set in environment")

HEADERS = {"Authorization": f"token {GH_TOKEN}"}

# ===========================
# HELPER FUNCTIONS
# ===========================
def github_raw(path):
    return f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{path}"

def get_latest_espen_csv():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents"
    r = requests.get(url, headers=HEADERS, params={"ref": BRANCH})
    r.raise_for_status()
    espen_files = {}
    for f in r.json():
        name = f["name"]
        if name.startswith("espen_") and name.endswith(".csv"):
            m = re.search(r"espen_(\d{4}-\d{2}-\d{2})\.csv", name)
            if m:
                espen_files[m.group(1)] = name
    if not espen_files:
        raise FileNotFoundError("No espen_*.csv file found")
    latest_date = max(espen_files.keys())
    latest_file = espen_files[latest_date]
    print(f"📂 Using: {latest_file}")
    return github_raw(latest_file)

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_score(row, sym_data_full):
    """Calculate multi-factor score for a specific date"""
    score = 0
    signals = []
    
    # Get data up to this date
    current_data = sym_data_full[sym_data_full['Date'] <= row['Date']].copy()
    if len(current_data) < 200:
        return None, []
    
    # Calculate all indicators
    current_data['RSI'] = calculate_rsi(current_data['Close'], 14)
    current_data['EMA_9'] = calculate_ema(current_data['Close'], 9)
    current_data['EMA_21'] = calculate_ema(current_data['Close'], 21)
    current_data['EMA_50'] = calculate_ema(current_data['Close'], 50)
    current_data['SMA_200'] = current_data['Close'].rolling(window=200).mean()
    current_data['BB_Upper'], current_data['BB_Middle'], current_data['BB_Lower'] = calculate_bollinger_bands(current_data['Close'])
    
    if 'Volume' in current_data.columns:
        current_data['Volume_SMA'] = current_data['Volume'].rolling(window=20).mean()
        current_data['Volume_Ratio'] = current_data['Volume'] / current_data['Volume_SMA']
    
    current_data['52W_High'] = current_data['Close'].rolling(window=252, min_periods=50).max()
    current_data['52W_Low'] = current_data['Close'].rolling(window=252, min_periods=50).min()
    current_data['Momentum_5'] = current_data['Close'].pct_change(5) * 100
    current_data['Momentum_20'] = current_data['Close'].pct_change(20) * 100
    
    latest = current_data.iloc[-1]
    
    if pd.isna(latest['RSI']) or pd.isna(latest['EMA_50']):
        return None, []
    
    # FACTOR 1: RSI
    if latest['RSI'] < 30:
        score += 15
        signals.append("Oversold_RSI")
    elif latest['RSI'] < 40:
        score += 10
        signals.append("Low_RSI")
    elif latest['RSI'] > 70:
        score -= 15
    
    # FACTOR 2: EMA Trend
    if latest['EMA_9'] > latest['EMA_21'] > latest['EMA_50']:
        score += 20
        signals.append("Strong_Uptrend")
    elif latest['EMA_9'] > latest['EMA_21']:
        score += 10
        signals.append("Weak_Uptrend")
    elif latest['EMA_9'] < latest['EMA_21'] < latest['EMA_50']:
        score -= 20
    
    # FACTOR 3: 200 SMA
    if not pd.isna(latest['SMA_200']):
        if latest['Close'] > latest['SMA_200']:
            score += 15
            signals.append("Above_200SMA")
        else:
            score -= 10
    
    # FACTOR 4: 52W position
    if not pd.isna(latest['52W_Low']) and not pd.isna(latest['52W_High']):
        range_52w = latest['52W_High'] - latest['52W_Low']
        if range_52w > 0:
            position_in_range = (latest['Close'] - latest['52W_Low']) / range_52w
            if position_in_range < 0.15:
                score += 15
                signals.append("Near_52W_Low")
            elif position_in_range < 0.30:
                score += 10
            elif position_in_range > 0.85:
                score -= 10
    
    # FACTOR 5: Bollinger Bands
    if not pd.isna(latest['BB_Lower']) and not pd.isna(latest['BB_Upper']):
        if latest['Close'] < latest['BB_Lower']:
            score += 10
            signals.append("Below_BB")
        elif latest['Close'] < latest['BB_Middle']:
            score += 5
    
    # FACTOR 6: Volume
    if 'Volume_Ratio' in current_data.columns and not pd.isna(latest['Volume_Ratio']):
        if latest['Volume_Ratio'] > 1.5:
            score += 10
            signals.append("High_Volume")
        elif latest['Volume_Ratio'] > 1.2:
            score += 5
    
    # FACTOR 7: Momentum
    if not pd.isna(latest['Momentum_5']):
        if latest['Momentum_5'] > 5:
            score += 5
        elif latest['Momentum_5'] < -5:
            score += 10
    
    if not pd.isna(latest['Momentum_20']):
        if latest['Momentum_20'] > 10:
            score += 10
        elif latest['Momentum_20'] < -15:
            score += 5
    
    return score, signals

# ===========================
# LOAD DATA
# ===========================
print("="*70)
print("🔬 MULTI-FACTOR STRATEGY BACKTEST")
print("="*70)

csv_url = get_latest_espen_csv()
response = requests.get(csv_url)
csv_content = response.text

lines = csv_content.strip().split('\n')
header_line = None
data_start_index = 0

for i, line in enumerate(lines):
    if 'Symbol' in line and 'Date' in line and 'Close' in line:
        header_line = line
        data_start_index = i
        break

if data_start_index > 0:
    reconstructed_csv = header_line + '\n' + '\n'.join(lines[:data_start_index])
else:
    reconstructed_csv = csv_content

try:
    df = pd.read_csv(StringIO(reconstructed_csv), sep='\t')
    if len(df.columns) == 1:
        df = pd.read_csv(StringIO(reconstructed_csv), sep=',')
except:
    df = pd.read_csv(StringIO(reconstructed_csv), sep=',')

df.columns = df.columns.str.strip()

original_dates = df["Date"].copy()
df["Date"] = pd.to_datetime(df["Date"], format='%m/%d/%Y', errors='coerce')
if df["Date"].isna().all():
    df["Date"] = pd.to_datetime(original_dates, format='%Y-%m-%d', errors='coerce')
if df["Date"].isna().all():
    df["Date"] = pd.to_datetime(original_dates, errors='coerce')

df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
if 'Volume' in df.columns:
    df["Volume"] = pd.to_numeric(df["Volume"], errors='coerce')

df = df.dropna(subset=["Symbol", "Date", "Close"])
df = df.sort_values(["Symbol", "Date"])

print(f"📊 Data loaded: {len(df)} rows, {df['Symbol'].nunique()} symbols")
print(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# ===========================
# BACKTEST PARAMETERS
# ===========================
print(f"\n⚙️ BACKTEST SETTINGS:")
print(f"   Initial Capital: NPR {INITIAL_CAPITAL:,}")
print(f"   Position Size: {POSITION_SIZE_PCT*100}% per trade")
print(f"   Max Positions: {MAX_POSITIONS}")
print(f"   Min Score: {MIN_SCORE}")
print(f"   Stop Loss: {STOP_LOSS_PCT*100}%")
print(f"   Take Profit: {TAKE_PROFIT_1*100}% (50%), {TAKE_PROFIT_2*100}% (50%)")

# ===========================
# RUN BACKTEST
# ===========================
print("\n🔄 Running backtest...")

# Backtest from 1 year ago to today
backtest_start = df['Date'].max() - pd.Timedelta(days=365)
backtest_end = df['Date'].max()

print(f"   Period: {backtest_start.date()} to {backtest_end.date()}")

# Track portfolio
cash = INITIAL_CAPITAL
positions = {}  # {symbol: {'qty': X, 'entry_price': Y, 'entry_date': Z, 'score': S}}
closed_trades = []

# Get all unique dates in backtest period
backtest_dates = sorted(df[(df['Date'] >= backtest_start) & (df['Date'] <= backtest_end)]['Date'].unique())

for current_date in backtest_dates:
    # Check existing positions for stop loss / take profit
    positions_to_close = []
    
    for symbol, pos in positions.items():
        # Get current price
        sym_today = df[(df['Symbol'] == symbol) & (df['Date'] == current_date)]
        if len(sym_today) == 0:
            continue
        
        current_price = sym_today.iloc[0]['Close']
        entry_price = pos['entry_price']
        pnl_pct = (current_price - entry_price) / entry_price
        
        # Check stop loss
        if pnl_pct <= -STOP_LOSS_PCT:
            exit_reason = "STOP_LOSS"
            sell_qty = pos['qty']
            positions_to_close.append((symbol, current_price, sell_qty, exit_reason))
        
        # Check take profit 1 (sell 50% at +10%)
        elif pnl_pct >= TAKE_PROFIT_1 and pos['qty'] > 0:
            exit_reason = "TAKE_PROFIT_1"
            sell_qty = pos['qty'] // 2  # Sell half
            if sell_qty > 0:
                positions_to_close.append((symbol, current_price, sell_qty, exit_reason))
        
        # Check take profit 2 (sell rest at +20%)
        elif pnl_pct >= TAKE_PROFIT_2:
            exit_reason = "TAKE_PROFIT_2"
            sell_qty = pos['qty']
            positions_to_close.append((symbol, current_price, sell_qty, exit_reason))
    
    # Close positions
    for symbol, exit_price, sell_qty, reason in positions_to_close:
        pos = positions[symbol]
        entry_price = pos['entry_price']
        pnl = (exit_price - entry_price) * sell_qty
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        
        cash += exit_price * sell_qty
        
        closed_trades.append({
            'Symbol': symbol,
            'Entry_Date': pos['entry_date'],
            'Entry_Price': entry_price,
            'Exit_Date': current_date,
            'Exit_Price': exit_price,
            'Qty': sell_qty,
            'PnL': pnl,
            'PnL_%': pnl_pct,
            'Score': pos['score'],
            'Exit_Reason': reason
        })
        
        # Update or remove position
        pos['qty'] -= sell_qty
        if pos['qty'] <= 0:
            del positions[symbol]
    
    # Look for new signals (only if we have room for more positions)
    if len(positions) < MAX_POSITIONS:
        # Get all symbols with data today
        today_data = df[df['Date'] == current_date]
        
        for symbol in today_data['Symbol'].unique():
            if symbol in EXCLUDED_SYMBOLS or symbol in positions:
                continue
            
            # Get full historical data for this symbol
            sym_data_full = df[df['Symbol'] == symbol].copy()
            sym_row = today_data[today_data['Symbol'] == symbol].iloc[0]
            
            # Calculate score
            score, signals = calculate_score(sym_row, sym_data_full)
            
            if score is None or score < MIN_SCORE:
                continue
            
            # We have a signal! Enter position
            if len(positions) < MAX_POSITIONS:
                entry_price = sym_row['Close']
                position_value = cash * POSITION_SIZE_PCT
                qty = int(position_value / entry_price)
                
                if qty > 0 and qty * entry_price <= cash:
                    positions[symbol] = {
                        'qty': qty,
                        'entry_price': entry_price,
                        'entry_date': current_date,
                        'score': score
                    }
                    cash -= qty * entry_price

# Close any remaining positions at end of backtest
for symbol, pos in positions.items():
    sym_final = df[(df['Symbol'] == symbol) & (df['Date'] == backtest_end)]
    if len(sym_final) > 0:
        exit_price = sym_final.iloc[0]['Close']
        pnl = (exit_price - pos['entry_price']) * pos['qty']
        pnl_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
        cash += exit_price * pos['qty']
        
        closed_trades.append({
            'Symbol': symbol,
            'Entry_Date': pos['entry_date'],
            'Entry_Price': pos['entry_price'],
            'Exit_Date': backtest_end,
            'Exit_Price': exit_price,
            'Qty': pos['qty'],
            'PnL': pnl,
            'PnL_%': pnl_pct,
            'Score': pos['score'],
            'Exit_Reason': 'END_OF_BACKTEST'
        })

# ===========================
# RESULTS
# ===========================
print("\n" + "="*70)
print("📊 BACKTEST RESULTS")
print("="*70)

trades_df = pd.DataFrame(closed_trades)

if len(trades_df) > 0:
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['PnL'] > 0])
    losing_trades = len(trades_df[trades_df['PnL'] < 0])
    win_rate = (winning_trades / total_trades) * 100
    
    total_pnl = trades_df['PnL'].sum()
    final_capital = cash
    total_return_pct = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    avg_win = trades_df[trades_df['PnL'] > 0]['PnL_%'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['PnL'] < 0]['PnL_%'].mean() if losing_trades > 0 else 0
    
    print(f"\n💰 CAPITAL:")
    print(f"   Initial: NPR {INITIAL_CAPITAL:,.2f}")
    print(f"   Final: NPR {final_capital:,.2f}")
    print(f"   Total Return: {total_return_pct:+.2f}%")
    print(f"   Total P/L: NPR {total_pnl:+,.2f}")
    
    print(f"\n📈 TRADE STATISTICS:")
    print(f"   Total Trades: {total_trades}")
    print(f"   Winning: {winning_trades} ({win_rate:.1f}%)")
    print(f"   Losing: {losing_trades} ({100-win_rate:.1f}%)")
    print(f"   Average Win: {avg_win:+.2f}%")
    print(f"   Average Loss: {avg_loss:+.2f}%")
    
    # Best and worst trades
    best_trade = trades_df.loc[trades_df['PnL_%'].idxmax()]
    worst_trade = trades_df.loc[trades_df['PnL_%'].idxmin()]
    
    print(f"\n🏆 BEST TRADE:")
    print(f"   {best_trade['Symbol']} - Score {best_trade['Score']}")
    print(f"   Entry: NPR {best_trade['Entry_Price']:.2f} on {best_trade['Entry_Date'].date()}")
    print(f"   Exit: NPR {best_trade['Exit_Price']:.2f} on {best_trade['Exit_Date'].date()}")
    print(f"   Return: {best_trade['PnL_%']:+.2f}% (NPR {best_trade['PnL']:+,.2f})")
    
    print(f"\n💔 WORST TRADE:")
    print(f"   {worst_trade['Symbol']} - Score {worst_trade['Score']}")
    print(f"   Entry: NPR {worst_trade['Entry_Price']:.2f} on {worst_trade['Entry_Date'].date()}")
    print(f"   Exit: NPR {worst_trade['Exit_Price']:.2f} on {worst_trade['Exit_Date'].date()}")
    print(f"   Return: {worst_trade['PnL_%']:+.2f}% (NPR {worst_trade['PnL']:+,.2f})")
    
    # Exit reason breakdown
    print(f"\n🚪 EXIT REASONS:")
    for reason in trades_df['Exit_Reason'].unique():
        count = len(trades_df[trades_df['Exit_Reason'] == reason])
        print(f"   {reason}: {count} trades")
    
    # Score analysis
    print(f"\n🎯 SCORE PERFORMANCE:")
    for score_range in [(70, 79), (80, 89), (90, 100)]:
        range_trades = trades_df[(trades_df['Score'] >= score_range[0]) & (trades_df['Score'] <= score_range[1])]
        if len(range_trades) > 0:
            range_win_rate = (len(range_trades[range_trades['PnL'] > 0]) / len(range_trades)) * 100
            range_avg_return = range_trades['PnL_%'].mean()
            print(f"   Score {score_range[0]}-{score_range[1]}: {len(range_trades)} trades, {range_win_rate:.1f}% win rate, {range_avg_return:+.2f}% avg")
    
    # Save detailed results
    trades_df['Entry_Date'] = trades_df['Entry_Date'].dt.strftime('%Y-%m-%d')
    trades_df['Exit_Date'] = trades_df['Exit_Date'].dt.strftime('%Y-%m-%d')
    
    output_file = "BACKTEST_RESULTS.csv"
    trades_df.to_csv(f"/home/claude/{output_file}", index=False)
    print(f"\n📁 Detailed results saved to: {output_file}")
    
    print("\n" + "="*70)
    print("💡 INTERPRETATION:")
    print("="*70)
    if total_return_pct > 15:
        print("✅ EXCELLENT: Strategy significantly outperformed")
    elif total_return_pct > 5:
        print("✅ GOOD: Strategy showed positive returns")
    elif total_return_pct > 0:
        print("⚠️  MARGINAL: Strategy barely profitable")
    else:
        print("❌ POOR: Strategy lost money - needs improvement")
    
    if win_rate >= 60:
        print("✅ High win rate - good signal quality")
    elif win_rate >= 50:
        print("⚠️  Moderate win rate - acceptable")
    else:
        print("❌ Low win rate - signals need improvement")
    
else:
    print("⚠️  No trades were generated in the backtest period")
    print("   Try adjusting MIN_SCORE or expanding the date range")

print("\n" + "="*70)
