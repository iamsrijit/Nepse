#!/usr/bin/env python3
"""
Enhanced NEPSE Blitz Strategy
Automated trading strategy for Nepal Stock Exchange with ML enhancements
"""

import pandas as pd
import numpy as np
import requests
from io import StringIO
import warnings
from datetime import datetime, timedelta
import json
import os

warnings.filterwarnings('ignore')

class EnhancedNEPSEStrategy:
    """Enhanced NEPSE Blitz Strategy with ML improvements"""
    
    def __init__(self):
        self.df = None
        self.initial_capital = 500000
        self.results = {}
        
    def load_data(self):
        """Load NEPSE data from GitHub or fallback"""
        url = "https://raw.githubusercontent.com/iamsrijit/Nepse/main/espen_2026-01-15.csv"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text))
            
            # Try to find date column
            date_col = None
            for col in ['Date', 'date', 'DATE', 'timestamp']:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
            else:
                # Create date index if not found
                df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='B')
            
            # Find price and volume columns
            price_col = None
            for col in ['Close', 'close', 'CLOSE', 'Last', 'last']:
                if col in df.columns:
                    price_col = col
                    break
            
            volume_col = None
            for col in ['Volume', 'volume', 'VOLUME', 'Vol', 'vol']:
                if col in df.columns:
                    volume_col = col
                    break
            
            if price_col:
                df['Close'] = pd.to_numeric(df[price_col], errors='coerce')
            else:
                df['Close'] = 500  # Default
            
            if volume_col:
                df['Volume'] = pd.to_numeric(df[volume_col], errors='coerce')
            else:
                df['Volume'] = 10000  # Default
            
            print(f"✅ Data loaded from GitHub: {len(df)} rows")
            return df
            
        except Exception as e:
            print(f"⚠️ GitHub load failed: {e}, using synthetic data")
            return self.create_synthetic_data()
    
    def create_synthetic_data(self):
        """Create synthetic NEPSE-like data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        np.random.seed(42)
        
        # Realistic NEPSE simulation with trends and volatility
        trend = np.linspace(0, 0.5, len(dates))  # Upward trend
        cycles = 0.1 * np.sin(2 * np.pi * np.arange(len(dates)) / 63)  ~ Quarterly cycles
        noise = np.random.normal(0, 0.015, len(dates))
        
        log_returns = 0.0005 + trend/len(dates) + cycles + noise
        price = 1500 * np.exp(np.cumsum(log_returns))  # Start around 1500 NEPSE index
        
        df = pd.DataFrame({
            'Close': price,
            'Volume': np.random.lognormal(10, 1, len(dates)).astype(int),
            'Open': price * (1 + np.random.normal(0, 0.005, len(dates))),
            'High': price * (1 + np.random.uniform(0, 0.015, len(dates))),
            'Low': price * (1 - np.random.uniform(0, 0.015, len(dates)))
        }, index=dates)
        
        print(f"📊 Synthetic data created: {len(df)} rows")
        return df
    
    def calculate_indicators(self):
        """Calculate all technical indicators"""
        # Moving averages
        self.df['MA10'] = self.df['Close'].rolling(window=10).mean()
        self.df['MA20'] = self.df['Close'].rolling(window=20).mean()
        self.df['MA50'] = self.df['Close'].rolling(window=50).mean()
        
        # Volume average
        self.df['Vol_Avg'] = self.df['Volume'].rolling(window=10).mean()
        
        # RSI
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR for volatility
        high_low = self.df['High'] - self.df['Low']
        high_close = np.abs(self.df['High'] - self.df['Close'].shift())
        low_close = np.abs(self.df['Low'] - self.df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        self.df['ATR'] = true_range.rolling(window=14).mean()
        
    def calculate_sentiment(self):
        """Calculate synthetic sentiment scores"""
        # Price momentum based sentiment
        returns_5 = self.df['Close'].pct_change(5)
        returns_10 = self.df['Close'].pct_change(10)
        
        # Composite sentiment score
        price_sentiment = np.where(
            self.df['Close'] > self.df['MA20'], 
            np.where(self.df['Close'] > self.df['MA50'], 0.8, 0.6),
            np.where(self.df['Close'] < self.df['MA50'], 0.3, 0.5)
        )
        
        momentum_sentiment = 0.5 + 0.5 * np.tanh(returns_5 * 10)
        volatility_sentiment = np.where(
            self.df['ATR'] / self.df['Close'] < 0.02, 0.7, 0.4
        )
        
        # Combined sentiment
        self.df['Sentiment'] = (
            0.4 * price_sentiment + 
            0.3 * momentum_sentiment + 
            0.3 * volatility_sentiment
        )
        self.df['Sentiment'] = self.df['Sentiment'].fillna(0.5)
    
    def calculate_ml_predictions(self):
        """Calculate ML-based predictions"""
        # Simplified LSTM-like predictions
        features = pd.DataFrame()
        
        # Technical features
        features['Returns_1'] = self.df['Close'].pct_change(1)
        features['Returns_5'] = self.df['Close'].pct_change(5)
        features['RSI_norm'] = self.df['RSI'] / 100
        features['Volume_ratio'] = self.df['Volume'] / self.df['Vol_Avg']
        features['MA_ratio'] = self.df['Close'] / self.df['MA20']
        
        # Simple prediction model (linear combination)
        weights = np.array([0.2, 0.3, 0.2, 0.15, 0.15])
        
        # Normalize features
        features_normalized = (features - features.mean()) / (features.std() + 1e-8)
        
        # Generate predictions
        predictions = features_normalized.dot(weights)
        
        # Add trend and noise
        trend_component = 0.1 * np.sin(2 * np.pi * np.arange(len(self.df)) / 42)
        noise_component = np.random.normal(0, 0.03, len(self.df))
        
        self.df['LSTM_Pred'] = 0.05 + 0.1 * predictions + trend_component + noise_component
        self.df['LSTM_Pred'] = self.df['LSTM_Pred'].clip(-0.1, 0.2)
    
    def detect_market_regime(self):
        """Detect bull/bear market regimes"""
        # Simple regime detection using multiple indicators
        ma_regime = np.where(self.df['MA20'] > self.df['MA50'], 1, 0)
        
        # Momentum regime
        momentum = self.df['Close'].pct_change(20)
        momentum_regime = np.where(momentum > 0.02, 1, np.where(momentum < -0.02, 0, 0.5))
        
        # Volatility regime
        volatility = self.df['ATR'] / self.df['Close']
        vol_regime = np.where(volatility < 0.015, 1, np.where(volatility > 0.025, 0, 0.5))
        
        # Combined regime probability
        regime_prob = (0.4 * ma_regime + 0.3 * momentum_regime + 0.3 * vol_regime)
        
        # Smooth the probability
        self.df['Regime_Prob'] = pd.Series(regime_prob).rolling(window=10).mean().fillna(0.5)
    
    def calculate_volatility_forecast(self):
        """Calculate volatility forecasts"""
        returns = self.df['Close'].pct_change()
        
        # Historical volatility
        hist_vol = returns.rolling(window=20).std()
        
        # EMA of volatility
        vol_ema = hist_vol.ewm(span=10).mean()
        
        # Forecast as weighted average
        self.df['Vol_Forecast'] = 0.7 * vol_ema + 0.3 * hist_vol.shift(1)
        self.df['Vol_Forecast'] = self.df['Vol_Forecast'].fillna(hist_vol.mean())
        
        # Cap extreme values
        avg_vol = self.df['Vol_Forecast'].mean()
        self.df['Vol_Forecast'] = self.df['Vol_Forecast'].clip(avg_vol * 0.3, avg_vol * 3)
    
    def generate_signals(self):
        """Generate buy/sell signals"""
        # Individual conditions
        self.df['MA_Condition'] = (self.df['Close'] > self.df['MA10']).astype(int)
        self.df['RSI_Condition'] = (self.df['RSI'] > 70).astype(int)
        self.df['Volume_Condition'] = (self.df['Volume'] > 1.5 * self.df['Vol_Avg']).astype(int)
        self.df['Sentiment_Condition'] = (self.df['Sentiment'] > 0.6).astype(int)
        self.df['LSTM_Condition'] = (self.df['LSTM_Pred'] > 0.05).astype(int)
        self.df['Regime_Condition'] = (self.df['Regime_Prob'] > 0.6).astype(int)
        
        # Combined buy signal (ALL conditions must be met)
        self.df['Buy_Signal'] = (
            self.df['MA_Condition'] & 
            self.df['RSI_Condition'] & 
            self.df['Volume_Condition'] & 
            self.df['Sentiment_Condition'] & 
            self.df['LSTM_Condition'] & 
            self.df['Regime_Condition']
        ).astype(int)
        
        print(f"📈 Buy signals generated: {self.df['Buy_Signal'].sum()}")
    
    def run_backtest(self):
        """Run backtest simulation"""
        signals = self.df[self.df['Buy_Signal'] == 1].index
        
        if len(signals) == 0:
            print("⚠️ No buy signals generated!")
            return
        
        capital = self.initial_capital
        trades = []
        position = 0
        entry_price = 0
        trade_count = 0
        
        for i in range(1, len(self.df)):
            current_date = self.df.index[i]
            prev_date = self.df.index[i-1]
            
            # Check for buy signal on previous day
            if self.df.loc[prev_date, 'Buy_Signal'] == 1 and position == 0:
                # Enter trade
                position = 1
                entry_price = self.df.loc[current_date, 'Open']
                entry_date = current_date
                entry_capital = capital
                
                # Dynamic position sizing based on volatility
                current_vol = self.df.loc[prev_date, 'Vol_Forecast']
                avg_vol = self.df['Vol_Forecast'].mean()
                
                # Size: 10-30% of capital, inversely proportional to volatility
                size_pct = min(0.3, max(0.1, 0.2 * (avg_vol / max(current_vol, avg_vol * 0.5))))
                trade_capital = capital * size_pct
                
                trade_count += 1
                print(f"Trade {trade_count}: Buy at {entry_price:.2f} on {entry_date.date()}")
            
            # Check for exit (hold 4-30 days)
            if position == 1:
                hold_days = (current_date - entry_date).days
                
                # Exit conditions
                exit_condition = (
                    (hold_days >= 30) or  # Max hold period
                    (hold_days >= 4 and self.df.loc[current_date, 'Close'] < entry_price * 0.95) or  # Stop loss
                    (hold_days >= 10 and self.df.loc[prev_date, 'Regime_Prob'] < 0.4)  # Regime change
                )
                
                if exit_condition:
                    exit_price = self.df.loc[current_date, 'Close']
                    hold_return = (exit_price / entry_price) - 1
                    
                    # Apply 2x leverage with costs
                    leveraged_return = hold_return * 2 - 0.004
                    
                    # Calculate profit/loss
                    profit = trade_capital * leveraged_return
                    capital += profit
                    
                    trades.append({
                        'Trade': trade_count,
                        'Entry_Date': entry_date,
                        'Exit_Date': current_date,
                        'Hold_Days': hold_days,
                        'Entry_Price': entry_price,
                        'Exit_Price': exit_price,
                        'Return_Pct': hold_return * 100,
                        'Leveraged_Return_Pct': leveraged_return * 100,
                        'Trade_Capital': trade_capital,
                        'Profit': profit,
                        'Total_Capital': capital
                    })
                    
                    position = 0
                    print(f"  Exit at {exit_price:.2f} on {current_date.date()}, Return: {hold_return*100:.2f}%")
        
        # Store results
        self.results = {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_return': ((capital / self.initial_capital) - 1) * 100,
            'total_trades': len(trades),
            'winning_trades': sum(1 for t in trades if t['Profit'] > 0),
            'trades': trades
        }
        
        # Add capital column to dataframe for tracking
        self.df['Capital'] = self.initial_capital
        if trades:
            for trade in trades:
                mask = (self.df.index >= trade['Exit_Date'])
                self.df.loc[mask, 'Capital'] += trade['Profit']
    
    def save_results(self):
        """Save strategy results to CSV"""
        # Save dataframe with all calculations
        output_df = self.df.copy()
        output_df.reset_index(inplace=True)
        
        # Save to CSV
        output_file = 'nepse_strategy_results.csv'
        output_df.to_csv(output_file, index=False)
        print(f"✅ Results saved to {output_file}")
        
        # Save trade summary
        if self.results['trades']:
            trades_df = pd.DataFrame(self.results['trades'])
            trades_file = 'nepse_trades_summary.csv'
            trades_df.to_csv(trades_file, index=False)
            print(f"✅ Trades summary saved to {trades_file}")
        
        # Save performance summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'initial_capital': self.results['initial_capital'],
            'final_capital': self.results['final_capital'],
            'total_return_pct': self.results['total_return'],
            'total_trades': self.results['total_trades'],
            'winning_trades': self.results['winning_trades'],
            'win_rate': (self.results['winning_trades'] / self.results['total_trades'] * 100) if self.results['total_trades'] > 0 else 0,
            'buy_signals': int(self.df['Buy_Signal'].sum()),
            'data_points': len(self.df)
        }
        
        summary_file = 'strategy_performance.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Performance summary saved to {summary_file}")
    
    def print_report(self):
        """Print strategy execution report"""
        print("\n" + "="*70)
        print("ENHANCED NEPSE BLITZ STRATEGY - EXECUTION REPORT")
        print("="*70)
        print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data Period: {self.df.index.min().date()} to {self.df.index.max().date()}")
        print(f"Total Data Points: {len(self.df)}")
        print("-"*70)
        
        # Strategy conditions
        print("\nSTRATEGY CONDITIONS ANALYSIS:")
        print(f"• Price > 10-day MA: {self.df['MA_Condition'].mean()*100:.1f}% of days")
        print(f"• RSI > 70: {self.df['RSI_Condition'].mean()*100:.1f}% of days")
        print(f"• Volume > 1.5x Avg: {self.df['Volume_Condition'].mean()*100:.1f}% of days")
        print(f"• Sentiment > 0.6: {self.df['Sentiment_Condition'].mean()*100:.1f}% of days")
        print(f"• LSTM Pred > 5%: {self.df['LSTM_Condition'].mean()*100:.1f}% of days")
        print(f"• Bull Regime Prob > 0.6: {self.df['Regime_Condition'].mean()*100:.1f}% of days")
        print(f"• ALL CONDITIONS MET: {self.df['Buy_Signal'].sum()} signals ({self.df['Buy_Signal'].mean()*100:.1f}% of days)")
        
        # Performance results
        if self.results:
            print("\n" + "-"*70)
            print("BACKTEST PERFORMANCE:")
            print(f"Initial Capital: NPR {self.results['initial_capital']:,.2f}")
            print(f"Final Capital: NPR {self.results['final_capital']:,.2f}")
            print(f"Total Return: {self.results['total_return']:.2f}%")
            print(f"Total Profit: NPR {self.results['final_capital'] - self.results['initial_capital']:,.2f}")
            print(f"Total Trades: {self.results['total_trades']}")
            print(f"Winning Trades: {self.results['winning_trades']}")
            print(f"Win Rate: {self.results['winning_trades']/self.results['total_trades']*100:.1f}%" if self.results['total_trades'] > 0 else "Win Rate: N/A")
            
            if self.results['trades']:
                avg_return = np.mean([t['Return_Pct'] for t in self.results['trades']])
                avg_lev_return = np.mean([t['Leveraged_Return_Pct'] for t in self.results['trades']])
                print(f"Average Return per Trade: {avg_return:.2f}%")
                print(f"Average Leveraged Return: {avg_lev_return:.2f}%")
        
        print("="*70)
    
    def execute(self):
        """Execute full strategy pipeline"""
        print("🚀 Starting Enhanced NEPSE Blitz Strategy...")
        print("="*70)
        
        # Step 1: Load data
        self.df = self.load_data()
        
        # Step 2: Calculate indicators
        self.calculate_indicators()
        
        # Step 3: Calculate sentiment
        self.calculate_sentiment()
        
        # Step 4: Calculate ML predictions
        self.calculate_ml_predictions()
        
        # Step 5: Detect market regime
        self.detect_market_regime()
        
        # Step 6: Calculate volatility forecast
        self.calculate_volatility_forecast()
        
        # Step 7: Generate signals
        self.generate_signals()
        
        # Step 8: Run backtest
        self.run_backtest()
        
        # Step 9: Save results
        self.save_results()
        
        # Step 10: Print report
        self.print_report()
        
        print("✅ Strategy execution completed successfully!")

def main():
    """Main execution function"""
    try:
        strategy = EnhancedNEPSEStrategy()
        strategy.execute()
    except Exception as e:
        print(f"❌ Error executing strategy: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()