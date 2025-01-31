import subprocess
import sys
import os
import requests
import base64
import nepse_scraper
import git
import pandas as pd
from datetime import datetime
import numpy as np
from joblib import Parallel, delayed

# Install required packages
packages = ["nepse-scraper", "xlsxwriter", "gitpython", "pandas", "matplotlib", "joblib"]
subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

# Import after installation
from nepse_scraper import Nepse_scraper
import matplotlib.pyplot as plt

# GitHub Configuration
GH_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = "iamsrijit"
REPO_NAME = "Nepse"

def upload_to_github(file_name, dataframe):
    try:
        # Convert DataFrame to CSV and encode
        csv_data = dataframe.to_csv(index=False)
        csv_b64 = base64.b64encode(csv_data.encode()).decode()
        
        # Prepare API request
        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{file_name}"
        headers = {
            "Authorization": f"token {GH_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Check if file exists to get SHA
        sha = None
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            sha = response.json().get("sha")

        # Create payload
        payload = {
            "message": f"Update {file_name}",
            "content": csv_b64,
            "branch": "main",
        }
        if sha:
            payload["sha"] = sha

        # Upload file
        response = requests.put(url, headers=headers, json=payload)
        if response.status_code in [200, 201]:
            print(f"Successfully uploaded {file_name}")
        else:
            print(f"Failed to upload {file_name}: {response.text}")
    except Exception as e:
        print(f"Error uploading {file_name}: {str(e)}")

# Main processing function
def main():
    # Scrape and process data
    scraper = Nepse_scraper()
    today_price = scraper.get_today_price()
    content_data = today_price.get('content', [])
    
    # Process today's data
    filtered_data = []
    for item in content_data:
        filtered_data.append({
            'Symbol': item.get('symbol', ''),
            'Date': item.get('businessDate', ''),
            'Open': item.get('openPrice', 0),
            'High': item.get('highPrice', 0),
            'Low': item.get('lowPrice', 0),
            'Close': item.get('closePrice', 0),
            'Percent Change': round(((item.get('closePrice', 0) - item.get('openPrice', 0)) / item.get('openPrice', 0) * 100), 2) if item.get('openPrice', 0) else 0,
            'Volume': item.get('totalTradedValue', 0)
        })
    
    today_df = pd.DataFrame(filtered_data)
    
    # Get historical data
    repo_url = 'https://github.com/iamsrijit/Nepse'
    historical_df = pd.read_csv(f"{repo_url}/main/espen_latest.csv", parse_dates=['Date'])
    
    # Merge data
    combined_df = pd.concat([today_df, historical_df], ignore_index=True)
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    
    # Filter and clean data
    exclude_symbols = ['SAEF', 'SEF', 'CMF1', 'NICGF', 'NBF2', 'CMF2', 'NMB50', 
                      'SIGS2', 'NICBF', 'SFMF', 'LUK', 'SLCF', 'KEF', 'SBCF', 
                      'PSF', 'NIBSF2', 'NICSF', 'RMF1', 'NBF3', 'MMF1', 'KDBY', 
                      'NICFC', 'GIBF1', 'NSIF2', 'SAGF', 'NIBLGF', 'SFEF', 
                      'PRSF', 'C30MF', 'SIGS3', 'RMF2', 'LVF2', 'H8020', 
                      'NICGF2', 'NIBLSTF', 'KSY','NBLD87', 'PBD88', 'OTHERS',
                      'HIDCLP','NIMBPO','MUTUAL','CIT','ILI','LEMF','NIBLPF',
                      'INVESTMENT','SENFLOAT','HEIP','SBID83','NICAD8283']
    
    final_df = combined_df[~combined_df['Symbol'].isin(exclude_symbols)]
    final_df = final_df.sort_values(['Symbol', 'Date']).drop_duplicates(['Symbol', 'Date'], keep='last')
    
    # Calculate technical indicators
    def process_symbol(group):
        group = group.sort_values('Date')
        group['EMA_20'] = group['Close'].ewm(span=20, adjust=False).mean()
        group['EMA_50'] = group['Close'].ewm(span=50, adjust=False).mean()
        group['RSI'] = calculate_rsi(group)
        group['30D_Avg_Volume'] = group['Volume'].rolling(30).mean()
        group['Crossover'] = (group['EMA_20'] > group['EMA_50']) & (group['EMA_20'].shift(1) <= group['EMA_50'].shift(1))
        return group
    
    def calculate_rsi(df, period=14):
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    # Parallel processing
    results = Parallel(n_jobs=-1)(
        delayed(process_symbol)(group) 
        for _, group in final_df.groupby('Symbol')
    )
    
    processed_df = pd.concat(results)
    valid_crossovers = processed_df[
        (processed_df['Crossover']) &
        (processed_df['RSI'].between(30, 70)) &
        (processed_df['Volume'] >= 0.3 * processed_df['30D_Avg_Volume'])
    ]
    
    # Prepare files for upload
    date_str = datetime.today().strftime("%Y-%m-%d")
    ema_file = f"EMA_Cross_for_{date_str}.csv"
    espen_file = f"espen_{date_str}.csv"
    
    # Upload files
    upload_to_github(ema_file, valid_crossovers)
    upload_to_github(espen_file, final_df)
    
    # Cleanup old files
    repo_dir = "nepse_repo"
    if not os.path.exists(repo_dir):
        Repo.clone_from(f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git", repo_dir)
    
    repo = Repo(repo_dir)
    repo.config_writer().set_value("user", "name", "github-actions[bot]").release()
    repo.config_writer().set_value("user", "email", "github-actions@users.noreply.github.com").release()
    
    # Delete old files (keep last 3 days)
    all_files = os.listdir(repo_dir)
    patterns = [r'^EMA_Cross_for_\d{4}-\d{2}-\d{2}\.csv$', 
                r'^espen_\d{4}-\d{2}-\d{2}\.csv$']
    
    for pattern in patterns:
        matched = sorted([f for f in all_files if re.match(pattern, f)], reverse=True)
        for f in matched[3:]:
            os.remove(os.path.join(repo_dir, f))
            repo.index.remove([f], working_tree=True)
    
    # Commit and push changes
    repo.git.add(A=True)
    repo.index.commit(f"Auto-update: {date_str}")
    origin = repo.remote(name='origin')
    origin.push()

if __name__ == "__main__":
    main()
