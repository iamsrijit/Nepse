import subprocess
import sys
import os, requests
import base64
import nepse_scraper
import xlsxwriter
import git
import pandas as pd
from nepse_scraper import Nepse_scraper
import pandas as pd
from datetime import datetime
import pandas as pd
import numpy as np



# List of required packages
packages = ["nepse-scraper", "xlsxwriter", "gitpython", "pandas","matplotlib","joblib"]

# Install missing packages
subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
# Now import the installed packages

# file_name = f"nepse_{datetime.today().strftime('%Y-%m-%d')}.csv"

# try:
#     # Convert finall_df to CSV format
#     csv_data = finall_df.to_csv(index=False)

#     # Encode the CSV data to Base64
#     csv_data_base64 = base64.b64encode(csv_data.encode()).decode()

#     # Define the GitHub repository URL
#     repo_url = 'https://github.com/iamsrijit/Nepse'

#     # Define the file name with today's date
#     # file_name = f"nepse_{datetime.today().strftime('%Y-%m-%d')}.csv"

#     # GitHub API request payload
#     payload = {
#         'message': 'Uploading file via automation',
#         'content': csv_data_base64,  # Corrected variable name
#         'branch': 'main'  # Specify the branch
#     }

# except Exception as e:
#     print(f"An error occurred: {e}")  # Handle the exception

# file_path = file_name
# upload_url = f'https://api.github.com/repos/iamsrijit/Nepse/contents/{file_path}'
# # upload_url = f'https://api.github.com/repos/iamsrijit/Nepse/contents{file_path}'

# headers = {
#     'Authorization': f'token {GH_TOKEN}',
#     'Accept': 'application/vnd.github.v3+json'
# }

token = os.getenv("GH_TOKEN")


# response = requests.put(upload_url, headers=headers, json=payload)


# Your script logic goes here
# print("All required packages are installed and imported successfully!")

# try:
#     # Convert finall_df to CSV format
#     csv_data = finall_df.to_csv(index=False)

#     # Encode the CSV data to Base64
#     csv_data_base64 = base64.b64encode(csv_data.encode()).decode()

#     # csv_data_base64 = base64.b64encode(csv_data.encode()).decode()

#     # Define the GitHub repository URL
#     repo_url = 'https://github.com/iamsrijit/Nepse'

#     # Define the file name with today's date
#     # file_name = f'espen_{datetime.today().strftime("%Y-%m-%d")}.csv'
#     file_name = f"nepse_{datetime.today().strftime('%Y-%m-%d')}.csv"

# # !pip install nepse-scraper
# # !pip install xlsxwriter
# # !pip install gitpython
# # !pip install gitpython pandas

# # """**Daily Nepse Scrapping**"""


# # Encode CSV data in Base64
# # csv_data_base64 = base64.b64encode(csv_data.encode()).decode()

# # GitHub API request payload
# data = {
#     'message': 'Uploading file via automation',
#     'content': csv_data_base64,  # Corrected variable name
#     'branch': 'main'  # Specify the branch
# }

# Upload the file to GitHub
# response = requests.put(upload_url, headers=headers, json=data)
request_obj = Nepse_scraper(verify_ssl=False)
today_price = request_obj.get_today_price()

if isinstance(today_price, dict):
    content_data = today_price.get('content', [])
else:
    content_data = today_price






# # Create an object from the Nepse_scraper class
# # request_obj = Nepse_scraper()

# # # Get today's price from NEPSE
# # today_price = request_obj.get_today_price()

# # Extract the 'content' section from the response
# content_data = today_price.get('content', [])

# Initialize an empty list to store filtered data
filtered_data = []

# Define the column names


columns = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Percent Change', 'Volume',"52High","52Low"]
# Iterate over each item in the 'content' section
def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0

for item in content_data:
    symbol = item.get('symbol', '')
    date = item.get('businessDate', '')

    open_price  = to_float(item.get('openPrice'))
    high_price  = to_float(item.get('highPrice'))
    low_price   = to_float(item.get('lowPrice'))
    close_price = to_float(item.get('closePrice'))
    volume_daily = to_float(item.get('totalTradedQuantity'))
    fiftyTwoWeekHigh=to_float(item.get('fiftyTwoWeekHigh'))
    fiftyTwoWeekLow=to_float(item.get('fiftyTwoWeekLow'))

    percent_change = ((close_price - open_price) / open_price * 100) if open_price > 0 else 0

    filtered_data.append({
        'Symbol': symbol,
        'Date': date,
        'Open': open_price,
        'High': high_price,
        'Low': low_price,
        'Close': close_price,
        'Percent Change': round(percent_change, 2),
        'Volume': volume_daily,
        '52High':fiftyTwoWeekHigh,
        '52Low':fiftyTwoWeekLow
    })


# Create DataFrame from filtered data
first = pd.DataFrame(filtered_data)

# Check if DataFrame has data
if not first.empty:
    # Display DataFrame
    print(first)

    # Get today's day name
    today_day_name = datetime.now().strftime('%A')

    # Save DataFrame to CSV with today's day name in the filename
    file_name = f"nepse_{today_day_name}.csv"
    first.to_csv(file_name, index=False)

    print(f"Data saved to '{file_name}'")
else:
    print("No data available to create DataFrame.")

first.head()

import os

# # Get GitHub token from environment variable
# GH_TOKEN = os.getenv("GH_TOKEN")

# if not GH_TOKEN:
#     raise ValueError("GitHub Token not found. Please set it as an environment variable.")



    

    
    
    # Define the headers with authentication token





    # Make the POST request to GitHub API
    # response = requests.put(upload_url, headers=headers, json=data)

    # Check for successful response
    # if response.status_code == 201:
    #     print(f"File uploaded successfully to {file_path}")
    # else:
    #     print(f"Failed to upload file: {response.status_code}, {response.text}")

# file_path = f'/{file_name}'

# file_name = f"nepse_{datetime.today().strftime('%Y-%m-%d')}.csv"
# file_path = file_name  # Remove the leading slash


# # upload_url = f'https://api.github.com/repos/iamsrijit/Nepse/contents/{file_path}'
# upload_url = f"https://api.github.com/repos/iamsrijit/Nepse/contents/{file_path}"

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

# Function to get the latest file URL
def get_latest_file_url(repo_url):
    # Send a GET request to the GitHub repository
    response = requests.get(repo_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all links to files in the repository
    file_links = soup.find_all('a', href=True)

    # Extract file names and corresponding URLs
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

    # Get the latest file URL based on the date
    latest_file_date = max(file_urls.keys())
    latest_file_url = file_urls[latest_file_date]
    print("Latest file date:", latest_file_date)
    print("Latest file URL:", latest_file_url)
    return latest_file_url

# Replace with the actual GitHub repository URL
repo_url = 'https://github.com/iamsrijit/Nepse/tree/main'

try:
    # Get the latest file URL
    latest_file_url = get_latest_file_url(repo_url)

    # Correct the file URL
    latest_file_url = latest_file_url.replace('/iamsrijit/Nepse/blob/main/', '/')

    # Read data from the latest file
    secondss = pd.read_csv(latest_file_url)

    # Assuming the column names are the same as in your example
    secondss.columns = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Percent Change', 'Volume',"52High","52Low"]

    print(secondss.head())
except Exception as e:
    print("An error occurred:", e)

secondss.head()

# """**merging ma date anusar milaunu paryoo**

# """

import pandas as pd

# Assuming 'first' and 'secondss' are your DataFrames

dfs = [first, secondss]

# Create an empty DataFrame to store the final results
finall_df = pd.DataFrame()

for df in dfs:
    try:
        if 'Date' not in df.columns:
            print(f"'Date' column not found in DataFrame.")
            continue

        # Convert the 'Date' column to datetime format and drop rows with invalid dates
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%m/%d/%Y')
        df.dropna(subset=['Date'], inplace=True)
    except Exception as e:
        print(f"Error processing DataFrame: {e}")

# Combine all the DataFrames
if not dfs:
    print("No valid data to process.")
else:
    combined_df = pd.concat(dfs, ignore_index=True, join='outer')
    combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='%m/%d/%Y')

    # Iterate over each unique symbol
    for symbol in combined_df['Symbol'].unique():
        symbol_df = combined_df[combined_df['Symbol'] == symbol]
        symbol_df = symbol_df.sort_values(by=['Date'], ascending=False)
        symbol_df = symbol_df.drop_duplicates(subset=['Date'], keep='first')

        # Initialize columns 'G' and 'H' as None
        # symbol_df['G'] = None
        # symbol_df['H'] = None

        # Append to the final DataFrame
        finall_df = pd.concat([finall_df, symbol_df], ignore_index=True)

    # Save the final DataFrame to a CSV file
    output_file_name = 'combined_data.csv'

    # Write the headers to the CSV file
    with open(output_file_name, 'w') as f:
        headers = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Percent Change', 'Volume',"52High","52Low"]
        f.write(','.join(headers) + '\n')

    # Append the final DataFrame without headers to the CSV file
    finall_df.to_csv(output_file_name, mode='a', index=False, header=False)

    # Optional: print the first few rows of the final DataFrame
    # print(final_df.head())

print(finall_df.head())

# """**exclude mutual funds **"""

import pandas as pd

# Assuming 'first' and 'secondss' are your DataFrames
dfs = [first, secondss]

# Create an empty DataFrame to store the final results
finall_df = pd.DataFrame()

# List of symbols to exclude
exclude_symbols = [
    
]

for df in dfs:
    try:
        if 'Date' not in df.columns:
            print(f"'Date' column not found in DataFrame.")
            continue

        # Convert the 'Date' column to datetime format and drop rows with invalid dates
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%m/%d/%Y')
        df.dropna(subset=['Date'], inplace=True)
    except Exception as e:
        print(f"Error processing DataFrame: {e}")

# Combine all the DataFrames
if not dfs:
    print("No valid data to process.")
else:
    combined_df = pd.concat(dfs, ignore_index=True, join='outer')

    # Filter out the excluded symbols
    combined_df = combined_df[~combined_df['Symbol'].isin(exclude_symbols)]

    combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='%m/%d/%Y')

    # Iterate over each unique symbol
    for symbol in combined_df['Symbol'].unique():
        symbol_df = combined_df[combined_df['Symbol'] == symbol]
        symbol_df = symbol_df.sort_values(by=['Date'], ascending=False)
        symbol_df = symbol_df.drop_duplicates(subset=['Date'], keep='first')

        # Initialize columns 'G' and 'H' as None
        # symbol_df['G'] = None
        # symbol_df['H'] = None

        # Append to the final DataFrame
        finall_df = pd.concat([finall_df, symbol_df], ignore_index=True)

    # Save the final DataFrame to a CSV file
    output_file_name = 'combined_data.csv'

    # Write the headers to the CSV file
    with open(output_file_name, 'w') as f:
        headers = ['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Percent Change', 'Volume',"52High","52Low"]
        f.write(','.join(headers) + '\n')

    # Append the final DataFrame without headers to the CSV file
    finall_df.to_csv(output_file_name, mode='a', index=False, header=False)

    # Optional: print the first few rows of the final DataFrame
    # print(final_df.head())

print(finall_df.head())

# """**Uploading aaile samma ko merged data in github**"""


import requests
import os
from datetime import datetime
import base64


try:
    file_name = f'espen_{datetime.today().strftime("%Y-%m-%d")}.csv'
    csv_data = finall_df.to_csv(index=False)
    csv_b64 = base64.b64encode(csv_data.encode()).decode()
    
    headers = {'Authorization': f'token {os.getenv("GH_TOKEN")}'}
    payload = {
        'message': f'Add {file_name}',
        'content': csv_b64,
        'branch': 'main'
    }
    
    # Debug print
    print(f"Attempting upload to: https://api.github.com/repos/iamsrijit/Nepse/contents/{file_name}")
    
    response = requests.put(
        f'https://api.github.com/repos/iamsrijit/Nepse/contents/{file_name}',
        headers=headers,
        json=payload
    )
    
    # Debug response
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    
except Exception as e:
    print(f"Upload failed completely: {str(e)}")
    raise









































# try:
#     # Convert finall_df to CSV format
#     csv_data = finall_df.to_csv(index=False)

#     # Encode the CSV data to Base64
#     csv_data_base64 = base64.b64encode(csv_data.encode()).decode()
#     file_name = f'espen_{datetime.today().strftime("%Y-%m-%d")}.csv'
#     file_path = file_name
#     upload_url = f'https://api.github.com/repos/iamsrijit/Nepse/contents/{file_path}'
    
#     # Define the GitHub repository URL
#     repo_url = 'https://github.com/iamsrijit/Nepse'

#     # # Define the file name with today's date
#     # # file_name = f'espen_{datetime.today().strftime("%Y-%m-%d")}.csv'

  
#     # # Define the file path in the repository
#     # file_path = file_name

#     # # Define the API URL for uploading files to GitHub
#     #  upload_url = f'https://api.github.com/repos/iamsrijit/Nepse/contents{file_path}'
#     # # cHANGED ABOVE LINE

#     # # upload_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"


    


#     headers = {
#         'Authorization': f'token {GH_TOKEN}',
#         'Accept': 'application/vnd.github.v3+json'
#     }
#     # Prepare the payload with file content
#     payload = {
#         'message': f'Upload {file_name}',
#         'content': csv_data_base64,
#         'branch': 'main'  # Specify the branch you want to upload to
#     }

#     # Send a PUT request to upload the file
#     response = requests.put(upload_url, headers=headers, json=payload)

#     # Check the response status
#     if response.status_code == 200:
#         print(f'File {file_name} uploaded successfully!')
#     elif response.status_code == 422:
#         print(f'Failed to upload {file_name}. Status code: 422 Unprocessable Entity')
#         print('Error Message:', response.json()['message'])
#     else:
#         print(f'Failed to upload {file_name}. Status code: {response.status_code}')
#         # print('Response Content:', response.text)

# except Exception as e:
#     print('An error occurred:', e)



 


# Load the data
data = finall_df  # Assigning finall_df to data


# Convert Date column to datetime, handling the format explicitly
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')  # Specify the correct format

# Sort by Symbol and Date
data = data.sort_values(['Symbol', 'Date']).reset_index(drop=True)

# Convert Close and Volume to numeric and optimize data types
# Handle invalid values in the Volume column (e.g., '-')
data['Close'] = data['Close'].astype(str).str.replace(',', '').astype('float32')
data['Volume'] = data['Volume'].astype(str).str.replace(',', '')  # Remove commas
data['Volume'] = data['Volume'].replace('-', np.nan)  # Replace '-' with NaN
data['Volume'] = data['Volume'].astype('float32')  # Convert to float32

# Function to calculate RSI (vectorized)
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to process each symbol
def process_symbol(symbol, group):
    # Calculate 20-day and 50-day EMAs
    group['EMA_20'] = group['Close'].ewm(span=20, adjust=False).mean()
    group['EMA_50'] = group['Close'].ewm(span=50, adjust=False).mean()

    # Calculate RSI (14-day)
    group['RSI'] = calculate_rsi(group)

    # Calculate 30-day average volume
    group['30D_Avg_Volume'] = group['Volume'].rolling(window=30, min_periods=1).mean()

    # Calculate EMA slopes (5-day change)
    group['Slope_20'] = group['EMA_20'].diff(5) / 5
    group['Slope_50'] = group['EMA_50'].diff(5) / 5

    # Detect crossovers
    group['Crossover'] = (group['EMA_20'] > group['EMA_50']) & (group['EMA_20'].shift(1) <= group['EMA_50'].shift(1))

    # Filter crossovers based on strategy
    valid_crossovers = group[
        group['Crossover'] &
        (group['Slope_20'] > group['Slope_50']) &
        (group['RSI'].between(30, 70)) &
        (group['Volume'] >= 0.3 * group['30D_Avg_Volume']) &
        (group['Close'] > group['Close'].rolling(window=60, min_periods=1).max().shift(1) * 0.95)
    ]

    return valid_crossovers

# Process all symbols in parallel
print("Processing data...")
results = Parallel(n_jobs=-1)(delayed(process_symbol)(symbol, group) for symbol, group in data.groupby('Symbol'))

# Combine results into a single DataFrame
all_valid_crossovers = pd.concat(results).reset_index(drop=True)

# Display valid crossover dates
# print("\nValid EMA Crossover Dates for All Symbols:")
# print(all_valid_crossovers[['Symbol', 'Date', 'Close', 'EMA_20', 'EMA_50', 'RSI', 'Volume', '30D_Avg_Volume']])

# Save the valid crossovers to a CSV file
all_valid_crossovers.to_csv('valid_ema_crossovers.csv', index=False)
# print("\nValid crossovers saved to 'valid_ema_crossovers.csv'.")

# Output only the latest dates in valid EMA Crossover Dates for each symbol
combinedd_df = all_valid_crossovers.sort_values('Date', ascending=False).drop_duplicates('Symbol')
print("\nLatest Valid EMA Crossover Dates for Each Symbol:")
print(combinedd_df[['Symbol', 'Date', 'Close']])
#  'EMA_20', 'EMA_50', 'RSI', 'Volume', '30D_Avg_Volume']])

# Save the latest valid crossovers to a CSV file
combinedd_df.to_csv('latest_valid_ema_crossovers.csv', index=False)
print("\nLatest valid crossovers saved to 'latest_valid_ema_crossovers.csv'.")




import requests
from datetime import datetime
import base64

# Assuming you have combined_df DataFrame containing your data
# Assuming sorted_df is defined elsewhere in your code

try:
    # Convert sorted_df to CSV format
    csv_data = combinedd_df.to_csv(index=False)

    # Encode the CSV data to Base64
    csv_data_base64 = base64.b64encode(csv_data.encode()).decode()

    # Define the GitHub repository URL
    repo_url = 'https://github.com/iamsrijit/Nepse'

    # Define the file name with today's date
    file_name = f'EMA_Cross_for_{datetime.today().strftime("%Y-%m-%d")}.csv'

    # Define your personal access token
    # token = 'TOKEN'

    # Define the file path in the repository
    file_path = f'/{file_name}'

    # Define the API URL for file content and upload
    upload_url = f'https://api.github.com/repos/iamsrijit/Nepse/contents{file_path}'

    # Prepare the headers with the authorization token
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
# CHANGED ABOVE
    # headers = {
    # 'Authorization': f'Bearer {token}',
    # 'Accept': 'application/vnd.github.v3+json'
    # }


    # Get the SHA of the existing file if it exists
    response = requests.get(upload_url, headers=headers)
    if response.status_code == 200:
        # File exists, extract SHA
        sha = response.json()['sha']
    else:
        # File doesn't exist, set sha to None
        sha = None

    # Prepare the payload with file content and optional SHA
    payload = {
        'message': f'Upload {file_name}',
        'content': csv_data_base64,
        'branch': 'main'  # Specify the branch you want to upload to
    }

    if sha:
        # Include the SHA if we're updating an existing file
        payload['sha'] = sha

    # Send a PUT request to upload/replace the file
    response = requests.put(upload_url, headers=headers, json=payload)

    # Check the response status
    if response.status_code == 201:
        print(f'File {file_name} uploaded successfully!')
    elif response.status_code == 200:
        print(f'File {file_name} updated successfully!')
    elif response.status_code == 422:
        print(f'Failed to upload {file_name}. Status code: 422 Unprocessable Entity')
        print('Error Message:', response.json()['message'])
    else:
        print(f'Failed to upload {file_name}. Status code: {response.status_code}')
        print('Response Content:', response.text)

except Exception as e:
    print('An error occurred:', e)

from git import Repo
import os
import re

# Clone repository
repo_dir = './nepse_repo'
if not os.path.exists(repo_dir):
    Repo.clone_from('https://github.com/iamsrijit/Nepse.git', repo_dir)
repo = Repo(repo_dir)

# Get all files and define patterns
all_files = os.listdir(repo_dir)
ema_pattern = r'^EMA_Cross_for_\d{4}-\d{2}-\d{2}\.csv$'
espen_pattern = r'^espen_\d{4}-\d{2}-\d{2}\.csv$'

# Identify latest files and delete others
latest_ema = max([f for f in all_files if re.match(ema_pattern, f)], default=None)
latest_espen = max([f for f in all_files if re.match(espen_pattern, f)], default=None)

files_to_delete = []
if latest_ema:
    files_to_delete += [f for f in all_files if re.match(ema_pattern, f) and f != latest_ema]
if latest_espen:
    files_to_delete += [f for f in all_files if re.match(espen_pattern, f) and f != latest_espen]

# Remove files and commit
for file in files_to_delete:
    os.remove(os.path.join(repo_dir, file))
    repo.index.remove([file], working_tree=True)

# Push changes with token authentication
try:
    repo.git.add(update=True)
    repo.index.commit('Remove old data files')
    origin = repo.remote(name='origin')
    origin.set_url(f'https://x-access-token:{os.getenv("GH_TOKEN")}@github.com/iamsrijit/Nepse.git')
    origin.push()
except Exception as e:
    print(f"Error pushing changes: {e}")


























# from git import Repo
# import os
# import re

# # Function to identify the latest file for a given pattern
# def get_latest_file(pattern, files):
#     matched_files = [f for f in files if re.match(pattern, f)]
#     return max(matched_files) if matched_files else None

# # Replace with your GitHub repository URL
# # repo_url = 'https://github.com/iamsrijit/Nepse.git'
# # repo_dir = '/content/nepse_new'  # Directory to clone the repository


# # Replace with your GitHub repository URL
# repo_url = 'https://github.com/iamsrijit/Nepse.git'
# repo_dir = './nepse_repo'  # Use a writable directory instead of '/content/nepse_new'

# # Clone the repository
# if not os.path.exists(repo_dir):  # Ensure the directory does not already exist
#     repo = Repo.clone_from(repo_url, repo_dir)
# else:
#     repo = Repo(repo_dir)  # Open existing repository

# print("Repository cloned successfully!")


# # List all files in the repository directory
# all_files = os.listdir(repo_dir)

# # Delete older files for EMA_Cross_for_
# latest_ema_file = get_latest_file(r'^EMA_Cross_for_\d{4}-\d{2}-\d{2}\.csv$', all_files)
# if latest_ema_file:
#     print(f"Latest EMA file: {latest_ema_file}")
#     files_to_delete = [f for f in all_files if re.match(r'^EMA_Cross_for_\d{4}-\d{2}-\d{2}\.csv$', f) and f != latest_ema_file]
#     for file_to_delete in files_to_delete:
#         file_path = os.path.join(repo_dir, file_to_delete)
#         try:
#             os.remove(file_path)
#             repo.index.remove([file_path], working_tree=True)
#             print(f"Deleted {file_to_delete}")
#         except Exception as e:
#             print(f"Error deleting {file_to_delete}: {e}")

# # Delete older files for espen_
# latest_espen_file = get_latest_file(r'^espen_\d{4}-\d{2}-\d{2}\.csv$', all_files)
# if latest_espen_file:
#     print(f"Latest espn file: {latest_espen_file}")
#     files_to_delete = [f for f in all_files if re.match(r'^espen_\d{4}-\d{2}-\d{2}\.csv$', f) and f != latest_espen_file]
#     for file_to_delete in files_to_delete:
#         file_path = os.path.join(repo_dir, file_to_delete)
#         try:
#             os.remove(file_path)
#             repo.index.remove([file_path], working_tree=True)
#             print(f"Deleted {file_to_delete}")
#         except Exception as e:
#             print(f"Error deleting {file_to_delete}: {e}")

# # Commit and push changes
# try:
#     repo.index.commit('Deleted old EMA_Cross_for_ and espen_ files')
#     origin = repo.remote(name='origin')
#     # origin.set_url(f'https://x-access-token:{os.getenv("GH_TOKEN")}@github.com/iamsrijit/Nepse.git')

#     # origin.set_url('https://iamsrijit:{token}@github.com/iamsrijit/Nepse.git')
#     # Correct Git remote URL with token
#     origin.set_url(f'https://x-access-token:{os.getenv("GH_TOKEN")}@github.com/iamsrijit/Nepse.git')
#     origin.push()
#     print("Pushed changes to GitHub.")
# except Exception as e:
#     print(f"Error pushing changes to GitHub: {e}")

# print("Previous files have been deleted from the GitHub repository.")
