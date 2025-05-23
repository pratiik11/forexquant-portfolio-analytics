import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

def update_currency_data():
    """
    Updates all currency pair CSV files in the Data directory with latest OHLC data.
    
    This function:
    1. Reads each CSV file in the Data directory
    2. Checks the latest date in each file
    3. If not up to date (doesn't contain today's date or latest trading day),
       fetches new data from a free Forex API and appends it to the CSV
    """
    # Get list of all CSV files in Data directory
    data_dir = 'Data'
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # Get today's date
    today = datetime.now().date()
    
    # If today is weekend (Saturday=5, Sunday=6), adjust to latest trading day (Friday)
    if today.weekday() >= 5:  # Weekend
        days_to_subtract = today.weekday() - 4  # Calculate days to go back to Friday
        today = today - timedelta(days=days_to_subtract)
    
    # Format the date string as it appears in the CSV files
    today_str = today.strftime('%Y-%m-%d')
    
    # Process each CSV file
    updated_files = 0
    skipped_files = 0
    api_calls = 0
    
    print(f"Starting update process for {len(csv_files)} currency pair files...")
    
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        
        try:
            # Read CSV file into DataFrame
            df = pd.read_csv(file_path)
            
            # Get currency pair symbol from the file
            ticker = csv_file.replace('.csv', '')
            
            # Get the latest date in the file
            latest_date_str = df['date'].iloc[-1]
            latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d').date()
            
            # Check if file already has today's data
            if latest_date >= today:
                skipped_files += 1
                print(f"Skipping {ticker}: Already up to date (latest: {latest_date_str})")
                continue
            
            # Calculate how many days to fetch
            days_diff = (today - latest_date).days
            
            if days_diff > 0:
                print(f"Updating {ticker}: Latest date is {latest_date_str}, fetching {days_diff} days of data...")
                
                # Fetch new data from API
                new_data = fetch_forex_data(ticker, latest_date, today)
                
                if new_data and len(new_data) > 0:
                    # Convert new data to DataFrame
                    new_df = pd.DataFrame(new_data)
                    
                    # Append new data to original DataFrame
                    combined_df = pd.concat([df, new_df], ignore_index=True)
                    
                    # Remove duplicate rows based on date, keeping the most recent one
                    updated_df = combined_df.drop_duplicates(subset=["date"], keep="last")
                    
                    # Sort by date to ensure chronological order
                    updated_df = updated_df.sort_values('date').reset_index(drop=True)
                    
                    # Save updated DataFrame back to CSV
                    updated_df.to_csv(file_path, index=False)
                    
                    updated_files += 1
                    print(f"✅ Updated {ticker} with {len(new_data)} new data points")
                else:
                    print(f"No new data available for {ticker}")
                
                api_calls += 1
                
                # Sleep to avoid hitting rate limits
                if api_calls % 5 == 0:  # Sleep after every 5 API calls
                    time.sleep(1)  # 1 second pause
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
    
    print(f"\nUpdate summary:")
    print(f"- Files processed: {len(csv_files)}")
    print(f"- Files updated: {updated_files}")
    print(f"- Files already up to date: {skipped_files}")
    print(f"- Files with errors: {len(csv_files) - updated_files - skipped_files}")


def fetch_forex_data(ticker, start_date, end_date):
    """
    Fetches Forex OHLC data for a given currency pair and date range.
    
    Args:
        ticker: The currency pair symbol (e.g., 'USDEUR')
        start_date: Starting date to fetch data from
        end_date: Ending date to fetch data until
    
    Returns:
        List of dictionaries with OHLC data
    """
    # Format dates for API
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Get base and quote currencies from ticker
    base_currency = ticker[:3]
    quote_currency = ticker[3:]

    # Option 1: Using FreeForexAPI (doesn't require API key but has limitations)
    try:
        # Note: FreeForexAPI has limitations - check their docs
        url = f"https://www.freeforexapi.com/api/historical?pair={base_currency}{quote_currency}&from={start_date_str}&to={end_date_str}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if "candles" in data and data["candles"]:
                results = []
                
                for date_str, candle in data["candles"].items():
                    # Convert timestamp to date
                    date = datetime.fromtimestamp(int(date_str)).strftime('%Y-%m-%d')
                    
                    results.append({
                        "ticker": ticker,
                        "date": date,
                        "open": candle["o"],
                        "high": candle["h"],
                        "low": candle["l"],
                        "close": candle["c"]
                    })
                
                return results
    except Exception as e:
        print(f"Error fetching data from FreeForexAPI: {str(e)}")
    
    # Option 2: Using TwelveData API (requires API key)
    try:
        api_key = os.getenv('TWELVEDATA_API_KEY')
        if not api_key:
            print("TwelveData API key not found in .env file")
            return []
        
        url = f"https://api.twelvedata.com/time_series"
        params = {
            "symbol": f"{base_currency}/{quote_currency}",
            "interval": "1day",
            "start_date": start_date_str,
            "end_date": end_date_str,
            "apikey": api_key
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if "values" in data:
                results = []
                
                for item in data["values"]:
                    results.append({
                        "ticker": ticker,
                        "date": item["datetime"],
                        "open": float(item["open"]),
                        "high": float(item["high"]),
                        "low": float(item["low"]),
                        "close": float(item["close"])
                    })
                
                return results
    except Exception as e:
        print(f"Error fetching data from TwelveData: {str(e)}")
    
    return []


if __name__ == "__main__":
    update_currency_data() 