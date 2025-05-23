import os
import pandas as pd
import numpy as np
from datetime import datetime, date
import math
from typing import List
import streamlit as st

@st.cache_data(ttl=86400)  # Cache data for 24 hours
def load_currency_data(ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Loads and processes currency data for a given ticker symbol.
    
    Args:
        ticker (str): Currency pair ticker symbol (e.g., 'USDEUR')
        start_date (str, optional): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format
        
    Returns:
        pd.DataFrame: DataFrame with original OHLC data and calculated metrics:
                     - pct_return: Simple percentage daily returns
                     - log_return: Logarithmic daily returns
                     - volatility_7d: 7-day rolling volatility of returns
                     - volatility_30d: 30-day rolling volatility of returns
                     - cum_return: Cumulative return from start date
    """
    # Construct file path
    file_path = os.path.join('Data', f'{ticker}.csv')
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Currency data file not found for {ticker}")
    
    # Load data from CSV
    df = pd.read_csv(file_path)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date (ascending)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Filter by date range if provided
    if start_date:
        start_datetime = pd.to_datetime(start_date)
        df = df[df['date'] >= start_datetime]
    
    if end_date:
        end_datetime = pd.to_datetime(end_date)
        df = df[df['date'] <= end_datetime]
    
    # Calculate daily returns
    df['pct_return'] = df['close'].pct_change()
    
    # Calculate log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate rolling volatilities (standard deviation of returns)
    df['volatility_7d'] = df['log_return'].rolling(window=7).std() * np.sqrt(252)  # Annualized
    df['volatility_30d'] = df['log_return'].rolling(window=30).std() * np.sqrt(252)  # Annualized
    
    # Calculate cumulative return (from start)
    df['cum_return'] = (1 + df['pct_return']).cumprod() - 1
    # Fix the first NaN value (more pandas 3.0 compatible way)
    df = df.assign(cum_return=lambda x: x['cum_return'].fillna(0))
    
    # Drop unnecessary columns to reduce memory usage
    # Keep only the columns we need for analysis
    df = df[['date', 'close', 'pct_return', 'log_return', 'volatility_7d', 'volatility_30d', 'cum_return']]
    
    return df

def get_available_currencies() -> list:
    """
    Returns a list of all available currency pairs from the Data directory.
    
    Returns:
        list: List of currency pair tickers
    """
    data_dir = 'Data'
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Get all CSV files and remove .csv extension to get tickers
    csv_files = [f.replace('.csv', '') for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    return sorted(csv_files)

@st.cache_data(ttl=86400)  # Cache data for 24 hours
def calculate_correlation_matrix(tickers: list, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Calculates correlation matrix between multiple currency pairs.
    
    Args:
        tickers (list): List of currency pair tickers
        start_date (str, optional): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format
        
    Returns:
        pd.DataFrame: Correlation matrix of daily returns
    """
    returns_data = {}
    
    for ticker in tickers:
        try:
            df = load_currency_data(ticker, start_date, end_date)
            returns_data[ticker] = df['log_return'].dropna()
        except Exception as e:
            print(f"Error loading data for {ticker}: {str(e)}")
    
    # Create a DataFrame with returns of all currency pairs
    returns_df = pd.DataFrame(returns_data)
    
    # Calculate correlation matrix
    correlation_matrix = returns_df.corr()
    
    return correlation_matrix

@st.cache_data(ttl=86400)  # Cache data for 24 hours
def calculate_portfolio_metrics(weights: dict, start_date: str = None, end_date: str = None) -> dict:
    """
    Calculate portfolio performance metrics given currency weights.
    
    Args:
        weights (dict): Dictionary mapping currency tickers to weights {ticker: weight}
        start_date (str, optional): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format
        
    Returns:
        dict: Dictionary with portfolio performance metrics:
             - daily_returns: Series of daily portfolio returns
             - cumulative_return: Final cumulative return
             - volatility: Portfolio volatility (annualized)
             - sharpe_ratio: Sharpe ratio (assuming risk-free rate of 0)
    """
    tickers = list(weights.keys())
    returns_data = {}
    
    # Validate weights sum to 1.0
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("Portfolio weights must sum to 1.0")
    
    # Load return data for each ticker
    for ticker in tickers:
        df = load_currency_data(ticker, start_date, end_date)
        returns_data[ticker] = df[['date', 'log_return']].copy()
        returns_data[ticker].set_index('date', inplace=True)
    
    # Create a single DataFrame with all returns aligned by date
    all_returns = pd.DataFrame()
    for ticker, data in returns_data.items():
        all_returns[ticker] = data['log_return']
    
    # Drop rows with any NaN values
    all_returns.dropna(inplace=True)
    
    # Calculate portfolio returns using weights
    portfolio_returns = pd.Series(0.0, index=all_returns.index)
    for ticker, weight in weights.items():
        portfolio_returns += all_returns[ticker] * weight
    
    # Calculate portfolio metrics
    portfolio_volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
    portfolio_cumulative_return = np.exp(portfolio_returns.sum()) - 1
    portfolio_sharpe = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))
    
    return {
        'daily_returns': portfolio_returns,
        'cumulative_return': portfolio_cumulative_return,
        'volatility': portfolio_volatility,
        'sharpe_ratio': portfolio_sharpe
    }

@st.cache_data(ttl=86400)  # Cache data for 24 hours
def get_common_start_date(tickers: List[str]) -> date:
    """
    Determines the earliest common date available across all selected tickers.
    
    Args:
        tickers (List[str]): List of currency ticker symbols to check
        
    Returns:
        date: The latest minimum date from all selected tickers
    """
    if not tickers:
        # Return a reasonable default if no tickers are provided
        return date(2010, 1, 1)
    
    earliest_dates = []
    
    for ticker in tickers:
        try:
            # Construct file path
            file_path = os.path.join('Data', f'{ticker}.csv')
            
            # Check if file exists
            if os.path.exists(file_path):
                # Load data from CSV - only read the date column for efficiency
                df = pd.read_csv(file_path, usecols=['date'])
                
                # Convert to datetime and get the minimum date
                df['date'] = pd.to_datetime(df['date'])
                min_date = df['date'].min().date()
                earliest_dates.append(min_date)
        except Exception as e:
            print(f"Error getting start date for {ticker}: {str(e)}")
    
    if not earliest_dates:
        # Return a reasonable default if no valid dates were found
        return date(2010, 1, 1)
    
    # Return the latest (max) of all minimum dates to ensure data exists for all tickers
    return max(earliest_dates)

if __name__ == "__main__":
    # Example usage
    try:
        # List available currencies
        available_currencies = get_available_currencies()
        print(f"Available currency pairs: {len(available_currencies)}")
        print(available_currencies[:5])  # Print first 5
        
        # Load and process data for a sample currency
        sample_ticker = available_currencies[0]
        df = load_currency_data(sample_ticker, start_date='2023-01-01', end_date='2023-12-31')
        
        # Display sample of processed data
        print(f"\nSample data for {sample_ticker}:")
        print(df.tail().to_string())
        
        # Example correlation matrix
        if len(available_currencies) >= 3:
            sample_tickers = available_currencies[:3]
            corr_matrix = calculate_correlation_matrix(sample_tickers, start_date='2023-01-01')
            print("\nCorrelation matrix:")
            print(corr_matrix)
            
        # Example portfolio calculation
        sample_weights = {ticker: 1.0/3 for ticker in available_currencies[:3]}
        portfolio_metrics = calculate_portfolio_metrics(sample_weights, start_date='2023-01-01')
        print("\nPortfolio metrics:")
        print(f"Cumulative return: {portfolio_metrics['cumulative_return']:.4f}")
        print(f"Volatility: {portfolio_metrics['volatility']:.4f}")
        print(f"Sharpe ratio: {portfolio_metrics['sharpe_ratio']:.4f}")
        
    except Exception as e:
        print(f"Error in example: {str(e)}") 