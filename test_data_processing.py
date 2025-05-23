import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.data_processing import load_currency_data, get_available_currencies, calculate_correlation_matrix

def test_load_currency_data():
    """Test the load_currency_data function with a sample currency pair"""
    
    # Get available currencies
    currencies = get_available_currencies()
    print(f"Found {len(currencies)} currency pairs")
    print(f"First 5 currencies: {currencies[:5]}")
    
    # Test with first currency
    ticker = currencies[0]
    print(f"\nTesting with {ticker}")
    
    # Load data for last 2 years
    df = load_currency_data(ticker, start_date='2023-01-01')
    
    # Print sample of the data
    print("\nSample data:")
    print(df.tail().to_string())
    
    # Print summary statistics
    print("\nSummary statistics:")
    print(df.describe())
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    # Plot 1: Price and Returns
    ax1 = axes[0]
    ax1.set_title(f'{ticker} Price and Daily Returns')
    ax1.plot(df['date'], df['close'], color='blue', label='Price')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df['date'], df['pct_return'], color='red', alpha=0.5, label='Return')
    ax1.set_ylabel('Price')
    ax1_twin.set_ylabel('Return %')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # Plot 2: Volatility
    ax2 = axes[1]
    ax2.set_title(f'{ticker} Rolling Volatility')
    ax2.plot(df['date'], df['volatility_7d'], label='7-day Volatility')
    ax2.plot(df['date'], df['volatility_30d'], label='30-day Volatility')
    ax2.set_ylabel('Annualized Volatility')
    ax2.legend()
    
    # Plot 3: Cumulative Return
    ax3 = axes[2]
    ax3.set_title(f'{ticker} Cumulative Return')
    ax3.plot(df['date'], df['cum_return'] * 100)
    ax3.set_ylabel('Cumulative Return %')
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_analysis.png')
    print(f"Saved analysis to {ticker}_analysis.png")
    
    return df

def test_correlation_matrix():
    """Test the correlation matrix calculation with a few currency pairs"""
    
    currencies = get_available_currencies()
    
    # Select first 5 currencies for correlation matrix
    test_currencies = currencies[:5]
    print(f"\nCalculating correlation matrix for: {test_currencies}")
    
    # Calculate correlation matrix for last year
    corr_matrix = calculate_correlation_matrix(test_currencies, start_date='2023-01-01')
    
    # Print correlation matrix
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Plot correlation matrix as heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Currency Pair Return Correlations')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    print(f"Saved correlation matrix to correlation_matrix.png")
    
    return corr_matrix

if __name__ == "__main__":
    print("Testing data processing module...")
    
    # Test data loading
    df = test_load_currency_data()
    
    # Test correlation matrix
    corr_matrix = test_correlation_matrix()
    
    print("\nAll tests completed successfully!") 