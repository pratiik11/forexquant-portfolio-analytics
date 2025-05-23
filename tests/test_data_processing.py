import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.data_processing import load_currency_data, get_available_currencies, calculate_correlation_matrix

class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Get list of available currencies for testing
        self.available_currencies = get_available_currencies()
        
        # Make sure we have at least one currency pair to test with
        self.assertTrue(len(self.available_currencies) > 0, "No currency data files available for testing")
        
        # Use first available currency for tests
        self.test_ticker = self.available_currencies[0]
        
        # Testing date range - last year
        self.end_date = datetime.now().date()
        self.start_date = self.end_date - timedelta(days=365)
        self.start_date_str = self.start_date.strftime('%Y-%m-%d')
        
    def test_load_currency_data(self):
        """Test that load_currency_data loads and processes data correctly"""
        # Load data for test ticker
        df = load_currency_data(self.test_ticker, start_date=self.start_date_str)
        
        # Check that DataFrame has expected columns
        expected_columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 
                           'pct_return', 'log_return', 'volatility_7d', 'volatility_30d', 'cum_return']
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Column {col} missing from loaded data")
        
        # Check that DataFrame has data and is sorted by date
        self.assertTrue(len(df) > 0, "DataFrame is empty")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['date']), "Date column is not datetime type")
        self.assertTrue((df['date'].diff()[1:] > timedelta(0)).all(), "DataFrame is not sorted by date")
        
        # Check that calculated returns are valid
        self.assertTrue(df['pct_return'].max() < 1.0, "Percentage returns are unrealistically high")
        self.assertTrue(df['pct_return'].min() > -1.0, "Percentage returns are unrealistically low")
        self.assertTrue(np.isclose(df['log_return'].iloc[1:], np.log(1 + df['pct_return'].iloc[1:])).all(), 
                       "Log returns don't match percentage returns")
        
        # Check that volatility columns have reasonable values
        self.assertTrue(df['volatility_7d'].max() < 2.0, "7-day volatility is unrealistically high")
        self.assertTrue(df['volatility_30d'].max() < 2.0, "30-day volatility is unrealistically high")
        
        # Check that cumulative returns are calculated correctly
        expected_cum_return = (1 + df['pct_return']).cumprod() - 1
        self.assertTrue(np.allclose(df['cum_return'], expected_cum_return, equal_nan=True), 
                       "Cumulative returns are not calculated correctly")
        
    def test_correlation_matrix(self):
        """Test that correlation matrix calculation works correctly"""
        # Need at least two currencies for correlation
        if len(self.available_currencies) >= 2:
            test_tickers = self.available_currencies[:2]
            corr_matrix = calculate_correlation_matrix(test_tickers, start_date=self.start_date_str)
            
            # Check that correlation matrix has correct shape
            self.assertEqual(corr_matrix.shape, (2, 2), "Correlation matrix has incorrect shape")
            
            # Check that diagonal elements are 1.0
            for ticker in test_tickers:
                if ticker in corr_matrix.index and ticker in corr_matrix.columns:
                    self.assertEqual(corr_matrix.loc[ticker, ticker], 1.0, 
                                    f"Correlation of {ticker} with itself is not 1.0")
            
            # Check that correlation values are between -1 and 1
            self.assertTrue((corr_matrix.values <= 1.0).all(), "Correlation values exceed 1.0")
            self.assertTrue((corr_matrix.values >= -1.0).all(), "Correlation values are below -1.0")

if __name__ == '__main__':
    unittest.main() 