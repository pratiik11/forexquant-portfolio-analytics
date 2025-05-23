import sys
import os
import unittest
import pandas as pd
import numpy as np

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.data_processing import get_available_currencies
from core.portfolio import construct_portfolio, calculate_risk_metrics

class TestPortfolio(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Get list of available currencies for testing
        self.available_currencies = get_available_currencies()
        
        # Make sure we have at least 5 currency pairs to test with
        self.assertTrue(len(self.available_currencies) >= 5, "Not enough currency data files for testing")
        
        # Use first 5 available currencies for tests
        self.test_tickers = self.available_currencies[:5]
        
        # Testing date range
        self.start_date = "2023-01-01"
        self.end_date = "2023-12-31"
    
    def test_equal_weight_portfolio(self):
        """Test that equal weight portfolio construction works properly"""
        # Construct equal weight portfolio
        portfolio_data = construct_portfolio(
            tickers=self.test_tickers, 
            start_date=self.start_date,
            end_date=self.end_date,
            method="equal"
        )
        
        # Check that all components exist
        self.assertIn('weights', portfolio_data)
        self.assertIn('returns_df', portfolio_data)
        self.assertIn('portfolio_returns', portfolio_data)
        self.assertIn('portfolio_cumulative', portfolio_data)
        
        # Check that weights sum to 1.0 and all are equal
        weights = portfolio_data['weights']
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        expected_weight = 1.0 / len(self.test_tickers)
        for ticker, weight in weights.items():
            self.assertAlmostEqual(weight, expected_weight)
    
    def test_min_var_portfolio(self):
        """Test that minimum variance portfolio construction works properly"""
        # Construct min var portfolio
        try:
            portfolio_data = construct_portfolio(
                tickers=self.test_tickers, 
                start_date=self.start_date,
                end_date=self.end_date,
                method="min_var"
            )
            
            # Check that weights sum to approximately 1.0
            weights = portfolio_data['weights']
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)
            
            # Calculate risk metrics
            risk_metrics = calculate_risk_metrics(portfolio_data['portfolio_returns'])
            
            # Check that risk metrics are reasonable
            self.assertIn('annualized_volatility', risk_metrics)
            self.assertIn('sharpe_ratio', risk_metrics)
            self.assertIn('max_drawdown', risk_metrics)
            self.assertIn('var_95', risk_metrics)
            self.assertIn('cvar_95', risk_metrics)
            
            # Volatility should be positive
            self.assertGreater(risk_metrics['annualized_volatility'], 0)
            
            # Max drawdown should be negative or zero
            self.assertLessEqual(risk_metrics['max_drawdown'], 0)
            
        except Exception as e:
            self.fail(f"Min var portfolio construction failed: {e}")
    
    def test_max_sharpe_portfolio(self):
        """Test that maximum Sharpe Ratio portfolio construction works properly"""
        # Construct max Sharpe portfolio
        try:
            portfolio_data = construct_portfolio(
                tickers=self.test_tickers, 
                start_date=self.start_date,
                end_date=self.end_date,
                method="max_sharpe"
            )
            
            # Check that weights sum to approximately 1.0
            weights = portfolio_data['weights']
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)
            
            # Calculate risk metrics
            risk_metrics = calculate_risk_metrics(portfolio_data['portfolio_returns'])
            
            # Check that Sharpe ratio is calculated
            self.assertIn('sharpe_ratio', risk_metrics)
            
        except Exception as e:
            self.fail(f"Max Sharpe portfolio construction failed: {e}")
    
    def test_risk_metrics(self):
        """Test the calculation of risk metrics"""
        # Construct a portfolio to get returns
        portfolio_data = construct_portfolio(
            tickers=self.test_tickers[:3],  # Use fewer tickers for speed
            start_date=self.start_date,
            end_date=self.end_date,
            method="equal"
        )
        
        # Calculate risk metrics
        risk_metrics = calculate_risk_metrics(portfolio_data['portfolio_returns'])
        
        # Check that all metrics are calculated
        expected_metrics = [
            'annualized_return', 'annualized_volatility', 'sharpe_ratio',
            'max_drawdown', 'var_95', 'cvar_95'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, risk_metrics)
        
        # Check that VaR is less than or equal to CVaR (CVaR should be more negative)
        self.assertLessEqual(risk_metrics['cvar_95'], risk_metrics['var_95'])

if __name__ == '__main__':
    unittest.main() 