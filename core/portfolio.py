"""
Portfolio Construction and Risk Metrics Module

This module provides tools for constructing currency portfolios and calculating risk metrics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Union
from datetime import datetime
import sys
import riskfolio as rf
import plotly.express as px

# Add the parent directory to sys.path when running as a script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.data_processing import load_currency_data, get_available_currencies
else:
    from core.data_processing import load_currency_data

def construct_portfolio(tickers: List[str], start_date=None, end_date=None, method="equal", rebalance_freq="None") -> Dict:
    """
    Constructs a currency portfolio using specified weighting method and calculates performance metrics.
    
    This function builds a portfolio from multiple currency pairs by:
    1. Loading historical price/return data for each currency pair
    2. Aligning all return data by date and handling missing values
    3. Determining optimal weights based on the specified portfolio construction method
    4. Calculating time series of portfolio returns and cumulative performance
    
    Args:
        tickers (List[str]): List of currency pair tickers (e.g. ["EURUSD", "GBPJPY"])
        start_date (str, optional): Start date in 'YYYY-MM-DD' format. If None, uses all available data
        end_date (str, optional): End date in 'YYYY-MM-DD' format. If None, uses data up to the present
        method (str): Portfolio construction method:
          - "equal": Equal weighting - allocates the same weight to each currency pair
          - "min_var": Minimum variance portfolio - optimizes for lowest portfolio volatility
          - "max_sharpe": Maximum Sharpe Ratio portfolio - optimizes for best risk-adjusted return
        rebalance_freq (str): Portfolio rebalancing frequency:
          - "None": No rebalancing, weights are determined once at the beginning
          - "Monthly": Rebalance the portfolio at the end of each month
          - "Quarterly": Rebalance the portfolio at the end of each quarter
    
    Returns:
        Dict: Dictionary containing:
          - weights (Dict[str, float]): Portfolio weights for each currency pair (final weights if rebalancing)
          - returns_df (pd.DataFrame): Daily returns for all currency pairs
          - portfolio_returns (pd.Series): Daily portfolio returns weighted according to allocation
          - portfolio_cumulative (pd.Series): Cumulative portfolio returns over time
          - rebalance_dates (List[pd.Timestamp]): Dates when portfolio was rebalanced (if applicable)
    
    Raises:
        ValueError: If no currency pairs are provided or if an unknown portfolio method is specified
    
    Example:
        >>> portfolio = construct_portfolio(
        ...     tickers=["EURUSD", "GBPJPY", "AUDUSD"],
        ...     start_date="2022-01-01",
        ...     end_date="2023-01-01",
        ...     method="min_var",
        ...     rebalance_freq="Monthly"
        ... )
        >>> print(f"Portfolio weights: {portfolio['weights']}")
    """
    if not tickers or len(tickers) == 0:
        raise ValueError("No currency pairs provided")
    
    # Validate rebalance frequency
    valid_frequencies = ["None", "Monthly", "Quarterly"]
    if rebalance_freq not in valid_frequencies:
        raise ValueError(f"Invalid rebalance frequency. Must be one of: {', '.join(valid_frequencies)}")
    
    # Load return data for each ticker
    returns_data = {}
    price_data = {}
    
    for ticker in tickers:
        df = load_currency_data(ticker, start_date, end_date)
        returns_data[ticker] = df[['date', 'log_return']].copy()
        price_data[ticker] = df[['date', 'close']].copy()
        
        # Set date as index for easier alignment
        returns_data[ticker].set_index('date', inplace=True)
        price_data[ticker].set_index('date', inplace=True)
    
    # Create combined DataFrame with returns of all assets aligned by date
    returns_df = pd.DataFrame()
    for ticker, data in returns_data.items():
        returns_df[ticker] = data['log_return']
    
    # Drop rows with any missing values
    returns_df = returns_df.dropna()
    
    # Create price DataFrame for plotting
    price_df = pd.DataFrame()
    for ticker, data in price_data.items():
        price_df[ticker] = data['close']
    price_df = price_df.loc[returns_df.index]  # Align with returns data
    
    # Initialize portfolio tracking variables
    portfolio_returns = pd.Series(0.0, index=returns_df.index)
    rebalance_dates = []
    current_weights = {}
    all_weights = {}  # Store weights for each rebalance period
    
    # Function to determine optimal weights based on method
    def calculate_weights(return_data):
        if method == "equal":
            # Equal weighting
            return {ticker: 1.0 / len(tickers) for ticker in tickers}
        
        elif method == "min_var":
            # Minimum variance portfolio
            # Convert returns to riskfolio format
            returns_rflio = return_data.copy()
            
            # Create portfolio object
            port = rf.Portfolio(returns=returns_rflio)
            
            # Calculate covariance matrix
            port.assets_stats(method_mu='hist', method_cov='hist')
            
            # Optimize for minimum variance
            w_minvar = port.optimization(model='Classic', rm='MV', obj='MinRisk', 
                                       rf=0, l=0, hist=True)
            
            # Extract weights
            return {ticker: w_minvar.loc[ticker].values[0] for ticker in tickers if ticker in return_data.columns}
        
        elif method == "max_sharpe":
            # Maximum Sharpe Ratio portfolio
            # Convert returns to riskfolio format
            returns_rflio = return_data.copy()
            
            # Create portfolio object
            port = rf.Portfolio(returns=returns_rflio)
            
            # Calculate covariance matrix and mean returns
            port.assets_stats(method_mu='hist', method_cov='hist')
            
            # Optimize for maximum Sharpe ratio
            w_sharpe = port.optimization(model='Classic', rm='MV', obj='Sharpe', 
                                       rf=0, l=0, hist=True)
            
            # Extract weights
            return {ticker: w_sharpe.loc[ticker].values[0] for ticker in tickers if ticker in return_data.columns}
        
        else:
            raise ValueError(f"Unknown portfolio construction method: {method}")
    
    # If no rebalancing is required, calculate weights once and apply to entire period
    if rebalance_freq == "None":
        weights = calculate_weights(returns_df)
        
        # Calculate portfolio returns using these weights
        for ticker, weight in weights.items():
            portfolio_returns += returns_df[ticker] * weight
        
    else:
        # Determine rebalancing frequency code for pandas resample
        freq_code = 'M' if rebalance_freq == "Monthly" else 'Q'
        
        # Group data by rebalancing periods
        period_groups = returns_df.resample(freq_code)
        
        # Initialize a series to store the portfolio returns
        portfolio_returns = pd.Series(index=returns_df.index)
        portfolio_returns.iloc[:] = 0.0
        
        # Process each rebalancing period
        last_end_date = None
        
        for period_start, period_data in period_groups:
            if period_data.empty:
                continue
            
            # The last date of the current period
            period_end = period_data.index[-1]
            
            # If this is the first period or we're at a rebalancing point
            if last_end_date is None or (rebalance_freq != "None" and last_end_date != period_end):
                # For the first period, use all prior data for calculating weights
                if last_end_date is None:
                    lookback_data = returns_df.loc[:period_end]
                else:
                    # For subsequent periods, use data since the last rebalance date
                    # This gives more weight to recent data while maintaining some history
                    lookback_months = 12  # Use up to 12 months of data for rebalancing decisions
                    lookback_start = period_end - pd.DateOffset(months=lookback_months)
                    lookback_data = returns_df.loc[lookback_start:period_end]
                
                # Calculate new weights for this period
                current_weights = calculate_weights(lookback_data)
                
                # Store this rebalance date
                rebalance_dates.append(period_end)
                
                # Store weights for this period
                all_weights[period_end] = current_weights
            
            # Calculate returns for this period using current weights
            for ticker, weight in current_weights.items():
                if ticker in period_data.columns:
                    portfolio_returns.loc[period_data.index] += period_data[ticker] * weight
            
            # Update last end date
            last_end_date = period_end
        
        # Use the most recent weights as the final weights
        weights = current_weights if current_weights else calculate_weights(returns_df)
    
    # Calculate cumulative portfolio returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod() - 1
    
    result = {
        'weights': weights,
        'returns_df': returns_df,
        'portfolio_returns': portfolio_returns,
        'portfolio_cumulative': portfolio_cumulative,
        'start_date': start_date,  # Store the start date
        'end_date': end_date,      # Store the end date
    }
    
    # Add rebalance information if applicable
    if rebalance_freq != "None":
        result['rebalance_dates'] = rebalance_dates
        result['all_weights'] = all_weights
    
    return result

def calculate_risk_metrics(portfolio_returns: pd.Series) -> Dict:
    """
    Calculates comprehensive risk and performance metrics for a portfolio return series.
    
    This function computes a suite of financial analytics to evaluate portfolio performance, 
    including return metrics, risk metrics, and risk-adjusted performance measures. The metrics
    are calculated on an annualized basis assuming 252 trading days per year.
    
    Args:
        portfolio_returns (pd.Series): Time series of daily portfolio returns, with dates as index
    
    Returns:
        Dict: Dictionary containing the following risk and performance metrics:
          - annualized_return (float): Annualized portfolio return
          - annualized_volatility (float): Annualized standard deviation of returns
          - sharpe_ratio (float): Ratio of excess return to volatility (assumes 0% risk-free rate)
          - max_drawdown (float): Maximum peak-to-trough decline in portfolio value
          - var_95 (float): Value at Risk at 95% confidence level (historical method)
          - cvar_95 (float): Conditional Value at Risk (Expected Shortfall) at 95% confidence level
    
    Example:
        >>> portfolio_returns = pd.Series([0.001, -0.002, 0.003, ...], index=[date1, date2, ...])
        >>> metrics = calculate_risk_metrics(portfolio_returns)
        >>> print(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
        >>> print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    Notes:
        - All return-based metrics are expressed as decimals (not percentages)
        - VAR and CVAR represent negative returns (losses) and are typically negative values
    """
    # Daily metrics
    daily_mean = portfolio_returns.mean()
    daily_std = portfolio_returns.std()
    
    # Annualized metrics (assuming 252 trading days per year)
    annual_factor = 252
    annual_return = (1 + daily_mean)**annual_factor - 1
    annual_volatility = daily_std * np.sqrt(annual_factor)
    
    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    # Calculate maximum drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    
    # Value at Risk (VaR) - Historical method
    var_95 = portfolio_returns.quantile(0.05)
    
    # Conditional Value at Risk (CVaR)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    
    return {
        'annualized_return': annual_return,
        'annualized_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'cvar_95': cvar_95
    }

def plot_portfolio_performance(portfolio_data: Dict, tickers: List[str], 
                              risk_metrics: Dict, method: str, 
                              start_date: str = None, end_date: str = None, 
                              save_path: str = None) -> None:
    """
    Creates and saves comprehensive visualization of portfolio performance and risk metrics.
    
    This function generates a multi-panel visualization showing:
    1. Portfolio asset allocation (bar chart of weights)
    2. Cumulative return over time (line chart)
    3. Rolling volatility to show changing risk profile (line chart)
    4. Drawdown chart to visualize periods of decline (area chart)
    
    All visualizations include embedded risk metrics and proper labeling for analysis.
    
    Args:
        portfolio_data (Dict): Portfolio data dictionary from construct_portfolio():
            - Must contain 'weights', 'portfolio_returns', and 'portfolio_cumulative' keys
        tickers (List[str]): List of currency pair tickers included in the portfolio
        risk_metrics (Dict): Risk metrics dictionary from calculate_risk_metrics()
        method (str): Portfolio construction method used ("equal", "min_var", or "max_sharpe")
        start_date (str, optional): Start date in 'YYYY-MM-DD' format for title/filename
        end_date (str, optional): End date in 'YYYY-MM-DD' format for title/filename
        save_path (str, optional): Directory path to save visualization. If None, displays plot instead
    
    Returns:
        None: Function either saves visualization to file or displays it
    
    Example:
        >>> portfolio_data = construct_portfolio(tickers, start_date, end_date, method="min_var")
        >>> risk_metrics = calculate_risk_metrics(portfolio_data["portfolio_returns"])
        >>> plot_portfolio_performance(
        ...     portfolio_data=portfolio_data, 
        ...     tickers=tickers, 
        ...     risk_metrics=risk_metrics, 
        ...     method="min_var",
        ...     start_date="2022-01-01", 
        ...     end_date="2023-01-01",
        ...     save_path="./reports/figures"
        ... )
    
    Notes:
        - The figure dimensions and layout are optimized for A4 printing/saving
        - When many assets are in the portfolio, only the first 3 are shown in the title
    """
    # Extract data
    portfolio_returns = portfolio_data['portfolio_returns']
    portfolio_cumulative = portfolio_data['portfolio_cumulative']
    weights = portfolio_data['weights']
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(15, 20))
    
    # Plot 1: Portfolio weights
    ax1 = plt.subplot(4, 1, 1)
    ax1.bar(list(weights.keys()), list(weights.values()))
    ax1.set_title(f'Portfolio Weights ({method.replace("_", " ").title()} Weighting)')
    ax1.set_xlabel('Currency Pairs')
    ax1.set_ylabel('Weight')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 2: Cumulative return
    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(portfolio_cumulative.index, portfolio_cumulative * 100)
    ax2.set_title('Portfolio Cumulative Return (%)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Return (%)')
    ax2.grid(True)
    
    # Plot 3: Rolling volatility (30-day window)
    ax3 = plt.subplot(4, 1, 3)
    rolling_vol = portfolio_returns.rolling(window=30).std() * np.sqrt(252) * 100
    ax3.plot(rolling_vol.index, rolling_vol)
    ax3.set_title('30-Day Rolling Annualized Volatility (%)')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Volatility (%)')
    ax3.grid(True)
    
    # Plot 4: Drawdown
    ax4 = plt.subplot(4, 1, 4)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    ax4.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.3)
    ax4.set_title('Portfolio Drawdown (%)')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Drawdown (%)')
    ax4.grid(True)
    
    # Overall title
    date_range = f"({start_date} to {end_date})" if start_date and end_date else ""
    plt.suptitle(f"Portfolio Performance: {', '.join(tickers[:3])}{' + ' + str(len(tickers) - 3) + ' more' if len(tickers) > 3 else ''} {date_range}", 
                 fontsize=16, y=0.995)
    
    # Add risk metrics as text
    metrics_text = (
        f"Annualized Return: {risk_metrics['annualized_return']*100:.2f}%\n"
        f"Annualized Volatility: {risk_metrics['annualized_volatility']*100:.2f}%\n"
        f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}\n"
        f"Maximum Drawdown: {risk_metrics['max_drawdown']*100:.2f}%\n"
        f"VaR (95%): {risk_metrics['var_95']*100:.2f}%\n"
        f"CVaR (95%): {risk_metrics['cvar_95']*100:.2f}%"
    )
    fig.text(0.15, 0.01, metrics_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save or show
    if save_path:
        method_name = method.replace('_', '-')
        ticker_str = '-'.join(tickers[:3])
        if len(tickers) > 3:
            ticker_str += f"-plus-{len(tickers)-3}"
        
        filename = f"portfolio_{method_name}_{ticker_str}.png"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path)
        print(f"Portfolio chart saved to {full_path}")
    else:
        plt.show()

def create_correlation_heatmap(returns_df: pd.DataFrame, save_path: str = None) -> None:
    """
    Creates a correlation heatmap of asset returns.
    
    Args:
        returns_df: DataFrame of asset returns
        save_path: Path to save the heatmap to
    """
    plt.figure(figsize=(10, 8))
    corr_matrix = returns_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                vmin=-1, vmax=1, center=0, linewidths=.5)
    
    plt.title('Asset Return Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        filename = "correlation_heatmap.png"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path)
        print(f"Correlation heatmap saved to {full_path}")
    else:
        plt.show()

def generate_strategy_scatter_plot(tickers: List[str], start_date=None, end_date=None, rebalance_freq="None"):
    """
    Generates a scatter plot comparing different portfolio construction strategies based on risk and return metrics.
    
    This function:
    1. Constructs portfolios using different strategies (Equal, Min Variance, Max Sharpe)
    2. Calculates risk/return metrics for each
    3. Creates an interactive scatter plot for visual comparison
    
    Args:
        tickers (List[str]): List of currency pair tickers
        start_date (str, optional): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format
        rebalance_freq (str): Portfolio rebalancing frequency ("None", "Monthly", "Quarterly")
        
    Returns:
        plotly.graph_objects.Figure: Interactive scatter plot showing risk/return profiles
    """
    # List to store results for each strategy
    results = []
    
    # Define the strategies to test
    strategies = [
        {"id": "equal", "name": "Equal Weight"},
        {"id": "min_var", "name": "Minimum Variance"},
        {"id": "max_sharpe", "name": "Maximum Sharpe"}
    ]
    
    # For each strategy, construct portfolio and calculate metrics
    for strategy in strategies:
        try:
            # Construct portfolio with this strategy
            portfolio_data = construct_portfolio(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                method=strategy["id"],
                rebalance_freq=rebalance_freq
            )
            
            # Calculate risk metrics
            risk_metrics = calculate_risk_metrics(portfolio_data["portfolio_returns"])
            
            # Store results
            results.append({
                "Strategy": strategy["name"],
                "Return": risk_metrics["annualized_return"],
                "Volatility": risk_metrics["annualized_volatility"],
                "Sharpe": risk_metrics["sharpe_ratio"],
                "Max Drawdown": risk_metrics["max_drawdown"],
                "VaR 95%": risk_metrics["var_95"],
                "CVaR 95%": risk_metrics["cvar_95"]
            })
        except Exception as e:
            print(f"Error analyzing {strategy['name']} strategy: {str(e)}")
    
    # Convert results to DataFrame
    df_plot = pd.DataFrame(results)
    
    # Format percentage values for display
    for col in ["Return", "Volatility", "Max Drawdown", "VaR 95%", "CVaR 95%"]:
        if col in df_plot.columns:
            df_plot[f"{col}_fmt"] = df_plot[col].apply(lambda x: f"{x*100:.2f}%")
    
    # Create scatter plot
    fig = px.scatter(
        df_plot,
        x="Volatility", y="Return",
        color="Strategy",
        size=[12] * len(df_plot),  # Consistent size
        hover_data={
            "Strategy": True,
            "Return_fmt": True,
            "Volatility_fmt": True,
            "Sharpe": ":.2f",
            "Max Drawdown_fmt": True,
            "VaR 95%_fmt": True,
            "CVaR 95%_fmt": True,
            "Return": False,
            "Volatility": False
        },
        labels={
            "Return": "Annualized Return",
            "Volatility": "Annualized Volatility",
            "Return_fmt": "Annualized Return",
            "Volatility_fmt": "Annualized Volatility",
            "Max Drawdown_fmt": "Maximum Drawdown",
            "VaR 95%_fmt": "Value at Risk (95%)",
            "CVaR 95%_fmt": "Conditional VaR (95%)"
        },
        title="Risk/Return Strategy Comparison"
    )
    
    # Update layout for better visualization
    fig.update_traces(
        marker=dict(size=18, line=dict(width=2, color='DarkSlateGrey')),
        selector=dict(mode='markers')
    )
    
    # Custom layout improvements
    fig.update_layout(
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        xaxis=dict(tickformat='.1%'),  # Format as percentages
        yaxis=dict(tickformat='.1%'),
        legend_title="Strategy",
        height=500,
        template="plotly_white",
        hovermode='closest'
    )
    
    # Add a reference line for the risk-free rate (assuming 0% for simplicity)
    # This helps visualize the efficient frontier concept
    risk_free_rate = 0
    max_vol = df_plot["Volatility"].max() * 1.1  # Extend line beyond the rightmost point
    
    # Add reference lines
    fig.add_shape(
        type="line",
        x0=0, y0=risk_free_rate,
        x1=max_vol, y1=risk_free_rate,
        line=dict(color="grey", width=1, dash="dash")
    )
    
    return fig

if __name__ == "__main__":
    # Example usage
    # Get available currencies
    all_currencies = get_available_currencies()
    
    # Select a subset of currencies for testing
    test_tickers = all_currencies[:5]  # First 5 currencies
    print(f"Testing with currencies: {test_tickers}")
    
    # Set date range for test
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    # Test all portfolio construction methods
    for method in ["equal", "min_var", "max_sharpe"]:
        print(f"\nConstructing {method} portfolio...")
        
        # Build portfolio
        portfolio_data = construct_portfolio(
            tickers=test_tickers,
            start_date=start_date,
            end_date=end_date,
            method=method
        )
        
        # Calculate risk metrics
        risk_metrics = calculate_risk_metrics(portfolio_data["portfolio_returns"])
        
        # Print results
        print(f"\nPortfolio Weights ({method}):")
        for ticker, weight in portfolio_data["weights"].items():
            print(f"  {ticker}: {weight:.4f}")
        
        print("\nRisk Metrics:")
        for metric, value in risk_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Plot portfolio performance
        plot_portfolio_performance(
            portfolio_data=portfolio_data,
            tickers=test_tickers,
            risk_metrics=risk_metrics,
            method=method,
            start_date=start_date,
            end_date=end_date,
            save_path="."
        )
    
    # Create correlation heatmap
    create_correlation_heatmap(portfolio_data["returns_df"], save_path=".")
    
    # Create strategy comparison scatter plot
    strategy_fig = generate_strategy_scatter_plot(
        tickers=test_tickers,
        start_date=start_date,
        end_date=end_date
    )
    strategy_fig.write_image("strategy_comparison.png")
    print("Strategy comparison plot saved as 'strategy_comparison.png'") 