"""
ForexQuant - Currency Portfolio & Risk Analytics Platform

Streamlit frontend for constructing and analyzing currency portfolios.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import io

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import core functionality
from core.data_processing import load_currency_data, get_available_currencies, get_common_start_date
from core.portfolio import construct_portfolio, calculate_risk_metrics, create_correlation_heatmap, generate_strategy_scatter_plot

# Page config
st.set_page_config(
    page_title="ForexQuant - Currency Portfolio & Risk Analytics",
    page_icon="📈",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 15px;
        margin: 5px 0;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    .stPlotlyChart {
        margin-top: 10px;
    }
    .main-header {
        font-size: 30px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .selected-pairs {
        font-size: 14px;
        color: #1E88E5;
        padding: 8px;
        background-color: #f9f9f9;
        border-radius: 4px;
        margin: 5px 0;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: #666;
        text-align: center;
        padding: 5px;
        font-size: 12px;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<div class="main-header">ForexQuant - Currency Portfolio & Risk Analytics</div>', unsafe_allow_html=True)

# Add last updated timestamp
st.caption(f"Last updated: {datetime.now().strftime('%d %b %Y, %H:%M')}")

# Initialize session state for storing portfolio data
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
if 'risk_metrics' not in st.session_state:
    st.session_state.risk_metrics = None
if 'selected_currencies' not in st.session_state:
    st.session_state.selected_currencies = None
if 'random_currencies' not in st.session_state:
    st.session_state.random_currencies = None

# Function to format percentage values
def format_pct(value):
    return f"{value * 100:.2f}%"

# Function to create metrics display
def display_metrics(risk_metrics):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_pct(risk_metrics["annualized_return"])}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Annualized Return</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_pct(risk_metrics["annualized_volatility"])}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Annualized Volatility</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{risk_metrics["sharpe_ratio"]:.2f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Sharpe Ratio</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_pct(risk_metrics["max_drawdown"])}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Maximum Drawdown</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_pct(risk_metrics["var_95"])}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Value at Risk (95%)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_pct(risk_metrics["cvar_95"])}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Conditional VaR (95%)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Function to plot portfolio cumulative returns
def plot_cumulative_returns(portfolio_data):
    cumulative_returns = portfolio_data['portfolio_cumulative']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=cumulative_returns * 100,
        mode='lines',
        name='Cumulative Return',
        line=dict(color='royalblue', width=2),
        hovertemplate='%{x|%d %b %Y}<br>Return: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Portfolio Cumulative Returns (%)',
        xaxis_title='Date',
        yaxis_title='Return (%)',
        template='plotly_white',
        height=400,
        font=dict(size=14),
        hovermode='x unified',
        showlegend=False
    )
    
    # Improved y-axis formatting
    fig.update_yaxes(ticksuffix='%')
    
    # Add range selector for better time navigation
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ])
        )
    )
    
    return fig

# Function to plot drawdowns
def plot_drawdown(portfolio_data):
    portfolio_returns = portfolio_data['portfolio_returns']
    
    # Calculate drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown * 100,
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        line=dict(color='crimson', width=1),
        hovertemplate='%{x|%d %b %Y}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Portfolio Drawdown (%)',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template='plotly_white',
        height=400,
        font=dict(size=14),
        hovermode='x unified',
        showlegend=False
    )
    
    # Improved y-axis formatting
    fig.update_yaxes(ticksuffix='%')
    
    # Add range selector for better time navigation
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ])
        )
    )
    
    return fig

# Function to plot portfolio weights
def plot_portfolio_weights(weights, all_weights=None):
    """
    Plot portfolio weights as a bar chart. If all_weights is provided from rebalancing,
    show final weights with an option to view weight evolution.
    
    Args:
        weights (dict): Dictionary of final portfolio weights
        all_weights (dict, optional): Dictionary of weights at each rebalance date
    """
    # Sort weights by value for better visualization
    sorted_weights = {k: v for k, v in sorted(weights.items(), key=lambda item: item[1], reverse=True)}
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(sorted_weights.keys()),
        y=[w * 100 for w in sorted_weights.values()],  # Convert to percentages
        marker_color='lightseagreen',
        hovertemplate='%{x}<br>Weight: %{y:.2f}%<extra></extra>'
    ))
    
    title = 'Portfolio Weights by Currency Pair'
    if all_weights:
        title += ' (Final Weights)'
    
    fig.update_layout(
        title=title,
        xaxis_title='Currency Pair',
        yaxis_title='Weight (%)',
        template='plotly_white',
        height=400,
        font=dict(size=14),
        xaxis={'tickangle': 45},
        hovermode='closest'
    )
    
    # Improved y-axis formatting
    fig.update_yaxes(ticksuffix='%')
    
    return fig

# Function to plot weight evolution over time
def plot_weight_evolution(all_weights, weight_threshold=0.0, selected_tickers=None, highlight_tickers=None):
    """
    Plot how weights evolved over rebalancing periods using a clear line chart.
    
    Args:
        all_weights (dict): Dictionary of weights at each rebalance date
        weight_threshold (float): Minimum weight threshold for inclusion in the chart
        selected_tickers (list): Optional list of selected tickers to display
        highlight_tickers (list): Optional list of tickers to highlight with bolder lines
    """
    if not all_weights:
        return None
    
    # Transform data for plotting
    # Instead of a stacked chart, create a long-format DataFrame for line plotting
    dates = []
    tickers = []
    weights = []
    
    # Get all unique tickers
    all_tickers = set()
    for date_weights in all_weights.values():
        all_tickers.update(date_weights.keys())
    
    # Filter by selected tickers if provided
    filtered_tickers = selected_tickers if selected_tickers else list(all_tickers)
    
    # Convert the dictionary structure to lists for DataFrame creation
    for date, date_weights in all_weights.items():
        for ticker, weight in date_weights.items():
            # Only include tickers that meet both the weight threshold and selection criteria
            if weight >= weight_threshold and ticker in filtered_tickers:
                dates.append(date)
                tickers.append(ticker)
                weights.append(weight * 100)  # Convert to percentage
    
    # Check if we have data after filtering
    if not dates:
        fig = go.Figure()
        fig.update_layout(
            title='No currency pairs match the filtering criteria',
            height=450,
            template='plotly_white',
            font=dict(size=14)
        )
        fig.add_annotation(
            text="Try lowering the weight threshold or selecting different currency pairs",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig
    
    # Create a DataFrame in long format suitable for plotly.express
    weight_df = pd.DataFrame({
        'Date': dates,
        'Ticker': tickers,
        'Weight': weights
    })
    
    # Create a dynamic line chart with plotly.express
    fig = px.line(
        weight_df, 
        x='Date', 
        y='Weight', 
        color='Ticker',
        markers=True,  # Add markers at each data point
        labels={'Weight': 'Weight (%)', 'Date': 'Rebalance Date'},
        title=f'Portfolio Weight Evolution Over Time ({len(filtered_tickers)} Pairs)',
        hover_data={'Weight': ':.2f'}  # Format weight as percentage in tooltip
    )
    
    # Highlight specific tickers if requested
    if highlight_tickers:
        for trace in fig.data:
            ticker = trace.name
            if ticker in highlight_tickers:
                trace.update(line=dict(width=4, dash=None))  # Make line bolder
            else:
                trace.update(line=dict(width=1.5, dash='dot'))  # Make others thinner and dotted
    
    # Customize the layout
    fig.update_layout(
        hovermode='x unified',  # Show all tickers at the same x position
        legend_title_text='Currency Pair',
        template='plotly_white',
        height=500,
        font=dict(size=14),
        xaxis_title='Rebalance Date',
        yaxis_title='Weight (%)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    # Improve y-axis labels to show as percentages
    fig.update_yaxes(ticksuffix='%')
    
    # Add range selector for better time navigation
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ])
        )
    )
    
    return fig

# Function to plot correlation heatmap
def plot_correlation_heatmap(returns_df):
    corr_matrix = returns_df.corr()
    
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        labels=dict(x="Currency Pair", y="Currency Pair", color="Correlation"),
        title="Currency Pair Correlations",
    )
    
    fig.update_layout(
        height=550,
        template='plotly_white',
        font=dict(size=14),
        coloraxis_colorbar=dict(
            title="Correlation",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"]
        )
    )
    
    # Add hover template with clearer correlation display
    for i in range(len(fig.data)):
        fig.data[i].update(
            hovertemplate='%{y} & %{x}<br>Correlation: %{z:.2f}<extra></extra>'
        )
    
    return fig

# Enhanced function to generate better strategy comparison plots
def generate_enhanced_strategy_plot(tickers, start_date, end_date, rebalance_freq, show_sharpe_lines=True):
    """
    Generate an enhanced scatter plot comparing different portfolio strategies.
    
    Args:
        tickers (list): List of currency pairs
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        rebalance_freq (str): Rebalancing frequency (None, Monthly, Quarterly)
        show_sharpe_lines (bool): Whether to show Sharpe ratio lines
        
    Returns:
        plotly.graph_objects.Figure: A scatter plot with risk-return metrics
    """
    # Get strategy comparison data from the core function
    strategy_fig = generate_strategy_scatter_plot(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        rebalance_freq=rebalance_freq
    )
    
    # Enhance the figure with better styling
    strategy_fig.update_layout(
        title="Strategy Comparison: Risk vs. Return",
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Return (%)",
        template="plotly_white",
        font=dict(size=14),
        legend_title_text="Strategy",
        height=500,
        hovermode="closest",
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Improve axis formatting
    strategy_fig.update_xaxes(
        ticksuffix="%",
        gridcolor="rgba(220, 220, 220, 0.5)",
        showgrid=True,
        zeroline=True,
        zerolinecolor="rgba(0, 0, 0, 0.2)",
        zerolinewidth=1
    )
    
    strategy_fig.update_yaxes(
        ticksuffix="%",
        gridcolor="rgba(220, 220, 220, 0.5)",
        showgrid=True,
        zeroline=True,
        zerolinecolor="rgba(0, 0, 0, 0.2)",
        zerolinewidth=1
    )
    
    # Add a diagonal line representing equal risk-reward (Sharpe ratio = 1)
    x_range = strategy_fig.layout.xaxis.range
    if not x_range:
        # If range is not set, estimate from the data
        x_data = [trace['x'][0] for trace in strategy_fig.data if len(trace['x']) > 0]
        if x_data:
            min_x = min(x_data) * 0.8
            max_x = max(x_data) * 1.2
        else:
            min_x, max_x = 0, 15  # Default range if no data
    else:
        min_x, max_x = x_range
    
    # Update marker sizes and add custom hover template
    for trace in strategy_fig.data:
        if trace.mode == 'markers' or trace.mode == 'markers+text':
            trace.update(
                marker=dict(
                    size=16, 
                    line=dict(width=2, color='DarkSlateGrey'),
                    symbol='circle'
                ),
                hovertemplate='<b>%{text}</b><br>Return: %{y:.2f}%<br>Volatility: %{x:.2f}%<br>Sharpe: %{customdata[0]:.2f}<br>Max Drawdown: %{customdata[1]:.2f}%<extra></extra>'
            )
    
    # Add Sharpe ratio reference lines
    if show_sharpe_lines:
        # Define Sharpe ratio values to display
        sharpe_values = [0.5, 1.0, 1.5, 2.0]
        
        # Add each Sharpe ratio line
        for sharpe in sharpe_values:
            # Add reference line
            strategy_fig.add_trace(go.Scatter(
                x=[0, max_x],
                y=[0, sharpe * max_x],
                mode='lines',
                line=dict(
                    color='rgba(100, 100, 100, 0.3)',
                    width=1.5,
                    dash='dot'
                ),
                name=f'SR = {sharpe}',
                hoverinfo='name',
                showlegend=True if sharpe == 1.0 else False  # Only show SR=1 in legend to avoid clutter
            ))
            
            # Add annotations
            strategy_fig.add_annotation(
                x=max_x * 0.85,
                y=sharpe * max_x * 0.85,
                text=f"SR = {sharpe}",
                showarrow=False,
                font=dict(size=11, color="rgba(100, 100, 100, 0.8)"),
                xanchor="left",
                yanchor="middle",
                textangle=np.degrees(np.arctan(sharpe))  # Align text with line angle
            )
    
    return strategy_fig

# Sidebar for inputs
with st.sidebar:
    st.title("Portfolio Settings")
    
    # Add Reset Button
    if st.button("🔄 Reset All Settings", help="Reset all settings to their default values"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()
    
    # Get available currencies
    available_currencies = get_available_currencies()
    
    # Currency selection mode
    selection_mode = st.selectbox(
        "Selection Mode",
        options=["🎯 Fixed", "🎲 Random", "✍️ Custom"],
        format_func=lambda x: x,
        help="Choose how to select currency pairs: Fixed (top N pairs), Random (randomly selected pairs), or Custom (manually select pairs)"
    )
    
    # Currency selection based on mode
    if selection_mode == "🎯 Fixed":
        num_currencies = st.selectbox(
            "Number of currency pairs",
            options=[3, 5, 10, 15, 20],
            index=1,  # Default to 5
            help="Select how many top currency pairs to include in your portfolio"
        )
        
        if len(available_currencies) >= num_currencies:
            selected_currencies = available_currencies[:num_currencies]
            st.markdown(f"<div class='selected-pairs'>Selected first {num_currencies} currency pairs:</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='selected-pairs'>{', '.join(selected_currencies)}</div>", unsafe_allow_html=True)
        else:
            selected_currencies = available_currencies
            st.markdown(f"<div class='selected-pairs'>Selected all {len(selected_currencies)} available pairs</div>", unsafe_allow_html=True)
            
    elif selection_mode == "🎲 Random":
        num_random = st.selectbox(
            "Number of random pairs",
            options=[5, 10],
            index=0,  # Default to 5
            help="Choose how many randomly selected currency pairs to include in your portfolio"
        )
        
        # Initialize random selection if not already in session state
        if 'random_currencies' not in st.session_state or st.session_state.random_currencies is None or len(st.session_state.random_currencies) != num_random:
            if len(available_currencies) >= num_random:
                st.session_state.random_currencies = random.sample(available_currencies, num_random)
            else:
                st.session_state.random_currencies = available_currencies
        
        if st.button("🔄 Re-randomize", help="Generate a new set of random currency pairs"):
            if len(available_currencies) >= num_random:
                st.session_state.random_currencies = random.sample(available_currencies, num_random)
            else:
                st.session_state.random_currencies = available_currencies
        
        selected_currencies = st.session_state.random_currencies
        st.markdown(f"<div class='selected-pairs'>Random {len(selected_currencies)} currency pairs:</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='selected-pairs'>{', '.join(selected_currencies)}</div>", unsafe_allow_html=True)
        
    else:  # Custom selection
        selected_currencies = st.multiselect(
            "Select currency pairs",
            options=available_currencies,
            default=available_currencies[:5] if len(available_currencies) >= 5 else available_currencies,
            help="Manually select which currency pairs to include in your portfolio analysis"
        )
        if selected_currencies:
            st.markdown(f"<div class='selected-pairs'>Selected {len(selected_currencies)} currency pairs</div>", unsafe_allow_html=True)
    
    # Date range selection with dynamic earliest date
    st.subheader("Date Range")
    
    # Get the earliest common date for the selected currencies
    min_date = get_common_start_date(selected_currencies)
    
    # Default dates - Smart defaults for 3 years of data
    latest_date = datetime.now().date()
    default_end_date = latest_date
    default_start_date = latest_date - timedelta(days=365*3)  # Default to 3 years
    
    # Ensure start_date is not earlier than min_date
    default_start_date = max(default_start_date, min_date)
    
    # Add a note about smart defaults
    st.markdown(f"<div class='selected-pairs' style='color: #4CAF50;'>📅 Using smart defaults: 3-year period ending today</div>", unsafe_allow_html=True)
    
    # Date inputs with DD/MM/YYYY display format
    start_date = st.date_input(
        "Start Date", 
        value=default_start_date, 
        min_value=min_date,
        help="Select the starting date for your analysis period (earliest available date depends on selected currencies)"
    )
    end_date = st.date_input(
        "End Date", 
        value=default_end_date, 
        min_value=start_date,
        help="Select the ending date for your analysis period"
    )
    
    # Show formatted dates in DD/MM/YYYY format
    st.markdown(f"<div class='selected-pairs'>Range: {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}</div>", unsafe_allow_html=True)
    
    # Portfolio strategy selection
    st.subheader("Portfolio Strategy")
    portfolio_method = st.selectbox(
        "Select portfolio construction method",
        options=["equal", "min_var", "max_sharpe"],
        index=2,  # Default to max_sharpe (index 2)
        format_func=lambda x: {
            "equal": "Equal Weighting",
            "min_var": "Minimum Variance",
            "max_sharpe": "Maximum Sharpe Ratio"
        }.get(x, x),
        help="Choose how to allocate weights to currency pairs: Equal Weighting (same weight to each pair), Minimum Variance (optimized for lowest volatility), or Maximum Sharpe Ratio (optimized for best risk-adjusted return)"
    )
    
    # Add note about optimal default
    st.markdown("<div class='selected-pairs' style='color: #4CAF50;'>📈 Default: Max Sharpe Ratio typically offers best risk-adjusted return</div>", unsafe_allow_html=True)
    
    # Add rebalancing options
    st.subheader("Rebalancing")
    rebalance_freq = st.selectbox(
        "Select rebalancing frequency",
        options=["None", "Monthly", "Quarterly"],
        index=2,  # Default to Quarterly (index 2)
        format_func=lambda x: {
            "None": "No Rebalancing",
            "Monthly": "Monthly Rebalancing",
            "Quarterly": "Quarterly Rebalancing"
        }.get(x, x),
        help="Rebalancing recalculates portfolio weights at regular intervals to maintain the strategy's target allocation"
    )
    
    # Add note about quarterly rebalancing
    st.markdown("<div class='selected-pairs' style='color: #4CAF50;'>🔄 Default: Quarterly rebalancing balances performance and trading costs</div>", unsafe_allow_html=True)
    
    # Add strategy comparison toggle
    st.subheader("Advanced Analytics")
    show_strategy_comparison = st.checkbox(
        "Show strategy comparison chart",
        value=True,
        help="Display a risk-return scatter plot comparing Equal Weighting, Minimum Variance, and Maximum Sharpe Ratio strategies"
    )
    
    # Construct portfolio button
    construct_button = st.button(
        "Construct Portfolio", 
        type="primary", 
        use_container_width=True,
        help="Generate portfolio analysis based on your selected currencies, date range, and strategy"
    )

# Error handling and validation
if construct_button:
    if not selected_currencies or len(selected_currencies) < 2:
        st.error("Please select at least 2 currency pairs.")
    elif start_date >= end_date:
        st.error("End date must be after start date.")
    else:
        # Show loading progress
        with st.spinner("Constructing portfolio..."):
            try:
                # Convert dates to string format for internal processing
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                
                # Construct portfolio
                portfolio_data = construct_portfolio(
                    tickers=selected_currencies,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    method=portfolio_method,
                    rebalance_freq=rebalance_freq
                )
                
                # Calculate risk metrics
                risk_metrics = calculate_risk_metrics(portfolio_data['portfolio_returns'])
                
                # Store in session state
                st.session_state.portfolio_data = portfolio_data
                st.session_state.risk_metrics = risk_metrics
                st.session_state.selected_currencies = selected_currencies
                
                st.success("Portfolio constructed successfully!")
            except Exception as e:
                st.error(f"Error constructing portfolio: {str(e)}")

# Display portfolio analysis if data is available
if st.session_state.portfolio_data and st.session_state.risk_metrics:
    portfolio_data = st.session_state.portfolio_data
    risk_metrics = st.session_state.risk_metrics
    
    # Display metrics
    st.markdown('<div class="section-header">Portfolio Performance Metrics</div>', unsafe_allow_html=True)
    display_metrics(risk_metrics)
    
    # Add export button for portfolio metrics
    # Create a DataFrame with all metrics for download
    metrics_data = {
        "Metric": [
            "Annualized Return",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Maximum Drawdown",
            "Value at Risk (95%)",
            "Conditional Value at Risk (95%)",
            "Construction Method",
            "Rebalancing Frequency",
            "Number of Currency Pairs",
            "Date Range"
        ],
        "Value": [
            f"{risk_metrics['annualized_return'] * 100:.2f}%",
            f"{risk_metrics['annualized_volatility'] * 100:.2f}%",
            f"{risk_metrics['sharpe_ratio']:.2f}",
            f"{risk_metrics['max_drawdown'] * 100:.2f}%",
            f"{risk_metrics['var_95'] * 100:.2f}%",
            f"{risk_metrics['cvar_95'] * 100:.2f}%",
            portfolio_method.title().replace("_", " "),
            rebalance_freq,
            len(selected_currencies),
            f"{start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}"
        ]
    }
    
    # Add portfolio currency pairs as additional rows
    for i, currency in enumerate(selected_currencies[:10], 1):  # Limit to first 10 pairs
        weight = portfolio_data['weights'].get(currency, 0)
        metrics_data["Metric"].append(f"Currency Pair {i}: {currency}")
        metrics_data["Value"].append(f"{weight * 100:.2f}%")
    
    # If there are more pairs, add a note
    if len(selected_currencies) > 10:
        metrics_data["Metric"].append("Note")
        metrics_data["Value"].append(f"{len(selected_currencies) - 10} more pairs not shown")
    
    # Create DataFrame
    df_metrics = pd.DataFrame(metrics_data)
    
    # Add timestamp for the report
    report_time = datetime.now().strftime("%d %b %Y, %H:%M")
    metrics_data["Metric"].append("Report Generated")
    metrics_data["Value"].append(report_time)
    
    # Convert to CSV for download
    csv = df_metrics.to_csv(index=False).encode('utf-8')
    
    # Add download buttons with appropriate styling
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        st.download_button(
            "📊 Download CSV",
            csv,
            f"forexquant_portfolio_{portfolio_method}_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            help="Download portfolio metrics as a CSV file for reporting or further analysis"
        )
    
    with col2:
        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write metrics to first sheet
            df_metrics.to_excel(writer, sheet_name='Portfolio Metrics', index=False)
            
            # Write returns data to second sheet
            returns_data = pd.DataFrame({
                'Date': portfolio_data['portfolio_returns'].index,
                'Daily Return (%)': portfolio_data['portfolio_returns'] * 100,
                'Cumulative Return (%)': portfolio_data['portfolio_cumulative'] * 100
            })
            returns_data.to_excel(writer, sheet_name='Returns Data', index=False)
            
            # Write final weights to third sheet
            weights_data = pd.DataFrame({
                'Currency Pair': list(portfolio_data['weights'].keys()),
                'Weight (%)': [w * 100 for w in portfolio_data['weights'].values()]
            })
            weights_data.to_excel(writer, sheet_name='Portfolio Weights', index=False)
            
            # Add some formatting
            workbook = writer.book
            worksheet = writer.sheets['Portfolio Metrics']
            
            # Add percent format for relevant cells
            percent_format = workbook.add_format({'num_format': '0.00%'})
            
            # Apply formatting (column B contains values)
            for row_num, value in enumerate(df_metrics['Value']):
                if '%' in str(value):
                    worksheet.write_string(row_num + 1, 1, value)
        
        # Get the Excel data
        excel_data = output.getvalue()
        
        # Add Excel download button
        st.download_button(
            "📊 Download Excel",
            excel_data,
            f"forexquant_portfolio_{portfolio_method}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            "application/vnd.ms-excel",
            help="Download complete portfolio analysis as an Excel file with multiple sheets"
        )
    
    # Strategy explanation
    with st.expander("What do these strategies mean?"):
        st.markdown("""
        ### Portfolio Construction Strategies
        
        - **Equal Weighting**: Each currency pair receives the same portfolio weight (1/n), regardless of its risk or return characteristics. Simple but can be effective for diversification.
        
        - **Minimum Variance**: Optimizes weights to create the portfolio with the lowest possible volatility. This approach prioritizes risk reduction over return maximization.
        
        - **Maximum Sharpe Ratio**: Seeks the optimal balance between risk and return by maximizing the Sharpe ratio (excess return per unit of risk). This typically creates the most efficient portfolio according to Modern Portfolio Theory.
        
        ### Rebalancing Frequency
        
        - **No Rebalancing**: Weights are set once at the beginning and allowed to drift with market movements.
        
        - **Monthly/Quarterly Rebalancing**: Portfolio weights are reset to target allocations at regular intervals, which helps maintain the intended risk profile and can capture mean reversion effects.
        """)
    
    # Strategy Comparison Chart (if enabled)
    if show_strategy_comparison:
        st.markdown('<div class="section-header">Strategy Comparison: Risk vs. Return</div>', unsafe_allow_html=True)
        
        # Add a toggle for Sharpe ratio lines
        show_sr_lines = st.checkbox("Show Sharpe Ratio Guide Lines", value=True, 
                                   help="Display reference lines showing different Sharpe ratio levels")
        
        with st.spinner("Generating strategy comparison..."):
            try:
                # Get the current date settings
                start_date_str = portfolio_data.get('start_date') or start_date.strftime('%Y-%m-%d')
                end_date_str = portfolio_data.get('end_date') or end_date.strftime('%Y-%m-%d')
                
                # Generate the strategy comparison chart
                strategy_fig = generate_enhanced_strategy_plot(
                    tickers=selected_currencies,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    rebalance_freq=rebalance_freq,
                    show_sharpe_lines=show_sr_lines
                )
                
                # Display the chart
                st.plotly_chart(strategy_fig, use_container_width=True)
                
                # Add explanatory text
                st.markdown("""
                <div style="font-size: 14px; color: #666;">
                This chart compares the risk-return profiles of different portfolio construction strategies:
                <ul>
                <li><strong>Top-left area</strong>: Higher return with lower risk (optimal)</li>
                <li><strong>Bottom-right area</strong>: Lower return with higher risk (suboptimal)</li>
                <li>The dotted lines represent constant Sharpe ratio values (return ÷ risk)</li>
                <li>Strategies above the SR=1 line deliver more than 1 unit of return per unit of risk</li>
                </ul>
                <em>Hover over each point to see detailed metrics including Sharpe ratio and maximum drawdown</em>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generating strategy comparison: {str(e)}")
    
    # Portfolio weights
    st.markdown('<div class="section-header">Portfolio Weights</div>', unsafe_allow_html=True)
    
    # Get all_weights if available
    all_weights = portfolio_data.get('all_weights', None)
    
    # Plot final weights
    weights_fig = plot_portfolio_weights(portfolio_data['weights'], all_weights)
    st.plotly_chart(weights_fig, use_container_width=True)
    
    # Show weight evolution if rebalancing was applied
    if all_weights:
        st.markdown('<div class="section-header">Weight Evolution</div>', unsafe_allow_html=True)
        
        # Create two columns for filters
        col1, col2 = st.columns(2)
        
        with col1:
            # Add weight threshold filter
            weight_threshold = st.slider(
                "Show pairs with weight above (%)", 
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.5,
                help="Filter out currency pairs with weights below this threshold to focus on significant allocations"
            ) / 100.0  # Convert from percentage to decimal
        
        # Get the list of all unique tickers in the weights
        all_tickers_set = set()
        for date_weights in all_weights.values():
            all_tickers_set.update(date_weights.keys())
        all_tickers_list = sorted(list(all_tickers_set))
        
        with col2:
            # Add ticker selection filter with a checkbox for "Show All"
            show_all_tickers = st.checkbox(
                "Show all currency pairs", 
                value=True,
                help="Display all currency pairs in the weight evolution chart, regardless of weight"
            )
            
            selected_tickers = None
            if not show_all_tickers:
                # Let user select specific currency pairs
                selected_tickers = st.multiselect(
                    "Select specific currency pairs",
                    options=all_tickers_list,
                    default=all_tickers_list[:5] if len(all_tickers_list) > 5 else all_tickers_list,
                    help="Choose which currency pairs to show in the weight evolution chart"
                )
        
        # Add highlight option for focusing on specific pairs
        highlight_pairs = st.multiselect(
            "Highlight specific currency pairs (optional)",
            options=all_tickers_list,
            default=None,
            help="Select currency pairs to highlight with bolder lines for easier comparison"
        )
        
        # Generate weight evolution chart with filters
        weight_evolution_fig = plot_weight_evolution(all_weights, weight_threshold, selected_tickers, highlight_pairs)
        st.plotly_chart(weight_evolution_fig, use_container_width=True)
        
        # Add a download button for the chart data
        if st.button(
            "📥 Download Weight Evolution Data", 
            help="Save the weight evolution data to a CSV file for external analysis"
        ):
            # Create a DataFrame for download
            download_data = []
            for date, date_weights in all_weights.items():
                for ticker, weight in date_weights.items():
                    # Only include data that matches the current filters
                    if (weight >= weight_threshold and 
                        (selected_tickers is None or ticker in selected_tickers)):
                        download_data.append({
                            'Date': date.strftime('%Y-%m-%d'),
                            'Ticker': ticker,
                            'Weight': weight
                        })
            
            if download_data:
                download_df = pd.DataFrame(download_data)
                csv = download_df.to_csv(index=False)
                st.download_button(
                    label="Save CSV",
                    data=csv,
                    file_name="portfolio_weight_evolution.csv",
                    mime="text/csv"
                )
    
    # Show rebalancing information if applicable
    if 'rebalance_dates' in portfolio_data and portfolio_data['rebalance_dates']:
        st.markdown('<div class="section-header">Rebalancing Information</div>', unsafe_allow_html=True)
        rebalance_dates = portfolio_data['rebalance_dates']
        
        # Create a string of formatted dates
        formatted_dates = [date.strftime('%d/%m/%Y') for date in rebalance_dates]
        
        st.markdown(f"<div class='selected-pairs'>Portfolio was rebalanced {len(rebalance_dates)} times</div>", unsafe_allow_html=True)
        
        # Show the dates in a compact way
        if len(formatted_dates) <= 10:
            st.markdown(f"<div class='selected-pairs'>Rebalance dates: {', '.join(formatted_dates)}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='selected-pairs'>First rebalance: {formatted_dates[0]}, Last rebalance: {formatted_dates[-1]}</div>", unsafe_allow_html=True)
            with st.expander("View all rebalance dates"):
                # Create a DataFrame for better display
                rebalance_df = pd.DataFrame({
                    'Date': rebalance_dates,
                    'Formatted Date': formatted_dates
                })
                st.dataframe(rebalance_df[['Formatted Date']], use_container_width=True)
    
    # Cumulative returns and drawdown charts in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">Cumulative Returns</div>', unsafe_allow_html=True)
        returns_fig = plot_cumulative_returns(portfolio_data)
        st.plotly_chart(returns_fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">Drawdown Analysis</div>', unsafe_allow_html=True)
        drawdown_fig = plot_drawdown(portfolio_data)
        st.plotly_chart(drawdown_fig, use_container_width=True)
    
    # Correlation heatmap
    st.markdown('<div class="section-header">Currency Pair Correlations</div>', unsafe_allow_html=True)
    corr_fig = plot_correlation_heatmap(portfolio_data['returns_df'])
    st.plotly_chart(corr_fig, use_container_width=True)
    
    # Optional: Add a data table showing the portfolio returns
    # Add tooltip as text comment instead of using help parameter (not supported in expander)
    # Tooltip info: Show daily and cumulative returns for the constructed portfolio
    with st.expander(
        "View Portfolio Returns Data ℹ️",
        expanded=False
    ):
        returns_df = pd.DataFrame({
            'Date': portfolio_data['portfolio_returns'].index,
            'Daily Return (%)': portfolio_data['portfolio_returns'] * 100,
            'Cumulative Return (%)': portfolio_data['portfolio_cumulative'] * 100
        }).set_index('Date')
        
        st.dataframe(returns_df.tail(100), use_container_width=True)

else:
    # First-time instructions
    st.info("""
    ### Welcome to ForexQuant!
    
    To get started:
    1. Select currency pairs from the sidebar
    2. Choose your date range
    3. Select a portfolio construction method
    4. Click "Construct Portfolio"
    
    You'll see a complete dashboard with portfolio analysis once constructed.
    """)
    
    # Display sample chart as placeholder
    st.markdown('<div class="section-header">Sample Chart (For Demonstration)</div>', unsafe_allow_html=True)
    
    # Create sample data
    dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='D')
    values = np.cumsum(np.random.normal(0.0005, 0.01, size=len(dates)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=values * 100,
        mode='lines',
        name='Sample Return',
        line=dict(color='royalblue', width=2)
    ))
    
    fig.update_layout(
        title='Sample Cumulative Return (Demo Only)',
        xaxis_title='Date',
        yaxis_title='Return (%)',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Add footer with timestamp
st.markdown('<div class="footer">ForexQuant Analytics Platform | ' + 
            f'Last updated: {datetime.now().strftime("%d %b %Y, %H:%M")} | ' +
            'Data source: Currency API</div>', 
            unsafe_allow_html=True) 