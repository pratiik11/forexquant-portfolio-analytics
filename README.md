# 📈 ForexQuant - Currency Portfolio & Risk Analytics

A comprehensive **Streamlit-powered** platform for constructing and analyzing currency portfolios using modern portfolio theory. Perfect for finance students, portfolio managers, and quant researchers who want to build & backtest real FX strategies.

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.27.0-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/forexquant-project.git
cd forexquant-project
pip install -r requirements.txt
streamlit run app/main.py
```

**➡️ Launch in seconds and start building currency portfolios!**

---

## ✨ Key Features

### 📊 **Portfolio Construction**
- **🎯 Equal Weighting** - Simple 1/n allocation across currency pairs
- **⚖️ Minimum Variance** - Optimized for lowest portfolio volatility using modern portfolio theory
- **🎯 Maximum Sharpe Ratio** - Optimized for best risk-adjusted returns

### 🔄 **Dynamic Rebalancing**
- **Monthly** or **Quarterly** rebalancing with automated weight recalculation
- Visual weight evolution tracking over time
- Performance comparison with and without rebalancing

### 📈 **Risk Analytics**
- **Annualized Return & Volatility** calculations
- **Sharpe Ratio** optimization and tracking
- **Value at Risk (VaR)** and **Conditional VaR** at 95% confidence
- **Maximum Drawdown** analysis
- **Rolling volatility** with 7-day and 30-day windows

### 🎨 **Interactive Visualizations**
- **Real-time** cumulative return charts with date range selectors
- **Correlation heatmaps** between currency pairs
- **Strategy comparison** scatter plots (Risk vs Return)
- **Portfolio weight** pie charts and evolution tracking
- **Drawdown** analysis with visual indicators

### 📂 **Data Management**
- **100+ currency pairs** included (USD, EUR, GBP, JPY, AUD, CAD, CHF, and more)
- **Daily OHLC data** going back several years
- **Automated data updates** via forex APIs
- **Smart data caching** for improved performance

### 💾 **Export & Analytics**
- **Excel download** with multi-sheet portfolio analysis
- **CSV export** for weight evolution and returns data
- **Complete metrics** dashboard with professional formatting

---

## 🎯 Who Is This For?

### 👨‍🎓 **Finance Students**
Learn modern portfolio theory hands-on with real FX data. Perfect for assignments on:
- Portfolio optimization
- Risk management
- Currency hedging strategies

### 👨‍💼 **Portfolio Managers**
- Backtest currency allocation strategies
- Compare different rebalancing frequencies
- Analyze correlation structures in FX markets
- Generate client-ready reports

### 🔬 **Quant Researchers** 
- Rapid prototyping of FX strategies
- Historical backtesting with multiple optimization methods
- Risk factor analysis across currency pairs

### 🏦 **Finance Professionals**
- Educational tool for explaining portfolio concepts to clients
- Quick analysis for tactical currency allocation decisions

---

## 📁 Project Structure

```bash
forexquant-project/
├── app/
│   └── main.py              # Streamlit frontend application
├── core/
│   ├── data_processing.py   # Data loading and preprocessing
│   ├── portfolio.py         # Portfolio construction & risk metrics
│   └── data_updater.py      # Automated data updates
├── Data/                    # 100+ currency pair CSV files
│   ├── USDEUR.csv
│   ├── GBPJPY.csv
│   └── ...
├── tests/                   # Unit tests
├── requirements.txt         # Python dependencies
├── .env                     # API keys (optional)
└── README.md
```

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.9+**
- **pip** package manager

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/forexquant-project.git
cd forexquant-project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch Application
```bash
streamlit run app/main.py
```

### Step 4: Open in Browser
The app will automatically open at `http://localhost:8501`

---

## 📊 Quick Tutorial

### 1. **Select Currency Pairs**
Choose from 3 modes:
- **🎯 Fixed**: Select top N currency pairs (3, 5, 10, 15, or 20)
- **🎲 Random**: Randomly sample pairs for diversified testing
- **✍️ Custom**: Hand-pick specific currency pairs

### 2. **Set Date Range**
- Smart defaults: 3-year analysis period ending today
- Minimum date automatically calculated based on data availability
- Visual date range confirmation

### 3. **Choose Strategy**
- **Equal Weighting**: Balanced 1/n allocation
- **Minimum Variance**: Risk-minimizing allocation
- **Maximum Sharpe Ratio**: Risk-adjusted optimal allocation ⭐ **(Recommended)**

### 4. **Configure Rebalancing**
- **No Rebalancing**: Set-and-forget approach
- **Monthly**: Active weight management
- **Quarterly**: Balanced approach ⭐ **(Recommended)**

### 5. **Analyze Results**
- View comprehensive performance metrics
- Compare strategies with interactive charts
- Export data for further analysis

---

## 🧮 Core Algorithms

### **Modern Portfolio Theory**
- **Mean-Variance Optimization** using the `riskfolio-lib` library
- **Efficient Frontier** calculation for risk-return optimization
- **Correlation-based** diversification analysis

### **Risk Metrics**
- **Sharpe Ratio**: `(Return - Risk_Free_Rate) / Volatility`
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Value at Risk**: 5th percentile of daily returns
- **Conditional VaR**: Expected loss beyond VaR threshold

### **Data Processing**
- **Log returns** for portfolio aggregation
- **252-day annualization** factor for volatility scaling
- **Missing data handling** with forward-fill and alignment

---

## 🔧 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | 1.27.0 | Web application framework |
| `pandas` | 2.1.0 | Data manipulation and analysis |
| `numpy` | 1.26.0 | Numerical computations |
| `plotly` | 5.16.0 | Interactive visualizations |
| `riskfolio-lib` | 7.0.1 | Portfolio optimization |
| `matplotlib` | 3.8.0 | Static plotting |
| `seaborn` | 0.13.0 | Statistical visualizations |
| `requests` | 2.31.0 | API data fetching |

---

## 🔄 Data Updates (Optional)

ForexQuant includes automated data updating capabilities:

### **Manual Update**
```bash
python core/data_updater.py
```

### **Scheduled Updates (Windows)**
```bash
# Run the included batch file
update_forex_data.bat
```

### **API Configuration**
Add your API keys to `.env` file:
```bash
TWELVEDATA_API_KEY=your_api_key_here
```

---

## 📈 Example Analysis

### **Sample Portfolio Performance**
```python
# Construct a Max Sharpe portfolio
portfolio = construct_portfolio(
    tickers=["USDEUR", "GBPJPY", "AUDUSD"],
    start_date="2022-01-01",
    end_date="2023-12-31",
    method="max_sharpe",
    rebalance_freq="Quarterly"
)

# Results:
# - Annualized Return: 8.5%
# - Volatility: 12.3%
# - Sharpe Ratio: 0.69
# - Max Drawdown: -4.2%
```

---

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

---

## 🚀 Future Enhancements

- [ ] **Machine Learning** optimization methods
- [ ] **Crypto currency** pairs integration
- [ ] **Transaction cost** modeling
- [ ] **Multi-factor** risk models
- [ ] **Real-time** data streaming
- [ ] **API endpoints** for programmatic access

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional optimization algorithms
- New risk metrics
- Performance improvements
- Data source integrations
- UI/UX enhancements

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 About

**ForexQuant** was built to demonstrate modern portfolio theory in action with real currency markets. It combines academic rigor with practical usability, making sophisticated portfolio analytics accessible to students and professionals alike.

**Built with** ❤️ using Python, Streamlit, and modern portfolio theory.

---

## 📞 Contact

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
- **Email**: your.email@example.com

*Looking for opportunities in quantitative finance, portfolio analytics, or fintech!*

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

**🔗 [Live Demo](https://your-streamlit-app.streamlitapp.com) | 📊 [Sample Analysis](link-to-sample) | 📖 [Documentation](link-to-docs)**

</div> 