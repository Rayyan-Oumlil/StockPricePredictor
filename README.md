# StockPricePredictor

This project provides a simple yet extensible framework for predicting
future stock prices based on historical data. The goal is to give
developers, data‑scientists and hobbyists a starting point for
experimenting with different machine‑learning and deep‑learning
approaches to time series forecasting.

## 🚀 **New Features (2024)**

### 📊 **Web Interface with Streamlit**
- **Modern UI** : Beautiful, responsive web interface
- **Interactive Charts** : Zoom, hover, and pan capabilities
- **Real-time Analysis** : Instant results with live data
- **Multiple Models** : Compare Linear Regression vs Random Forest

### 🔧 **Advanced Technical Indicators**
- **RSI** : Relative Strength Index for overbought/oversold signals
- **MACD** : Moving Average Convergence Divergence
- **Bollinger Bands** : Volatility and trend analysis
- **Moving Averages** : SMA and EMA for trend identification
- **Volume Analysis** : Volume-based indicators

### 🤖 **Enhanced Machine Learning**
- **Feature Engineering** : 19+ technical indicators
- **Model Comparison** : Performance metrics (MSE, RMSE, MAE, R²)
- **Feature Importance** : Understand what drives predictions
- **Forecast Visualization** : See predictions on interactive charts



## 📁 **Project Structure**

```
StockPricePredictor/
├── main.py              # Original CLI version
├── app.py               # Full-featured Streamlit app
├── app_simple.py        # Simplified Streamlit app (recommended)
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## 🎯 **Quick Start**

### Option 1: Web Interface (Recommended) 🌐

**Launch the simplified version:**
```bash
streamlit run app_simple.py
```

**Launch the full-featured version:**
```bash
streamlit run app.py
```

**Then open your browser to:** `http://localhost:8501`

### Option 2: Command Line Interface 💻

```bash
python main.py --ticker AAPL --start 2023-01-01 --end 2024-01-01 --model random_forest --forecast_horizon 7
```

## 🛠 **Installation**

1. **Clone the repository:**
```bash
git clone <repository-url>
cd StockPricePredictor
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 📊 **Usage Examples**

### Web Interface
1. **Select Stock** : Choose any ticker (AAPL, MSFT, TSLA, etc.)
2. **Set Date Range** : Pick start and end dates
3. **Choose Model** : Linear Regression or Random Forest
4. **Set Forecast Days** : 1-30 days into the future
5. **Click "Analyze Stock"** : Get instant results!

### Command Line
```bash
# Basic usage
python main.py --ticker AAPL

# Advanced usage
python main.py --ticker TSLA --start 2023-01-01 --end 2024-01-01 --model random_forest --forecast_horizon 10 --no_plots

# Compare different stocks
python main.py --ticker MSFT --start 2022-01-01 --end 2024-01-01 --model linear --forecast_horizon 5
```

## 🔧 **Technical Indicators**

### 📈 **Price-based Indicators**
- **SMA (Simple Moving Average)** : 5, 10, 20-day averages
- **EMA (Exponential Moving Average)** : 12, 26-day averages
- **Bollinger Bands** : Upper, middle, lower bands for volatility

### 📊 **Momentum Indicators**
- **RSI (Relative Strength Index)** : Overbought/oversold levels
- **MACD** : Trend and momentum analysis
- **MACD Signal** : Moving average of MACD
- **MACD Histogram** : MACD - Signal difference

### 📈 **Volume Indicators**
- **Volume SMA** : 20-day average volume
- **Volume Ratio** : Current volume / average volume

## 🤖 **Machine Learning Models**

### Linear Regression
- **Pros** : Fast, interpretable, good baseline
- **Cons** : Limited to linear relationships
- **Best for** : Quick analysis, educational purposes

### Random Forest
- **Pros** : Handles non-linear relationships, feature importance
- **Cons** : Slower training, more complex
- **Best for** : Production use, detailed analysis

## 📊 **Performance Metrics**

- **MSE (Mean Squared Error)** : Average squared prediction error
- **RMSE (Root Mean Squared Error)** : Square root of MSE
- **MAE (Mean Absolute Error)** : Average absolute prediction error
- **R² (R-squared)** : Proportion of variance explained (0-1)

## 🎨 **Visualizations**

### Interactive Charts
- **Price Chart** : Historical prices with moving averages
- **Volume Chart** : Trading volume over time
- **Technical Indicators** : RSI, MACD, Bollinger Bands
- **Predictions** : Future price forecasts

### Data Tables
- **Prediction Table** : Daily forecasts with dates
- **Model Information** : Feature importance and metrics
- **Performance Summary** : Model evaluation results

## 🔮 **Future Enhancements**

- **LSTM Models** : Deep learning for time series
- **Multi-stock Comparison** : Compare multiple stocks
- **Portfolio Analysis** : Risk and return metrics
- **Real-time Alerts** : Price movement notifications
- **Backtesting** : Historical performance validation
- **API Integration** : Real-time data feeds

## 📋 **Dependencies**

```
pandas          # Data manipulation
numpy           # Numerical computing
matplotlib      # Basic plotting
seaborn         # Statistical visualization
scikit-learn    # Machine learning
yfinance        # Stock data download
streamlit       # Web interface
plotly          # Interactive charts
```

## ⚠️ **Disclaimer**

**Educational Purpose Only**: This tool is designed for educational and research purposes. 
Stock price predictions are inherently uncertain and should not be used as the sole basis 
for investment decisions. Always conduct thorough research and consider consulting with 
financial professionals before making investment decisions.

## 📄 **License**

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, 
please open an issue first to discuss what you would like to change.

## 📞 **Support**

If you encounter any issues or have questions:
1. Check the troubleshooting section below
2. Open an issue on GitHub
3. Review the code comments for guidance

## 🔧 **Troubleshooting**

### Common Issues

**"No data returned for ticker"**
- Verify the ticker symbol is correct
- Check your internet connection
- Try a different date range

**"Missing required columns"**
- This is usually a yfinance data issue
- Try the simplified version (`app_simple.py`)
- Use a different date range

**Streamlit not loading**
- Ensure all dependencies are installed
- Check if port 8501 is available
- Try `streamlit run app_simple.py` instead

### Performance Tips

- **Use Random Forest** for better accuracy
- **Longer date ranges** provide more training data
- **Recent data** is more relevant for predictions
- **Multiple indicators** improve model performance
