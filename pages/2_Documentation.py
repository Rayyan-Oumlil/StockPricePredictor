import streamlit as st

st.set_page_config(page_title="Documentation", page_icon="📖", layout="wide")

st.header("📖 Documentation & Help")
st.info("Complete guide to using the Stock Price Predictor application")

# Table of Contents
st.subheader("📋 Table of Contents")
toc = """
1. [Application Overview](#application-overview)
2. [Pages Guide](#pages-guide)
3. [Machine Learning Models](#machine-learning-models)
4. [Technical Indicators](#technical-indicators)
5. [Performance Metrics](#performance-metrics)
6. [Alerts System](#alerts-system)
7. [Data Sources](#data-sources)
8. [FAQ](#faq)
9. [Troubleshooting](#troubleshooting)
10. [Useful Links](#useful-links)
"""
st.markdown(toc)

# Application Overview
st.subheader("🎯 Application Overview")
st.markdown("""
The **Stock Price Predictor** is a comprehensive web application that combines machine learning with technical analysis to provide stock price predictions and market insights.

**Key Features:**
- **Multi-Stock Analysis**: Analyze up to 16 stocks simultaneously
- **Machine Learning Models**: Linear Regression and Random Forest predictions
- **Technical Indicators**: Advanced technical analysis tools
- **Real-Time Data**: Live market data from Yahoo Finance
- **Interactive Charts**: Dynamic visualizations with Plotly
- **Alert System**: Price threshold notifications
- **Market Overview**: Real-time market insights
- **Comparison Tools**: Side-by-side stock analysis
- **Backtesting**: Validate predictions against historical data
- **Export Features**: Download predictions and charts
""")

# Pages Guide
st.subheader("📱 Pages Guide")

st.markdown("### 🏠 App (Main Analysis)")
st.markdown("""
**Primary analysis page with the following features:**

**Stock Selection:**
- Choose from 16 popular stocks or enter custom tickers
- Multi-select interface for analyzing multiple stocks simultaneously

**Analysis Parameters:**
- **Start Date**: Beginning of historical data period
- **End Date**: End of historical data period (can be in the past for backtesting)
- **Forecast Days**: Number of days to predict (1-30 days)
- **Model Selection**: Linear Regression or Random Forest

**Results Display:**
- **Summary Table**: Current price, predicted price, change percentage, and model performance metrics
- **Price Charts**: Interactive charts showing historical data and predictions
- **Technical Charts**: Advanced indicators and analysis
- **Backtesting Results**: If end date is in the past, shows prediction accuracy metrics

**Export Options:**
- Download predictions as Excel file
- Export charts as PNG images
- Direct download links (no page reload)
""")

st.markdown("### 📈 Market Overview")
st.markdown("""
**Real-time market insights and performance metrics:**

**Major Indices:**
- S&P 500, NASDAQ, DOW JONES, Russell 2000
- Real-time prices and daily changes

**Top Movers:**
- Top 5 gainers and losers from popular stocks
- Price changes and percentages

**Sector Performance:**
- Technology, Healthcare, Financial, Consumer sectors
- Average performance and stock counts
- Interactive bar charts

**Market Sentiment:**
- Positive vs negative stocks ratio
- Average market change
- Overall market direction indicators
""")

st.markdown("### 📊 Comparison Tool")
st.markdown("""
**Side-by-side stock comparison and analysis:**

**Stock Selection:**
- Quick selection from popular stocks
- Custom ticker input
- Multiple stock comparison

**Time Periods:**
- 1 month to 5 years analysis periods
- Flexible time range selection

**Comparison Features:**
- **Price Comparison**: Overlaid price charts
- **Performance Metrics**: Returns, volatility, Sharpe ratio, max drawdown
- **Risk-Return Analysis**: Scatter plots and correlation matrices
- **Performance Ranking**: Best and worst performers

**Advanced Analytics:**
- Cumulative returns comparison
- Rolling volatility analysis
- Correlation heatmaps
- Summary insights and recommendations
""")

st.markdown("### 🔔 Alerts")
st.markdown("""
**Price threshold alert system:**

**Alert Configuration:**
- Set alerts for specific stocks
- Choose "Above threshold" or "Below threshold"
- Configure price levels
- Persistent storage (saves between sessions)

**Alert Types:**
- **Above Threshold**: Notify when price goes above set level
- **Below Threshold**: Notify when price goes below set level

**Management:**
- View all configured alerts
- Delete individual alerts
- Clear all alerts
- In-app notifications during analysis
""")

st.markdown("### 📖 Documentation")
st.markdown("""
**This page - comprehensive help and guides:**
- Complete feature documentation
- Model explanations
- Technical indicator guides
- FAQ and troubleshooting
- Useful resources and links
""")

# Machine Learning Models
st.subheader("🤖 Machine Learning Models")

st.markdown("### Linear Regression")
st.markdown("""
**Simple and interpretable model for trend-based predictions:**

**How it works:**
- Uses historical price data and technical indicators
- Finds linear relationships between features and future prices
- Extrapolates trends into the future

**Advantages:**
- Fast computation
- Easy to understand
- Good for trending markets
- Low computational requirements

**Best for:**
- Short-term predictions (1-7 days)
- Trending stocks
- Stable market conditions
- Quick analysis
""")

st.markdown("### Random Forest")
st.markdown("""
**Advanced ensemble model for complex pattern recognition:**

**How it works:**
- Combines multiple decision trees
- Uses bagging (bootstrap aggregating)
- Handles non-linear relationships
- Provides feature importance

**Advantages:**
- Captures complex patterns
- Handles outliers well
- Provides confidence intervals
- Feature importance analysis

**Characteristics:**
- Step-wise predictions (non-continuous lines)
- Better for longer-term forecasts
- More robust to market noise
- Higher computational cost

**Best for:**
- Medium-term predictions (7-30 days)
- Volatile stocks
- Complex market conditions
- Detailed analysis
""")

# Technical Indicators
st.subheader("📊 Technical Indicators")

st.markdown("### Moving Averages")
st.markdown("""
**Trend-following indicators:**

**Simple Moving Averages (SMA):**
- **SMA_5**: 5-day moving average
- **SMA_10**: 10-day moving average  
- **SMA_20**: 20-day moving average

**Usage:**
- Identify trend direction
- Support and resistance levels
- Crossover signals
- Trend strength measurement
""")

st.markdown("### Momentum Indicators")
st.markdown("""
**RSI (Relative Strength Index):**
- Measures speed and magnitude of price changes
- Range: 0-100
- **Overbought**: >70 (potential sell signal)
- **Oversold**: <30 (potential buy signal)
- **Neutral**: 30-70 range

**MACD (Moving Average Convergence Divergence):**
- **MACD Line**: 12-day EMA - 26-day EMA
- **Signal Line**: 9-day EMA of MACD line
- **Histogram**: MACD line - Signal line
- **Bullish**: MACD above signal line
- **Bearish**: MACD below signal line
""")

st.markdown("### Volatility Indicators")
st.markdown("""
**Bollinger Bands:**
- **Middle Band**: 20-day SMA
- **Upper Band**: Middle + (2 × Standard Deviation)
- **Lower Band**: Middle - (2 × Standard Deviation)
- **Squeeze**: Bands narrow (low volatility)
- **Expansion**: Bands widen (high volatility)
""")

st.markdown("### Volume Indicators")
st.markdown("""
**Volume Analysis:**
- **Volume_SMA**: 20-day average volume
- **Volume_Ratio**: Current volume / Average volume
- **High Volume**: >1.5 ratio (strong moves)
- **Low Volume**: <0.5 ratio (weak moves)
""")

# Performance Metrics
st.subheader("📈 Performance Metrics")

st.markdown("### Model Evaluation")
st.markdown("""
**Backtesting Metrics (when end date is in the past):**

**MSE (Mean Squared Error):**
- Average squared difference between predictions and actual values
- Lower values indicate better accuracy
- Penalizes large errors more heavily

**RMSE (Root Mean Squared Error):**
- Square root of MSE
- Same units as the target variable (price)
- More interpretable than MSE

**MAE (Mean Absolute Error):**
- Average absolute difference between predictions and actual values
- Less sensitive to outliers than MSE
- Easy to interpret

**R² (R-squared):**
- Proportion of variance explained by the model
- Range: 0-1 (1 = perfect prediction)
- Higher values indicate better fit
""")

# Alerts System
st.subheader("🔔 Alerts System")

st.markdown("### How Alerts Work")
st.markdown("""
**Alert Configuration:**
1. Go to the Alerts page
2. Enter ticker symbol
3. Select alert type (Above/Below threshold)
4. Set price threshold
5. Alerts are automatically saved

**Alert Triggers:**
- During stock analysis, the app checks all configured alerts
- If any prediction meets the threshold criteria, an in-app warning appears
- Alerts are checked for each stock in your analysis

**Alert Types:**
- **Above Threshold**: Triggers when predicted price ≥ threshold
- **Below Threshold**: Triggers when predicted price ≤ threshold

**Management:**
- View all alerts on the Alerts page
- Delete individual alerts with the 🗑️ button
- Clear all alerts with "Clear All Alerts"
- Alerts persist between sessions
""")

# Data Sources
st.subheader("📡 Data Sources")

st.markdown("""
**Primary Data Source: Yahoo Finance (yfinance)**

**Data Types:**
- **Historical Prices**: OHLCV data (Open, High, Low, Close, Volume)
- **Company Information**: Market cap, sector, industry
- **Real-time Data**: Current prices and market data
- **Technical Indicators**: Calculated from price data

**Data Quality:**
- Free and reliable
- Regular updates
- Wide coverage of stocks
- Historical data available

**Limitations:**
- Rate limiting may apply
- Data may have slight delays
- Some stocks may have limited data
""")

# FAQ
st.subheader("❓ Frequently Asked Questions")

st.markdown("### General Questions")

st.markdown("**Q: How accurate are the predictions?**")
st.markdown("A: Predictions are based on historical patterns and technical indicators. While models can identify trends, stock markets are inherently unpredictable. Always use predictions as one of many tools in your investment decision-making process.")

st.markdown("**Q: Can I analyze any stock?**")
st.markdown("A: Yes, you can analyze any stock available on Yahoo Finance. The app includes 16 popular stocks for quick selection, or you can enter any valid ticker symbol.")

st.markdown("**Q: What's the difference between Linear Regression and Random Forest?**")
st.markdown("A: Linear Regression is faster and better for short-term trends, while Random Forest captures complex patterns and is better for longer-term predictions. Random Forest may show step-wise predictions due to its nature.")

st.markdown("**Q: How does backtesting work?**")
st.markdown("A: When you set an end date in the past, the model trains on data up to that date, then makes predictions for the following days. These predictions are compared to actual historical prices to calculate accuracy metrics.")

st.markdown("**Q: Why do my alerts disappear when I reload the page?**")
st.markdown("A: Alerts are automatically saved to a file and should persist between sessions. If they disappear, check that the alerts_config.json file is being created in your app directory.")

# Troubleshooting
st.subheader("🔧 Troubleshooting")

st.markdown("### Common Issues")

st.markdown("**Problem: No data loading for stocks**")
st.markdown("""
**Solutions:**
- Check your internet connection
- Verify the ticker symbol is correct
- Try a different stock
- Wait a few minutes and try again (rate limiting)
""")

st.markdown("**Problem: Charts not displaying**")
st.markdown("""
**Solutions:**
- Refresh the page
- Check browser compatibility
- Ensure JavaScript is enabled
- Try a different browser
""")

st.markdown("**Problem: Predictions seem inaccurate**")
st.markdown("""
**Solutions:**
- Try different date ranges
- Use different models
- Consider market conditions
- Remember predictions are estimates, not guarantees
""")

st.markdown("**Problem: App is slow**")
st.markdown("""
**Solutions:**
- Reduce the number of stocks analyzed
- Use shorter date ranges
- Close other browser tabs
- Check your internet speed
""")

# Useful Links
st.subheader("🔗 Useful Links")

st.markdown("""
**Educational Resources:**
- [Investopedia - Technical Analysis](https://www.investopedia.com/technical-analysis-4689657)
- [Yahoo Finance](https://finance.yahoo.com/)
- [StockCharts.com](https://stockcharts.com/)

**Trading Platforms:**
- [Robinhood](https://robinhood.com/)
- [TD Ameritrade](https://www.tdameritrade.com/)
- [E*TRADE](https://us.etrade.com/)

**Market News:**
- [MarketWatch](https://www.marketwatch.com/)
- [CNBC](https://www.cnbc.com/)
- [Bloomberg](https://www.bloomberg.com/)

**Technical Analysis Tools:**
- [TradingView](https://www.tradingview.com/)
- [Finviz](https://finviz.com/)
- [StockCharts](https://stockcharts.com/)
""")

st.markdown("---")
st.caption("""
**Disclaimer:** This application is for educational and informational purposes only. 
Stock predictions are based on historical data and technical analysis, which may not be indicative of future performance. 
Always conduct your own research and consider consulting with a financial advisor before making investment decisions. 
Past performance does not guarantee future results.
""") 