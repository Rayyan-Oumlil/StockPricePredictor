#!/usr/bin/env python3
"""
Multi-Stock Streamlit Web App for Stock Price Predictor

A version that allows analyzing multiple stocks simultaneously.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Page configuration
st.set_page_config(
    page_title="Multi-Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stock-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download historical data for a given ticker from Yahoo Finance."""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        
        if data.empty:
            return None
        
        # Handle multi-index columns properly
        if isinstance(data.columns, pd.MultiIndex):
            new_columns = []
            for col in data.columns:
                if isinstance(col, tuple):
                    new_columns.append(col[0])
                else:
                    new_columns.append(col)
            data.columns = new_columns
        else:
            if len(data.columns) == 5:
                data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        return data
    except Exception as e:
        return None

def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add simple technical indicators to the dataset."""
    df = data.copy()
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return None
    
    # Simple moving averages
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # Target variable: next day's closing price
    df["Target"] = df["Close"].shift(-1)
    
    # Drop rows with NaN values
    df = df.dropna()
    return df

def train_model(df: pd.DataFrame, model_name: str):
    """Train a regression model on the feature set."""
    feature_cols = ["Open", "High", "Low", "Close", "Volume", "SMA_5", "SMA_10", "SMA_20", "RSI"]
    
    X = df[feature_cols]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, feature_cols, {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

def forecast(model, df: pd.DataFrame, feature_cols, horizon: int) -> pd.DataFrame:
    """Generate forecasts for the specified number of future days."""
    last_row = df.iloc[-1].copy()
    predictions = []
    current_date = df.index[-1]
    
    last_closes = df["Close"].iloc[-20:].values
    
    for i in range(horizon):
        X_last = last_row[feature_cols].values.reshape(1, -1)
        next_close = model.predict(X_last)[0]
        next_date = current_date + timedelta(days=1)
        predictions.append((next_date, next_close))
        
        # Update features for next iteration
        last_row["Open"] = next_close
        last_row["High"] = next_close
        last_row["Low"] = next_close
        last_row["Close"] = next_close
        
        last_closes = np.append(last_closes[1:], next_close)
        
        # Recompute technical indicators
        last_row["SMA_5"] = np.mean(last_closes[-5:]) if len(last_closes) >= 5 else np.mean(last_closes)
        last_row["SMA_10"] = np.mean(last_closes[-10:]) if len(last_closes) >= 10 else np.mean(last_closes)
        last_row["SMA_20"] = np.mean(last_closes[-20:]) if len(last_closes) >= 20 else np.mean(last_closes)
        last_row["RSI"] = 50
        
        current_date = next_date
    
    forecast_df = pd.DataFrame(predictions, columns=["Date", "Predicted_Close"])
    return forecast_df

def create_comparison_chart(all_data: dict, all_forecasts: dict):
    """Create a comparison chart for multiple stocks."""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for i, (ticker, data) in enumerate(all_data.items()):
        color = colors[i % len(colors)]
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name=f'{ticker} (Historical)',
            line=dict(color=color, width=2),
            opacity=0.7
        ))
        
        # Forecast data
        if ticker in all_forecasts:
            forecast = all_forecasts[ticker]
            fig.add_trace(go.Scatter(
                x=forecast['Date'],
                y=forecast['Predicted_Close'],
                mode='lines+markers',
                name=f'{ticker} (Prediction)',
                line=dict(color=color, width=2, dash='dot'),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title='Multi-Stock Price Comparison & Predictions',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Multi-Stock Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Compare and predict multiple stocks simultaneously")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Stock selection - multiple stocks
    st.sidebar.subheader("Select Stocks")
    
    # Predefined popular stocks
    popular_stocks = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA", "NFLX"]
    
    # Stock selection methods
    selection_method = st.sidebar.radio(
        "Choose stocks:",
        ["Popular Stocks", "Custom Tickers", "Both"]
    )
    
    selected_tickers = []
    
    if selection_method in ["Popular Stocks", "Both"]:
        st.sidebar.write("**Popular Stocks:**")
        for stock in popular_stocks:
            if st.sidebar.checkbox(stock, value=(stock in ["AAPL", "MSFT"])):
                selected_tickers.append(stock)
    
    if selection_method in ["Custom Tickers", "Both"]:
        st.sidebar.write("**Custom Tickers:**")
        custom_tickers = st.sidebar.text_area(
            "Enter custom tickers (one per line):",
            value="",
            height=100
        )
        if custom_tickers:
            custom_list = [t.strip().upper() for t in custom_tickers.split('\n') if t.strip()]
            selected_tickers.extend(custom_list)
    
    # Remove duplicates and limit to 8 stocks for performance
    selected_tickers = list(set(selected_tickers))[:8]
    
    if not selected_tickers:
        st.warning("Please select at least one stock ticker.")
        return
    
    # Date selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "ML Model",
        ["Linear Regression", "Random Forest"],
        index=0
    )
    
    # Forecast horizon
    forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 5)
    
    # Analysis button
    if st.sidebar.button("🚀 Analyze Stocks", type="primary"):
        if not selected_tickers:
            st.error("Please select at least one stock ticker.")
            return
        
        with st.spinner(f"Analyzing {len(selected_tickers)} stocks..."):
            all_data = {}
            all_forecasts = {}
            all_metrics = {}
            all_predictions = {}
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, ticker in enumerate(selected_tickers):
                status_text.text(f"Processing {ticker}...")
                
                # Download data
                data = download_data(ticker, start_date, end_date)
                
                if data is not None:
                    # Add features
                    df = add_features(data)
                    
                    if df is not None:
                        # Train model
                        model, feature_cols, metrics = train_model(df, model_name)
                        
                        # Generate forecast
                        forecast_df = forecast(model, df, feature_cols, forecast_days)
                        
                        # Store results
                        all_data[ticker] = data
                        all_forecasts[ticker] = forecast_df
                        all_metrics[ticker] = metrics
                        all_predictions[ticker] = forecast_df.copy()
                
                progress_bar.progress((i + 1) / len(selected_tickers))
            
            status_text.text("Analysis complete!")
            
            if all_data:
                st.success(f"✅ Analysis complete for {len(all_data)} stocks!")
                
                # Comparison chart
                st.subheader("📊 Multi-Stock Comparison")
                comparison_chart = create_comparison_chart(all_data, all_forecasts)
                st.plotly_chart(comparison_chart, use_container_width=True)
                
                # Results table
                st.subheader("📋 Results Summary")
                
                # Create summary table
                summary_data = []
                for ticker in all_data.keys():
                    metrics = all_metrics[ticker]
                    latest_price = all_data[ticker]['Close'].iloc[-1]
                    predicted_price = all_forecasts[ticker]['Predicted_Close'].iloc[-1]
                    price_change = ((predicted_price - latest_price) / latest_price) * 100
                    
                    summary_data.append({
                        'Ticker': ticker,
                        'Current Price': f"${latest_price:.2f}",
                        'Predicted Price': f"${predicted_price:.2f}",
                        'Change %': f"{price_change:+.2f}%",
                        'R² Score': f"{metrics['R²']:.3f}",
                        'RMSE': f"{metrics['RMSE']:.2f}"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Individual stock details
                st.subheader("🎯 Detailed Predictions")
                
                for ticker in all_data.keys():
                    with st.expander(f"📈 {ticker} Details"):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        metrics = all_metrics[ticker]
                        with col1:
                            st.metric("MSE", f"{metrics['MSE']:.4f}")
                        with col2:
                            st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                        with col3:
                            st.metric("MAE", f"{metrics['MAE']:.4f}")
                        with col4:
                            st.metric("R²", f"{metrics['R²']:.4f}")
                        
                        # Predictions table
                        predictions_display = all_predictions[ticker].copy()
                        predictions_display['Date'] = predictions_display['Date'].dt.strftime('%Y-%m-%d')
                        predictions_display['Predicted_Close'] = predictions_display['Predicted_Close'].round(2)
                        predictions_display.columns = ['Date', 'Predicted Price ($)']
                        st.dataframe(predictions_display, use_container_width=True)
            else:
                st.error("No data could be retrieved for the selected stocks.")
    
    # Instructions
    else:
        st.info("👈 Use the sidebar to select multiple stocks and click 'Analyze Stocks' to get started!")
        
        # Sample data
        st.subheader("📈 Multi-Stock Analysis Features")
        st.write("This app allows you to:")
        st.write("- 📊 Compare multiple stocks simultaneously")
        st.write("- 🤖 Train ML models on each stock")
        st.write("- 📈 View comparison charts with predictions")
        st.write("- 🎯 Get detailed predictions for each stock")
        st.write("- 📋 Compare performance metrics across stocks")
        
        # Popular stocks info
        st.subheader("📊 Popular Stocks Available")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write("**Tech Giants:**")
            st.write("• AAPL (Apple)")
            st.write("• MSFT (Microsoft)")
            st.write("• GOOGL (Google)")
        
        with col2:
            st.write("**Growth Stocks:**")
            st.write("• TSLA (Tesla)")
            st.write("• AMZN (Amazon)")
            st.write("• META (Meta)")
        
        with col3:
            st.write("**Semiconductors:**")
            st.write("• NVDA (NVIDIA)")
            st.write("• AMD (Advanced Micro)")
            st.write("• INTC (Intel)")
        
        with col4:
            st.write("**Entertainment:**")
            st.write("• NFLX (Netflix)")
            st.write("• DIS (Disney)")
            st.write("• SPOT (Spotify)")

if __name__ == "__main__":
    main() 