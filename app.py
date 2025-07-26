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
import json
import os
import pickle
from pathlib import Path
import io
import base64
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Page configuration
st.set_page_config(page_title="App", page_icon="📈", layout="wide")

# Responsive meta tag for mobile
st.markdown("""
<meta name='viewport' content='width=device-width, initial-scale=1.0'>
""", unsafe_allow_html=True)

# Custom CSS for mobile responsiveness
st.markdown("""
<style>
    html, body, [class*="css"]  {
        font-size: 16px !important;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    .main-header {
        font-size: 2.2rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card, .stock-card {
        padding: 0.7rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stButton>button, .stDownloadButton>button {
        font-size: 1.1rem;
        padding: 0.7rem 1.2rem;
        border-radius: 0.5rem;
    }
    .stSlider, .stCheckbox, .stSelectbox, .stTextInput {
        font-size: 1.1rem !important;
    }
    .stDataFrame, .stTable {
        overflow-x: auto;
        max-width: 100vw;
    }
    @media (max-width: 900px) {
        .main-header { font-size: 1.5rem; }
        .stDataFrame, .stTable { font-size: 0.95rem; }
        .stButton>button, .stDownloadButton>button { font-size: 1rem; padding: 0.6rem 1rem; }
    }
    @media (max-width: 600px) {
        .main-header { font-size: 1.1rem; }
        .stDataFrame, .stTable { font-size: 0.85rem; }
        .stButton>button, .stDownloadButton>button { font-size: 0.95rem; padding: 0.5rem 0.7rem; }
        .stSlider, .stCheckbox, .stSelectbox, .stTextInput { font-size: 1rem !important; }
        .metric-card, .stock-card { padding: 0.4rem; }
    }
</style>
""", unsafe_allow_html=True)

# Remove all user authentication and account-related code

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

# Utility to convert DataFrame to Excel for download
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    processed_data = output.getvalue()
    return processed_data

# Utility to download Plotly figure as PNG
def fig_to_png_bytes(fig):
    img_bytes = fig.to_image(format="png")
    return img_bytes

# Remove any save_analysis_history or history saving logic

# --- Email alert configuration (user must fill these in) ---
# Remove email alert configuration and functions

# Remove GMAIL_ADDRESS, GMAIL_APP_PASSWORD, send_email_alert, and all related code

def load_alerts_from_file():
    """Load alerts from JSON file"""
    if os.path.exists("alerts_config.json"):
        try:
            with open("alerts_config.json", "r") as f:
                data = json.load(f)
                return data.get("alerts", []), data.get("email", "")
        except:
            return [], ""
    return [], ""

def main():
    # Load saved alerts and email
    saved_alerts, saved_email = load_alerts_from_file()
    if 'alertes' not in st.session_state:
        st.session_state.alertes = saved_alerts
    if 'alert_email' not in st.session_state:
        st.session_state.alert_email = saved_email
    
    # Header
    st.markdown('<h1 class="main-header">📈 Multi-Stock Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Compare and predict multiple stocks simultaneously")
    
    # Si une analyse à revoir est demandée, l'afficher
    if 'history_to_show' in st.session_state:
        import os, json
        file = st.session_state['history_to_show']
        path = os.path.join('history', file)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            st.info(f"Affichage d'une analyse sauvegardée du {data.get('date', file)}")
            st.write(f"**Tickers :** {', '.join(data.get('tickers', []))}")
            st.write(f"**Modèle :** {data.get('model', '-')}")
            st.write(f"**Jours de prévision :** {data.get('forecast_days', '-')}")
            # Afficher un tableau de prédictions pour le premier ticker
            preds = data.get('all_predictions', {})
            if preds:
                first_ticker = list(preds.keys())[0]
                val = preds[first_ticker]
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                    df = pd.DataFrame(val)
                elif isinstance(val, dict):
                    df = pd.DataFrame.from_dict(val)
                elif isinstance(val, list):
                    df = pd.DataFrame({'Predicted': val})
                else:
                    st.write("Format de prédiction inconnu :", type(val))
                    st.write(val)
                    df = pd.DataFrame()
                st.write(f"**Prédictions pour {first_ticker} :**")
                st.dataframe(df, use_container_width=True)
            else:
                st.write("Aucune prédiction enregistrée.")
            if st.button("Retour à l'analyse en direct"):
                del st.session_state['history_to_show']
                st.experimental_rerun()
            return
    
    # Sidebar
    st.sidebar.header("Settings")
    # Stock selection - multiple stocks
    st.sidebar.subheader("Select Stocks")
    
    # Predefined popular stocks - curated list of 16
    popular_stocks = [
        "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA", "NFLX",
        "AMD", "INTC", "DIS", "SPOT", "BABA", "ORCL", "ADBE", "CRM"
    ]
    
    # Stock selection methods
    selection_method = st.sidebar.radio(
        "Choose stocks:",
        ["Popular Stocks", "Custom Tickers", "Both"]
    )
    
    selected_tickers = []
    
    # Load user's favorite stocks if logged in
    # default_favorites = ["AAPL", "MSFT"] # Removed user-specific favorites
    
    if selection_method in ["Popular Stocks", "Both"]:
        st.sidebar.write("**Popular Stocks:**")
        
        # Create a scrollable container for stock selection
        with st.sidebar.container():
            # Use columns to create a more compact layout
            col1, col2 = st.columns(2)
            
            for i, stock in enumerate(popular_stocks):
                # Alternate between columns for better space usage
                if i % 2 == 0:
                    with col1:
                        if st.checkbox(stock, value=(stock in ["AAPL", "MSFT"]), key=f"stock_{stock}"):
                            selected_tickers.append(stock)
                else:
                    with col2:
                        if st.checkbox(stock, value=(stock in ["AAPL", "MSFT"]), key=f"stock_{stock}"):
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
        index=0 # No user preference saved
    )
    
    # Forecast horizon
    default_forecast = 5
    # default_forecast = st.session_state.get('forecast_days', 5) # Removed user preference
    
    forecast_days = st.sidebar.slider("Forecast Days", 1, 30, default_forecast)
    
    # Analysis button
    if st.sidebar.button("🚀 Analyze Stocks", type="primary"):
        if not selected_tickers:
            st.error("Please select at least one stock ticker.")
            return
        
        # Store current settings in session state for saving
        st.session_state.selected_tickers = selected_tickers
        st.session_state.model_name = model_name
        st.session_state.forecast_days = forecast_days
        
        with st.spinner(f"Analyzing {len(selected_tickers)} stocks..."):
            all_data = {}
            all_forecasts = {}
            all_metrics = {}
            all_predictions = {}
            all_backtest_metrics = {}
            all_backtest_forecasts = {}
            all_backtest_actuals = {}
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Detect if backtesting mode (end date < today)
            today = datetime.now().date()
            backtest_mode = end_date < today
            
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
                        
                        # --- Alert check ---
                        if 'alertes' in st.session_state:
                            for alert in st.session_state['alertes']:
                                if alert['ticker'].upper() == ticker.upper():
                                    # Check if any predicted value triggers the alert
                                    if alert['type'] == "above":
                                        triggered = (forecast_df['Predicted_Close'] >= alert['seuil']).any()
                                        symbol = "≥"
                                    else:  # below
                                        triggered = (forecast_df['Predicted_Close'] <= alert['seuil']).any()
                                        symbol = "≤"
                                    
                                    if triggered:
                                        st.warning(f"ALERT: Prediction for {ticker} {symbol} {alert['seuil']}!", icon="⚠️")
                        
                        # --- Backtesting logic ---
                        if backtest_mode:
                            # We'll backtest on the last N days (N = forecast_days), if enough data
                            if len(df) > forecast_days:
                                # Use data up to -(forecast_days) as training, then predict next N days
                                train_df = df.iloc[:-forecast_days]
                                test_df = df.iloc[-forecast_days:]
                                if len(train_df) > 0:
                                    backtest_model, backtest_feature_cols, _ = train_model(train_df, model_name)
                                    backtest_forecast = forecast(backtest_model, train_df, backtest_feature_cols, forecast_days)
                                    # Align dates with actuals
                                    backtest_forecast = backtest_forecast.set_index('Date')
                                    test_actual = test_df['Close']
                                    # Only keep overlapping dates
                                    common_dates = backtest_forecast.index.intersection(test_actual.index)
                                    y_true = test_actual.loc[common_dates]
                                    y_pred = backtest_forecast.loc[common_dates]['Predicted_Close']
                                    # Compute metrics
                                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                                    mse = mean_squared_error(y_true, y_pred)
                                    rmse = np.sqrt(mse)
                                    mae = mean_absolute_error(y_true, y_pred)
                                    r2 = r2_score(y_true, y_pred)
                                    all_backtest_metrics[ticker] = {
                                        'MSE': mse,
                                        'RMSE': rmse,
                                        'MAE': mae,
                                        'R²': r2
                                    }
                                    all_backtest_forecasts[ticker] = backtest_forecast.loc[common_dates]
                                    all_backtest_actuals[ticker] = y_true
                progress_bar.progress((i + 1) / len(selected_tickers))
            status_text.text("Analysis complete!")
            if all_data:
                # Store results in session state for saving
                st.session_state.current_results = {
                    'all_data': all_data,
                    'all_forecasts': all_forecasts,
                    'all_metrics': all_metrics,
                    'all_predictions': all_predictions,
                    'all_backtest_metrics': all_backtest_metrics,
                    'all_backtest_forecasts': all_backtest_forecasts,
                    'all_backtest_actuals': all_backtest_actuals,
                    'backtest_mode': backtest_mode,
                    'tickers': selected_tickers,
                    'model': model_name,
                    'forecast_days': forecast_days,
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                # Sauvegarde dans l'historique
                # Remove any save_analysis_history or history saving logic
                
                if backtest_mode:
                    st.info(f"🔎 Backtesting Mode: End date is in the past. Comparing model predictions with actual historical prices for the last {forecast_days} days.")
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
                    summary_row = {
                        'Ticker': ticker,
                        'Current Price': f"${latest_price:.2f}",
                        'Predicted Price': f"${predicted_price:.2f}",
                        'Change %': f"{price_change:+.2f}%",
                        'R² Score': f"{metrics['R²']:.3f}",
                        'RMSE': f"{metrics['RMSE']:.2f}"
                    }
                    if backtest_mode and ticker in all_backtest_metrics:
                        summary_row['Backtest RMSE'] = f"{all_backtest_metrics[ticker]['RMSE']:.2f}"
                        summary_row['Backtest R²'] = f"{all_backtest_metrics[ticker]['R²']:.3f}"
                    summary_data.append(summary_row)
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                # Download predictions as Excel (only this button remains)
                all_predictions_combined = []
                for ticker in all_data.keys():
                    predictions = all_predictions[ticker].copy()
                    predictions['Ticker'] = ticker
                    predictions['Date'] = predictions['Date'].dt.strftime('%Y-%m-%d')
                    predictions['Predicted_Close'] = predictions['Predicted_Close'].round(2)
                    all_predictions_combined.append(predictions)
                
                if all_predictions_combined:
                    # Large format: one column per ticker, one row per date
                    # Build a DataFrame with Date as index, columns as tickers
                    wide_df = None
                    for ticker in all_data.keys():
                        pred = all_predictions[ticker].copy()
                        pred = pred[['Date', 'Predicted_Close']]
                        pred['Date'] = pred['Date'].dt.strftime('%Y-%m-%d')
                        pred = pred.set_index('Date')
                        pred = pred.rename(columns={'Predicted_Close': ticker})
                        if wide_df is None:
                            wide_df = pred
                        else:
                            wide_df = wide_df.join(pred, how='outer')
                    wide_df = wide_df.reset_index()
                    # Export to Excel
                    detailed_excel_bytes = to_excel(wide_df)
                    import base64
                    b64 = base64.b64encode(detailed_excel_bytes).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="predictions.xlsx" style="display:inline-block;padding:0.5em 1.2em;font-size:1.1rem;font-weight:600;color:#fff;background:#1f77b4;border-radius:0.4em;text-decoration:none;margin:0.5em 0;">⬇️ Download Predictions as Excel</a>'
                    st.markdown(href, unsafe_allow_html=True)
                # Download main chart as PNG (no reload, no session_state)
                try:
                    png_bytes = comparison_chart.to_image(format="png")
                    import base64
                    b64_png = base64.b64encode(png_bytes).decode()
                    href_png = f'<a href="data:image/png;base64,{b64_png}" download="comparison_chart.png" style="display:inline-block;padding:0.5em 1.2em;font-size:1.1rem;font-weight:600;color:#fff;background:#ff7f0e;border-radius:0.4em;text-decoration:none;margin:0.5em 0;">⬇️ Download Chart as PNG</a>'
                    st.markdown(href_png, unsafe_allow_html=True)
                except Exception as e:
                    st.warning("⚠️ Chart download not available. Install kaleido: pip install kaleido")
                    st.info("💡 You can still download the Excel data and take screenshots of the charts.")
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
                        if backtest_mode and ticker in all_backtest_metrics:
                            st.markdown("**Backtest Metrics (last {} days):**".format(forecast_days))
                            bmetrics = all_backtest_metrics[ticker]
                            bcol1, bcol2, bcol3, bcol4 = st.columns(4)
                            with bcol1:
                                st.metric("Backtest MSE", f"{bmetrics['MSE']:.4f}")
                            with bcol2:
                                st.metric("Backtest RMSE", f"{bmetrics['RMSE']:.4f}")
                            with bcol3:
                                st.metric("Backtest MAE", f"{bmetrics['MAE']:.4f}")
                            with bcol4:
                                st.metric("Backtest R²", f"{bmetrics['R²']:.4f}")
                            # Show overlay chart
                            st.markdown("**Backtest: Predicted vs. Actual**")
                            import plotly.graph_objs as go
                            fig = go.Figure()
                            y_true = all_backtest_actuals[ticker]
                            y_pred = all_backtest_forecasts[ticker]['Predicted_Close']
                            fig.add_trace(go.Scatter(x=y_true.index, y=y_true, mode='lines+markers', name='Actual', line=dict(color='green')))
                            fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred, mode='lines+markers', name='Predicted', line=dict(color='orange', dash='dot')))
                            fig.update_layout(title=f"Backtest: {ticker} - Predicted vs. Actual", xaxis_title="Date", yaxis_title="Price (USD)", height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        # Predictions table
                        predictions_display = all_predictions[ticker].copy()
                        predictions_display['Date'] = predictions_display['Date'].dt.strftime('%Y-%m-%d')
                        predictions_display['Predicted_Close'] = predictions_display['Predicted_Close'].round(2)
                        predictions_display.columns = ['Date', 'Predicted Price ($)']
                        st.dataframe(predictions_display, use_container_width=True)
                        # Download predictions as Excel
                        pred_excel_bytes = to_excel(predictions_display)
                        st.download_button(
                            label=f"⬇️ Download {ticker} Predictions as Excel",
                            data=pred_excel_bytes,
                            file_name=f"{ticker}_predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            else:
                st.error("No data could be retrieved for the selected stocks.")

if __name__ == "__main__":
    main() 