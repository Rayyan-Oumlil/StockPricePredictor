#!/usr/bin/env python3
"""
Main entry point for the StockPricePredictor project.

This script downloads historical stock data, performs basic visualisations
and trains a simple regression model to forecast future closing prices.

Usage examples:

    python main.py --ticker AAPL --start 2015-01-01 --end 2024-01-01 \
        --model random_forest --forecast_horizon 7

By default the script uses a linear regression model and forecasts five
days into the future.  See the `--model` flag for supported models.

Note: The forecast provided by this script is for educational purposes
only and should not be used for actual trading decisions.
"""

import argparse
from datetime import datetime, timedelta
import sys

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Predict future stock prices using historical data")
    parser.add_argument("--ticker", type=str, default="AAPL",
                        help="Ticker symbol (e.g. AAPL, GOOGL)")
    parser.add_argument("--start", type=str, default="2010-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=datetime.today().strftime("%Y-%m-%d"),
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--model", type=str, default="linear",
                        choices=["linear", "random_forest"],
                        help="Regression model to use")
    parser.add_argument("--forecast_horizon", type=int, default=5,
                        help="Number of days to forecast into the future")
    parser.add_argument("--no_plots", action="store_true",
                        help="Suppress plotting of time series and moving averages")
    return parser.parse_args()


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download historical data for a given ticker from Yahoo Finance."""
    # Use yfinance to fetch historical data
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker symbol and date range.")
    return data


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add simple technical indicators to the dataset as additional features."""
    df = data.copy()
    # Simple moving averages
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    # Target variable: next day's closing price
    df["Target"] = df["Close"].shift(-1)
    # Drop rows with NaN values created by rolling and shifting
    df = df.dropna()
    return df


def train_model(df: pd.DataFrame, model_name: str):
    """Train a regression model on the feature set."""
    feature_cols = ["Open", "High", "Low", "Close", "Volume", "SMA_5", "SMA_10", "SMA_20"]
    X = df[feature_cols]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if model_name == "linear":
        model = LinearRegression()
    elif model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model: {model_name}\nMSE: {mse:.4f}  R^2: {r2:.4f}\n")
    return model, feature_cols


def forecast(model, df: pd.DataFrame, feature_cols, horizon: int) -> pd.DataFrame:
    """Generate forecasts for the specified number of future days.

    This function uses a simple iterative approach: it takes the last row
    of the existing dataset, predicts the next day's closing price and
    then appends that prediction as part of the feature set for the
    next prediction.  This is a simplistic method and should be
    replaced with a more sophisticated approach for real world use.
    """
    last_row = df.iloc[-1].copy()
    predictions = []
    current_date = df.index[-1]
    
    # Get the last few closing prices for moving average calculations
    last_closes = df["Close"].iloc[-20:].values  # Get last 20 values
    
    for i in range(horizon):
        # Prepare input for prediction
        X_last = last_row[feature_cols].values.reshape(1, -1)
        next_close = model.predict(X_last)[0]
        next_date = current_date + timedelta(days=1)
        predictions.append((next_date, next_close))
        
        # Update last_row for next iteration: shift features
        # For simplicity we set open/high/low/close equal to the predicted close
        last_row["Open"] = next_close
        last_row["High"] = next_close
        last_row["Low"] = next_close
        last_row["Close"] = next_close
        last_row["Volume"] = last_row["Volume"]  # volume unchanged
        
        # Update the last_closes array for moving average calculations
        last_closes = np.append(last_closes[1:], next_close)
        
        # Recompute moving averages
        for window, col in [(5, "SMA_5"), (10, "SMA_10"), (20, "SMA_20")]:
            if len(last_closes) >= window:
                last_row[col] = np.mean(last_closes[-window:])
            else:
                last_row[col] = np.mean(last_closes)
        
        current_date = next_date
    
    forecast_df = pd.DataFrame(predictions, columns=["Date", "Predicted_Close"])
    return forecast_df


def plot_data(data: pd.DataFrame, df: pd.DataFrame, ticker: str) -> None:
    """Display closing price and moving averages."""
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Close"], label="Close", linewidth=1.5)
    plt.plot(df.index, df["SMA_5"], label="5‑Day SMA", linestyle='--')
    plt.plot(df.index, df["SMA_10"], label="10‑Day SMA", linestyle='--')
    plt.plot(df.index, df["SMA_20"], label="20‑Day SMA", linestyle='--')
    plt.title(f"{ticker} Closing Price and Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    print(f"Downloading data for {args.ticker} from {args.start} to {args.end}...")
    data = download_data(args.ticker, args.start, args.end)
    df = add_features(data)
    if not args.no_plots:
        plot_data(data, df, args.ticker)
    model, feature_cols = train_model(df, args.model)
    forecast_df = forecast(model, df, feature_cols, args.forecast_horizon)
    print("Predicted closing prices:")
    for date, price in forecast_df.values:
        print(f"{date.date()}: ${price:.2f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
