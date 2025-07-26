# StockPricePredictor
This project provides a simple yet extensible framework for predicting
future stock prices based on historical data.  The goal is to give
developers, data‑scientists and hobbyists a starting point for
experimenting with different machine‑learning and deep‑learning
approaches to time series forecasting.

The current implementation covers the basics:

- **Data acquisition** – Uses the [`yfinance`](https://pypi.org/project/yfinance/) package to download
  historical stock data directly from Yahoo Finance.
- **Data exploration** – Generates plots of the closing price and
  moving averages to help you understand trend and volatility.
- **Machine learning model** – Trains a regression model using
  `scikit‑learn` (Linear Regression by default) to forecast future
  closing prices.  The code is structured so that you can easily
  substitute other models (e.g. Random Forest, Gradient Boosting).
- **(Optional) Deep learning model** – The project layout leaves room
  for adding an LSTM or other recurrent neural network.  See
  comments in the source code for guidance on where to implement this.

## Installation

Clone or download this repository and install the dependencies listed in
`requirements.txt`.  A Python 3.9+ environment is recommended.

```bash
python -m venv venv        # optional – create a virtual environment
source venv/bin/activate   # on Windows use `venv\\Scripts\\activate`
pip install -r requirements.txt
```

## Usage

Run the main script with the stock ticker symbol and date range you
would like to analyse.  The default ticker is **AAPL** and the
default date range is from 2010‑01‑01 to today.

```bash
python main.py --ticker AAPL --start 2010-01-01 --end 2024-01-01
```

Options:

* `--ticker` – Stock symbol to download (e.g. `MSFT`, `TSLA`).
* `--start` – Start date in YYYY‑MM‑DD format.
* `--end` – End date in YYYY‑MM‑DD format.
* `--model` – Choice of regression model (e.g. `linear`, `random_forest`).
* `--forecast_horizon` – How many days into the future to forecast (default: 5).

The script will download the data, plot the closing price along with
moving averages, train the selected model on the historical data and
print predictions for the specified number of future days.

## Extending the Project

- **Additional indicators** – Add columns to the feature set such as
  technical indicators (RSI, MACD, Bollinger Bands) to capture more
  nuanced market behaviour.
- **Hyper‑parameter tuning** – Use tools such as `GridSearchCV` or
  `RandomizedSearchCV` to optimise model parameters.
- **LSTM or GRU models** – Replace or complement the `scikit‑learn`
  models with a recurrent neural network using `TensorFlow` or
  `PyTorch`.  The current structure makes it easy to add such a
  function.
- **Streamlit dashboard** – Build a simple user interface with
  [`Streamlit`](https://streamlit.io/) to interactively select
  tickers, date ranges and models and display results.

## License

This project is licensed under the MIT License.  See the `LICENSE`
file for details.
