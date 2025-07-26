import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Market Overview", page_icon="📈", layout="wide")

st.header("📈 Market Overview")
st.info("Real-time market summary and performance metrics")

# Popular indices and sectors
indices = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC", 
    "DOW JONES": "^DJI",
    "RUSSELL 2000": "^RUT"
}

sectors = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT"],
    "Financial": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK"],
    "Consumer": ["PG", "KO", "WMT", "HD", "MCD", "NKE", "SBUX"]
}

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_market_data():
    """Get real-time market data"""
    market_data = {}
    
    # Get major indices
    for name, symbol in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="5d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
                
                market_data[name] = {
                    'symbol': symbol,
                    'price': current_price,
                    'change': change,
                    'change_pct': change_pct,
                    'volume': hist['Volume'].iloc[-1]
                }
        except:
            continue
    
    return market_data

@st.cache_data(ttl=300)
def get_sector_performance():
    """Get sector performance data"""
    sector_data = {}
    
    for sector_name, tickers in sectors.items():
        sector_prices = []
        sector_changes = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d")
                if not hist.empty and len(hist) > 1:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2]
                    change_pct = ((current - previous) / previous) * 100
                    sector_prices.append(current)
                    sector_changes.append(change_pct)
            except:
                continue
        
        if sector_changes:
            sector_data[sector_name] = {
                'avg_change': np.mean(sector_changes),
                'stocks': len(sector_changes),
                'positive': sum(1 for x in sector_changes if x > 0)
            }
    
    return sector_data

@st.cache_data(ttl=300)
def get_top_movers():
    """Get top gainers and losers"""
    # Popular stocks for top movers
    popular_stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "V",
        "PG", "HD", "MA", "UNH", "BAC", "PFE", "ABBV", "KO", "PEP", "TMO"
    ]
    
    movers = []
    
    for ticker in popular_stocks:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if not hist.empty and len(hist) > 1:
                current = hist['Close'].iloc[-1]
                previous = hist['Close'].iloc[-2]
                change = current - previous
                change_pct = (change / previous) * 100
                volume = hist['Volume'].iloc[-1]
                
                movers.append({
                    'ticker': ticker,
                    'price': current,
                    'change': change,
                    'change_pct': change_pct,
                    'volume': volume
                })
        except:
            continue
    
    return movers

# Load data
with st.spinner("Loading market data..."):
    market_data = get_market_data()
    sector_data = get_sector_performance()
    movers = get_top_movers()

# Market indices overview
st.subheader("📊 Major Indices")
if market_data:
    cols = st.columns(len(market_data))
    
    for i, (name, data) in enumerate(market_data.items()):
        with cols[i]:
            color = "green" if data['change_pct'] >= 0 else "red"
            st.metric(
                label=name,
                value=f"${data['price']:,.2f}",
                delta=f"{data['change_pct']:+.2f}%",
                delta_color="normal"
            )

# Top gainers and losers
st.subheader("🔥 Top Movers")
if movers:
    # Sort by percentage change
    gainers = sorted([m for m in movers if m['change_pct'] > 0], key=lambda x: x['change_pct'], reverse=True)[:5]
    losers = sorted([m for m in movers if m['change_pct'] < 0], key=lambda x: x['change_pct'])[:5]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📈 Top Gainers**")
        for stock in gainers:
            st.write(f"**{stock['ticker']}** ${stock['price']:.2f} (+{stock['change_pct']:.2f}%)")
    
    with col2:
        st.write("**📉 Top Losers**")
        for stock in losers:
            st.write(f"**{stock['ticker']}** ${stock['price']:.2f} ({stock['change_pct']:.2f}%)")

# Sector performance
st.subheader("🏭 Sector Performance")
if sector_data:
    # Create sector performance chart
    sectors_list = list(sector_data.keys())
    changes = [sector_data[s]['avg_change'] for s in sectors_list]
    colors = ['green' if x >= 0 else 'red' for x in changes]
    
    fig = go.Figure(data=[
        go.Bar(
            x=sectors_list,
            y=changes,
            marker_color=colors,
            text=[f"{x:.2f}%" for x in changes],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Average Sector Performance (5-day change)",
        xaxis_title="Sector",
        yaxis_title="Average Change (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sector details table
    sector_df = pd.DataFrame([
        {
            'Sector': sector,
            'Avg Change (%)': f"{data['avg_change']:.2f}%",
            'Stocks Tracked': data['stocks'],
            'Positive Stocks': data['positive']
        }
        for sector, data in sector_data.items()
    ])
    
    st.dataframe(sector_df, use_container_width=True)

else:
    st.warning("Unable to load sector data at the moment.")

# Market sentiment
st.subheader("📊 Market Sentiment")
if movers:
    positive_count = sum(1 for m in movers if m['change_pct'] > 0)
    negative_count = sum(1 for m in movers if m['change_pct'] < 0)
    total = len(movers)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Positive Stocks", f"{positive_count}", f"{positive_count/total*100:.1f}%")
    
    with col2:
        st.metric("Negative Stocks", f"{negative_count}", f"{negative_count/total*100:.1f}%")
    
    with col3:
        avg_change = np.mean([m['change_pct'] for m in movers])
        st.metric("Average Change", f"{avg_change:.2f}%")

st.markdown("---")
st.caption("Data refreshes every 5 minutes. Market data provided by Yahoo Finance.") 