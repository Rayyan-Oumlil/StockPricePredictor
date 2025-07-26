import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Comparison Tool", page_icon="📊", layout="wide")

st.header("📊 Stock Comparison Tool")
st.info("Compare multiple stocks side-by-side with charts and metrics")

# Popular stocks for quick selection
popular_stocks = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "V",
    "PG", "HD", "MA", "UNH", "BAC", "PFE", "ABBV", "KO", "PEP", "TMO"
]

# Stock selection
st.subheader("📋 Select Stocks to Compare")
col1, col2 = st.columns(2)

with col1:
    st.write("**Quick Selection**")
    selected_quick = st.multiselect(
        "Choose from popular stocks:",
        popular_stocks,
        default=["AAPL", "MSFT", "GOOGL"]
    )

with col2:
    st.write("**Custom Selection**")
    custom_tickers = st.text_input(
        "Enter custom tickers (comma-separated):",
        placeholder="e.g., NVDA, TSLA, AMD"
    )
    
    if custom_tickers:
        custom_list = [ticker.strip().upper() for ticker in custom_tickers.split(",") if ticker.strip()]
    else:
        custom_list = []

# Combine selected stocks
all_selected = list(set(selected_quick + custom_list))

if not all_selected:
    st.warning("Please select at least one stock to compare.")
    st.stop()

# Time period selection
st.subheader("⏰ Time Period")
period = st.selectbox(
    "Select time period:",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=2
)

# Get data for comparison
@st.cache_data(ttl=300)
def get_comparison_data(tickers, period):
    """Get historical data for comparison"""
    comparison_data = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if not hist.empty:
                # Calculate additional metrics
                hist['Returns'] = hist['Close'].pct_change()
                hist['Cumulative_Returns'] = (1 + hist['Returns']).cumprod()
                hist['Volatility'] = hist['Returns'].rolling(20).std() * np.sqrt(252) * 100
                
                comparison_data[ticker] = {
                    'data': hist,
                    'info': stock.info
                }
        except Exception as e:
            st.error(f"Error loading data for {ticker}: {str(e)}")
    
    return comparison_data

# Load data
with st.spinner("Loading comparison data..."):
    comparison_data = get_comparison_data(all_selected, period)

if not comparison_data:
    st.error("Unable to load data for any of the selected stocks.")
    st.stop()

# Price comparison chart
st.subheader("📈 Price Comparison")
if comparison_data:
    fig_price = go.Figure()
    
    for ticker, data in comparison_data.items():
        hist = data['data']
        fig_price.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist['Close'],
                mode='lines',
                name=ticker,
                hovertemplate=f'{ticker}<br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
            )
        )
    
    fig_price.update_layout(
        title=f"Price Comparison ({period})",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig_price, use_container_width=True)

# Performance metrics
st.subheader("📊 Performance Metrics")
if comparison_data:
    metrics_data = []
    
    for ticker, data in comparison_data.items():
        hist = data['data']
        info = data['info']
        
        if not hist.empty:
            # Calculate metrics
            start_price = hist['Close'].iloc[0]
            end_price = hist['Close'].iloc[-1]
            total_return = (end_price / start_price - 1) * 100
            
            # Volatility (annualized)
            volatility = hist['Returns'].std() * np.sqrt(252) * 100
            
            # Sharpe ratio (simplified, assuming 0% risk-free rate)
            sharpe = (hist['Returns'].mean() * 252) / (hist['Returns'].std() * np.sqrt(252)) if hist['Returns'].std() > 0 else 0
            
            # Max drawdown
            cumulative = hist['Cumulative_Returns']
            running_max = cumulative.expanding().max()
            drawdown = (cumulative / running_max - 1) * 100
            max_drawdown = drawdown.min()
            
            # Current metrics
            current_price = hist['Close'].iloc[-1]
            volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            
            metrics_data.append({
                'ticker': ticker,
                'name': info.get('longName', ticker),
                'current_price': current_price,
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'volume': volume,
                'avg_volume': avg_volume,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0)
            })
    
    if metrics_data:
        # Create metrics table
        df_metrics = pd.DataFrame(metrics_data)
        
        # Format display
        display_df = df_metrics.copy()
        display_df['Current Price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
        display_df['Total Return'] = display_df['total_return'].apply(lambda x: f"{x:+.2f}%")
        display_df['Volatility'] = display_df['volatility'].apply(lambda x: f"{x:.1f}%")
        display_df['Sharpe Ratio'] = display_df['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
        display_df['Max Drawdown'] = display_df['max_drawdown'].apply(lambda x: f"{x:.1f}%")
        display_df['Market Cap'] = display_df['market_cap'].apply(lambda x: f"${x/1e9:.1f}B" if x >= 1e9 else f"${x/1e6:.1f}M")
        display_df['P/E Ratio'] = display_df['pe_ratio'].apply(lambda x: f"{x:.1f}" if x > 0 else "N/A")
        
        # Reorder columns
        display_df = display_df[['ticker', 'name', 'Current Price', 'Total Return', 'Volatility', 
                               'Sharpe Ratio', 'Max Drawdown', 'Market Cap', 'P/E Ratio']]
        
        st.dataframe(display_df, use_container_width=True)

# Performance comparison charts
st.subheader("📈 Performance Analysis")
if comparison_data and len(comparison_data) > 1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Returns comparison
        fig_returns = go.Figure()
        
        for ticker, data in comparison_data.items():
            hist = data['data']
            cumulative_returns = (1 + hist['Returns']).cumprod() * 100
            
            fig_returns.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=cumulative_returns,
                    mode='lines',
                    name=ticker,
                    hovertemplate=f'{ticker}<br>Date: %{{x}}<br>Return: %{{y:.1f}}%<extra></extra>'
                )
            )
        
        fig_returns.update_layout(
            title="Cumulative Returns Comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            height=400
        )
        
        st.plotly_chart(fig_returns, use_container_width=True)
    
    with col2:
        # Volatility comparison
        fig_vol = go.Figure()
        
        for ticker, data in comparison_data.items():
            hist = data['data']
            volatility = hist['Returns'].rolling(20).std() * np.sqrt(252) * 100
            
            fig_vol.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=volatility,
                    mode='lines',
                    name=ticker,
                    hovertemplate=f'{ticker}<br>Date: %{{x}}<br>Volatility: %{{y:.1f}}%<extra></extra>'
                )
            )
        
        fig_vol.update_layout(
            title="Rolling Volatility (20-day)",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            height=400
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)

# Risk-return scatter plot
st.subheader("🎯 Risk-Return Analysis")
if comparison_data and len(comparison_data) > 1:
    risk_return_data = []
    
    for ticker, data in comparison_data.items():
        hist = data['data']
        total_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
        volatility = hist['Returns'].std() * np.sqrt(252) * 100
        
        risk_return_data.append({
            'ticker': ticker,
            'return': total_return,
            'risk': volatility
        })
    
    if risk_return_data:
        df_risk_return = pd.DataFrame(risk_return_data)
        
        fig_scatter = px.scatter(
            df_risk_return,
            x='risk',
            y='return',
            text='ticker',
            title="Risk-Return Scatter Plot"
        )
        
        fig_scatter.update_traces(
            textposition="top center",
            marker=dict(size=12)
        )
        
        fig_scatter.update_layout(
            xaxis_title="Risk (Volatility %)",
            yaxis_title="Return (%)",
            height=500
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)

# Correlation matrix
st.subheader("🔗 Correlation Matrix")
if comparison_data and len(comparison_data) > 1:
    # Create returns dataframe
    returns_df = pd.DataFrame()
    
    for ticker, data in comparison_data.items():
        returns_df[ticker] = data['data']['Returns']
    
    # Calculate correlation matrix
    corr_matrix = returns_df.corr()
    
    # Create heatmap
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Returns Correlation Matrix"
    )
    
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)

# Summary insights
st.subheader("💡 Summary Insights")
if comparison_data and len(comparison_data) > 1:
    # Find best and worst performers
    returns_list = []
    for ticker, data in comparison_data.items():
        hist = data['data']
        total_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
        returns_list.append((ticker, total_return))
    
    returns_list.sort(key=lambda x: x[1], reverse=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🏆 Performance Ranking**")
        for i, (ticker, return_pct) in enumerate(returns_list, 1):
            st.write(f"{i}. **{ticker}**: {return_pct:+.2f}%")
    
    with col2:
        st.write("**📊 Key Insights**")
        best_performer = returns_list[0]
        worst_performer = returns_list[-1]
        
        st.write(f"• **Best performer**: {best_performer[0]} ({best_performer[1]:+.2f}%)")
        st.write(f"• **Worst performer**: {worst_performer[0]} ({worst_performer[1]:+.2f}%)")
        
        # Calculate average correlation
        if len(returns_list) > 2:
            returns_df = pd.DataFrame()
            for ticker, data in comparison_data.items():
                returns_df[ticker] = data['data']['Returns']
            
            corr_matrix = returns_df.corr()
            avg_corr = (corr_matrix.sum().sum() - len(corr_matrix)) / (len(corr_matrix) ** 2 - len(corr_matrix))
            st.write(f"• **Average correlation**: {avg_corr:.2f}")

st.markdown("---")
st.caption("Data provided by Yahoo Finance. Past performance does not guarantee future results.") 