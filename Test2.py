import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from finvizfinance.quote import finvizfinance
import nltk
import warnings
import re
import time

# --- Setup and Configuration ---
st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
warnings.filterwarnings('ignore')

# Download NLTK data for sentiment analysis (only needs to run once)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- Data Caching & Analysis Functions ---

@st.cache_data
def get_portfolio_data(tickers, start_date='2020-01-01'):
    """
    Downloads all historical data from yfinance.
    """
    # Download without group_by for better compatibility
    data = yf.download(tickers, start=start_date)
    if data.empty:
        raise ValueError("No data returned from yfinance. Check tickers.")
    return data

@st.cache_data
def calculate_portfolio_performance(data, weights, tickers):
    """
    FIXED: Handles yfinance MultiIndex structure (metric, ticker).
    Uses 'Close' price when 'Adj Close' is not available.
    """
    try:
        # Handle different yfinance output structures
        if len(tickers) == 1:
            # Single ticker - simple column structure
            # For single ticker, yfinance returns a simple DataFrame
            if 'Adj Close' in data.columns:
                price_col = data['Adj Close']
            elif 'Close' in data.columns:
                price_col = data['Close']
            else:
                st.error(f"Available columns: {data.columns.tolist()}")
                return None
            
            # Check if it's already a Series or needs conversion
            if isinstance(price_col, pd.Series):
                price_data = pd.DataFrame({tickers[0]: price_col})
            elif isinstance(price_col, pd.DataFrame):
                # It's already a DataFrame, just rename the column
                price_data = price_col.copy()
                price_data.columns = tickers
            else:
                # Convert to DataFrame
                price_data = pd.DataFrame({tickers[0]: price_col.squeeze()}, index=data.index)
            
        else:
            # Multiple tickers - check for MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                # MultiIndex structure: (metric, ticker)
                level_0_cols = data.columns.get_level_values(0).unique().tolist()
                
                # Try 'Adj Close' first, fall back to 'Close'
                if 'Adj Close' in level_0_cols:
                    price_data = data['Adj Close'].copy()
                elif 'Close' in level_0_cols:
                    price_data = data['Close'].copy()
                else:
                    st.error(f"No price data found. Available: {level_0_cols}")
                    return None
            else:
                # Flat structure - shouldn't happen with multiple tickers but handle it
                st.error(f"Unexpected data structure: {data.columns.tolist()}")
                return None
        
        # Verify all tickers are present
        missing_cols = [t for t in tickers if t not in price_data.columns]
        if missing_cols:
            st.error(f"Missing data for: {', '.join(missing_cols)}")
            st.write(f"Available columns: {price_data.columns.tolist()}")
            return None
            
    except Exception as e:
        st.error(f"Error processing data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None
    
    # Drop any rows with NaN values
    price_data = price_data.dropna()
    
    if price_data.empty:
        st.error("No valid data after cleaning")
        return None
    
    # Normalize data to start at 100
    normalized_data = (price_data / price_data.iloc[0]) * 100
    
    # Apply weights to normalized data
    # Convert weights to array if needed and ensure proper broadcasting
    weights_array = np.array(weights)
    
    # Multiply normalized data by weights
    # For single ticker, this ensures proper handling
    weighted_data = normalized_data.copy()
    for i, ticker in enumerate(tickers):
        weighted_data[ticker] = normalized_data[ticker] * weights_array[i]
    
    # Sum to get portfolio index value
    portfolio_index = weighted_data[tickers].sum(axis=1)
    
    # Calculate Moving Averages
    portfolio_df = pd.DataFrame(portfolio_index, columns=['Portfolio Value'])
    portfolio_df['MA20'] = portfolio_df['Portfolio Value'].rolling(window=20).mean()
    portfolio_df['MA50'] = portfolio_df['Portfolio Value'].rolling(window=50).mean()
    
    return portfolio_df

@st.cache_data
def get_arima_forecast(data_series):
    """Runs an ARIMA(5,1,0) forecast on the provided time series."""
    try:
        model = ARIMA(data_series.dropna(), order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=30)
        return forecast
    except Exception as e:
        st.error(f"Error in ARIMA forecast: {e}")
        return pd.Series([data_series.iloc[-1]] * 30)

@st.cache_data
def get_portfolio_sentiment(tickers, weights):
    """
    FIXED: Uses finvizfinance correctly to get news for each ticker.
    """
    analyzer = SentimentIntensityAnalyzer()
    total_weighted_score = 0
    all_news_dfs = []

    for ticker, weight in zip(tickers, weights):
        try:
            # Initialize finvizfinance for this ticker
            stock = finvizfinance(ticker)
            
            # Get news using the correct method
            news_df = stock.ticker_news()
            
            if news_df is None or news_df.empty:
                st.warning(f"No news found for {ticker}.")
                continue
            
            # Get headlines
            if 'Title' in news_df.columns:
                headlines_list = news_df['Title'].tolist()
            elif 'title' in news_df.columns:
                headlines_list = news_df['title'].tolist()
            else:
                st.warning(f"Could not find title column for {ticker}")
                continue
            
            if not headlines_list:
                st.warning(f"No news headlines found for {ticker}.")
                continue

            # Calculate sentiment for this ticker
            ticker_total_score = 0
            for title in headlines_list:
                if title and isinstance(title, str):
                    ticker_total_score += analyzer.polarity_scores(title)['compound']
            
            if len(headlines_list) > 0:
                avg_ticker_score = ticker_total_score / len(headlines_list)
                total_weighted_score += avg_ticker_score * weight
            
            # Store for display
            news_df['Ticker'] = ticker
            all_news_dfs.append(news_df)
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            st.error(f"Error processing news for {ticker}: {e}")
            continue
            
    if not all_news_dfs:
        return 0, pd.DataFrame(columns=['Date', 'Title', 'Link', 'Ticker'])

    # Combine all news into one DataFrame
    full_news_df = pd.concat(all_news_dfs, ignore_index=True)
    
    # Standardize column names
    if 'Title' not in full_news_df.columns and 'title' in full_news_df.columns:
        full_news_df.rename(columns={'title': 'Title'}, inplace=True)
    if 'Date' not in full_news_df.columns and 'date' in full_news_df.columns:
        full_news_df.rename(columns={'date': 'Date'}, inplace=True)
    if 'Link' not in full_news_df.columns and 'link' in full_news_df.columns:
        full_news_df.rename(columns={'link': 'Link'}, inplace=True)
    
    # Clean up date formatting
    if 'Date' in full_news_df.columns:
        full_news_df['Date'] = pd.to_datetime(full_news_df['Date'], errors='coerce')
        full_news_df = full_news_df.sort_values(by='Date', ascending=False)
    
    return total_weighted_score, full_news_df

# --- Plotting Functions ---

def plot_price_and_forecast(data, forecast, title):
    """Plots the historical portfolio value, MAs, and ARIMA forecast."""
    fig = go.Figure()

    # 1. Historical Portfolio Value
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Portfolio Value'],
        name='Portfolio Value',
        line=dict(color='blue')
    ))

    # 2. Moving Averages
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MA20'],
        name='20-Day MA',
        line=dict(color='orange', dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MA50'],
        name='50-Day MA',
        line=dict(color='green', dash='dot')
    ))
    
    # 3. ARIMA Forecast
    forecast_dates = pd.date_range(start=data.index[-1], periods=31)[1:]  # Skip first to avoid overlap
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast,
        name='ARIMA Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Portfolio Index Value (Normalized to 100)',
        legend_title='Legend',
        hovermode='x unified',
        height=500
    )
    return fig

def plot_sentiment_gauge(score):
    """Plots the weighted sentiment score on a gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Portfolio-Weighted News Sentiment"},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.2], 'color': "red"},
                {'range': [-0.2, 0.2], 'color': "gray"},
                {'range': [0.2, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(height=400)
    return fig

# --- Helper Functions ---

def parse_inputs(ticker_str, weight_str):
    """Parses and validates the user's ticker and weight inputs."""
    
    # Parse tickers
    tickers = [t.strip().upper() for t in re.split(r'[,\s]+', ticker_str) if t]
    
    # Parse weights
    try:
        weights = [float(w.strip()) for w in re.split(r'[,\s]+', weight_str) if w]
    except ValueError:
        st.error("Error: Weights must be numbers. Example: 0.5, 0.25, 0.25")
        return None, None

    # --- Validation ---
    if not tickers:
        st.warning("Please enter at least one ticker.")
        return None, None
        
    if len(tickers) != len(weights):
        st.error(f"Error: Mismatch! You entered {len(tickers)} tickers but {len(weights)} weights. Please provide one weight for each ticker.")
        return None, None
    
    if not np.isclose(sum(weights), 1.0):
        st.error(f"Error: Your weights sum to {sum(weights):.2f}. They must sum to 1.0.")
        return None, None
        
    return tickers, np.array(weights)

# --- Streamlit App Layout ---

st.title("📈 Portfolio Analysis & Insights Dashboard")

# --- Sidebar for User Input ---
st.sidebar.header("Portfolio Controls")
ticker_input = st.sidebar.text_input("Enter Tickers (comma/space separated)", "AAPL, MSFT, GOOG, AMZN")
weight_input = st.sidebar.text_input("Enter Weights (comma/space separated)", "0.25, 0.25, 0.25, 0.25")

analyze_button = st.sidebar.button("Analyze Portfolio")

if analyze_button:
    tickers, weights = parse_inputs(ticker_input, weight_input)
    
    if tickers and weights is not None:
        # --- Main Dashboard Area ---
        portfolio_title = f"Portfolio Analysis: ({', '.join(f'{t} {w*100:.0f}%' for t, w in zip(tickers, weights))})"
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(portfolio_title)
            with st.spinner('Loading portfolio data and running forecast...'):
                try:
                    # 1. Get Data
                    stock_data = get_portfolio_data(tickers)
                    
                    # 2. Calculate Portfolio Performance
                    portfolio_df = calculate_portfolio_performance(stock_data, weights, tickers)
                    
                    if portfolio_df is not None and not portfolio_df.empty:
                        # 3. Get Forecast
                        forecast = get_arima_forecast(portfolio_df['Portfolio Value'])
                        
                        # 4. Plot Price and Forecast
                        price_fig = plot_price_and_forecast(portfolio_df, forecast, portfolio_title)
                        st.plotly_chart(price_fig, use_container_width=True)
                        
                        # 5. Generate insights
                        current_value = portfolio_df['Portfolio Value'].iloc[-1]
                        forecast_end = forecast.iloc[-1]
                        change_pct = ((forecast_end - current_value) / current_value) * 100
                        
                        ma20_current = portfolio_df['MA20'].iloc[-1]
                        ma50_current = portfolio_df['MA50'].iloc[-1]
                        
                        # Display insights
                        st.subheader("📊 Portfolio Insights")
                        
                        # Forecast insight
                        if change_pct > 5:
                            trend_emoji = "📈"
                            trend_word = "strong upward"
                            recommendation = "The model predicts significant growth. This is a positive signal."
                        elif change_pct > 0:
                            trend_emoji = "📊"
                            trend_word = "modest upward"
                            recommendation = "The model predicts slight growth. Market looks stable."
                        elif change_pct > -5:
                            trend_emoji = "📉"
                            trend_word = "slight downward"
                            recommendation = "The model predicts minor decline. Consider monitoring closely."
                        else:
                            trend_emoji = "⚠️"
                            trend_word = "significant downward"
                            recommendation = "The model predicts notable decline. Review your positions."
                        
                        st.info(f"""
                        **30-Day Forecast:** {trend_emoji}
                        
                        Your portfolio is currently valued at **{current_value:.2f}** (normalized). 
                        The forecast predicts a **{trend_word} trend** over the next 30 days, 
                        with an expected value of **{forecast_end:.2f}** (a **{change_pct:+.2f}%** change).
                        
                        {recommendation}
                        """)
                        
                        # Moving average insight
                        if current_value > ma20_current > ma50_current:
                            ma_signal = "🟢 **Bullish Signal**: Your portfolio is above both moving averages, suggesting upward momentum."
                        elif current_value < ma20_current < ma50_current:
                            ma_signal = "🔴 **Bearish Signal**: Your portfolio is below both moving averages, suggesting downward pressure."
                        else:
                            ma_signal = "🟡 **Mixed Signal**: Moving averages show conflicting trends. Market may be consolidating."
                        
                        st.info(f"""
                        **Technical Analysis:**
                        
                        {ma_signal}
                        
                        - **20-Day MA**: {ma20_current:.2f} (short-term trend)
                        - **50-Day MA**: {ma50_current:.2f} (medium-term trend)
                        """)
                        
                    else:
                        st.error("Could not generate portfolio data")
                    
                except Exception as e:
                    st.error(f"Error loading price data: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        with col2:
            st.subheader("Sentiment Analysis")
            with st.spinner('Fetching and analyzing news headlines...'):
                try:
                    # 5. Get Sentiment
                    sentiment_score, news_df = get_portfolio_sentiment(tickers, weights)
                    
                    # 6. Plot Gauge
                    gauge_fig = plot_sentiment_gauge(sentiment_score)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    # Show sentiment interpretation
                    if sentiment_score > 0.2:
                        sentiment_text = "🟢 **Positive News Sentiment**"
                        sentiment_detail = "Recent news about your portfolio stocks is generally positive. This could support price appreciation."
                    elif sentiment_score < -0.2:
                        sentiment_text = "🔴 **Negative News Sentiment**"
                        sentiment_detail = "Recent news about your portfolio stocks is generally negative. This could create headwinds for prices."
                    else:
                        sentiment_text = "🟡 **Neutral News Sentiment**"
                        sentiment_detail = "Recent news about your portfolio stocks is balanced. No strong positive or negative signals from media."
                    
                    st.markdown(f"""
                    {sentiment_text}
                    
                    **Score: {sentiment_score:.3f}**
                    
                    {sentiment_detail}
                    """)
                    
                except Exception as e:
                    st.error(f"Error loading sentiment data: {e}")

        # --- Section for Raw News Data ---
        st.subheader("Recent News Headlines (All Tickers)")
        try:
            if 'news_df' in locals() and not news_df.empty:
                # Select columns to display
                display_cols = []
                if 'Date' in news_df.columns:
                    display_cols.append('Date')
                if 'Ticker' in news_df.columns:
                    display_cols.append('Ticker')
                if 'Title' in news_df.columns:
                    display_cols.append('Title')
                if 'Link' in news_df.columns:
                    display_cols.append('Link')
                
                if display_cols:
                    st.dataframe(news_df[display_cols], use_container_width=True, height=500)
                else:
                    st.write("News data structure is different than expected.")
                    st.dataframe(news_df.head(), use_container_width=True)
            else:
                st.write("No news data to display.")
        except Exception as e:
            st.write("No news data to display.")
else:
    st.info("Enter your portfolio tickers and weights in the sidebar and click 'Analyze' to begin.")