import streamlit as st
import warnings
import traceback
import numpy as np

import data_fetcher
import portfolio_analyzer
import sentiment_analyzer
from plotting import plot_price_and_forecast, plot_sentiment_gauge
from utils import parse_inputs

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
warnings.filterwarnings('ignore')

st.title("📈 Portfolio Analysis & Insights Dashboard")

st.sidebar.header("Portfolio Controls")
ticker_input = st.sidebar.text_input("Enter Tickers (comma/space separated)", "AAPL, MSFT, GOOG, AMZN")
weight_input = st.sidebar.text_input("Enter Weights (comma/space separated)", "0.25, 0.25, 0.25, 0.25")

analyze_button = st.sidebar.button("Analyze Portfolio")

if analyze_button:
    tickers, weights = parse_inputs(ticker_input, weight_input)
    
    if tickers and weights is not None:
        portfolio_title = f"Portfolio Analysis: ({', '.join(f'{t} {w*100:.0f}%' for t, w in zip(tickers, weights))})"
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(portfolio_title)
            with st.spinner('Loading portfolio data and running forecast...'):
                try:
                    stock_data = data_fetcher.get_stock_data(tickers)
                    
                    portfolio_df = portfolio_analyzer.calculate_portfolio_performance(
                        stock_data, weights, tickers
                    )
                    
                    if portfolio_df is not None and not portfolio_df.empty:
                        forecast = portfolio_analyzer.get_arima_forecast(
                            portfolio_df['Portfolio Value']
                        )
                        
                        price_fig = plot_price_and_forecast(portfolio_df, forecast, portfolio_title)
                        st.plotly_chart(price_fig, use_container_width=True)
                        
                        current_value = portfolio_df['Portfolio Value'].iloc[-1]
                        forecast_end = forecast.iloc[-1]
                        change_pct = ((forecast_end - current_value) / current_value) * 100
                        
                        ma20_current = portfolio_df['MA20'].iloc[-1]
                        ma50_current = portfolio_df['MA50'].iloc[-1]
                        
                        st.subheader("📊 Portfolio Insights")
                        
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
                    st.code(traceback.format_exc())

        with col2:
            st.subheader("Sentiment Analysis")
            with st.spinner('Fetching and analyzing news headlines...'):
                try:
                    news_df = data_fetcher.get_news_data(tickers)
                    
                    sentiment_score = sentiment_analyzer.calculate_weighted_sentiment(
                        news_df, tickers, weights
                    )
                    
                    gauge_fig = plot_sentiment_gauge(sentiment_score)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    
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

        st.subheader("Recent News Headlines (All Tickers)")
        try:
            if 'news_df' in locals() and not news_df.empty:
                display_cols = ['Date', 'Ticker', 'Title', 'Link']
                
                # Filter to columns that actually exist
                display_cols = [col for col in display_cols if col in news_df.columns]
                
                if display_cols:
                    st.dataframe(news_df[display_cols], use_container_width=True, height=500)
                else:
                    st.write("News data structure is different than expected.")
                    st.dataframe(news_df.head(), use_container_width=True)
            else:
                st.write("No news data to display.")
        except Exception as e:
            st.write(f"Error displaying news: {e}")
else:
    st.info("Enter your portfolio tickers and weights in the sidebar and click 'Analyze' to begin.")