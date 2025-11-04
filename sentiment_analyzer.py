import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

@st.cache_data
def calculate_weighted_sentiment(news_df, tickers, weights):
    if news_df.empty:
        return 0

    analyzer = SentimentIntensityAnalyzer()
    total_weighted_score = 0
    weights_map = {ticker: weight for ticker, weight in zip(tickers, weights)}

    for ticker in tickers:
        ticker_news = news_df[news_df['Ticker'] == ticker]
        if ticker_news.empty:
            continue

        if 'Title' not in ticker_news.columns:
            st.warning(f"No 'Title' column for {ticker} news.")
            continue

        headlines_list = ticker_news['Title'].dropna().tolist()
        if not headlines_list:
            continue

        ticker_total_score = sum(
            analyzer.polarity_scores(title)['compound']
            for title in headlines_list if isinstance(title, str)
        )

        if headlines_list:
            avg_ticker_score = ticker_total_score / len(headlines_list)
            total_weighted_score += avg_ticker_score * weights_map.get(ticker, 0)

    return total_weighted_score
