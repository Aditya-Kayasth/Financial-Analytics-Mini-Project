import streamlit as st
import yfinance as yf
import pandas as pd
from finvizfinance.quote import finvizfinance
import time

@st.cache_data
def get_stock_data(tickers, start_date='2020-01-01'):
    data = yf.download(tickers, start=start_date)
    if data.empty:
        raise ValueError("No data returned from yfinance. Check tickers.")
    return data

@st.cache_data
def get_news_data(tickers):
    all_news_dfs = []
    for ticker in tickers:
        try:
            stock = finvizfinance(ticker)
            news_df = stock.ticker_news()
            
            if news_df is None or news_df.empty:
                st.warning(f"No news found for {ticker}.")
                continue
                
            news_df['Ticker'] = ticker
            all_news_dfs.append(news_df)
            time.sleep(0.5)
            
        except Exception as e:
            st.error(f"Error processing news for {ticker}: {e}")
            continue
            
    if not all_news_dfs:
        return pd.DataFrame(columns=['Date', 'Title', 'Link', 'Ticker'])

    full_news_df = pd.concat(all_news_dfs, ignore_index=True)
    
    if 'Title' not in full_news_df.columns and 'title' in full_news_df.columns:
        full_news_df.rename(columns={'title': 'Title'}, inplace=True)
    if 'Date' not in full_news_df.columns and 'date' in full_news_df.columns:
        full_news_df.rename(columns={'date': 'Date'}, inplace=True)
    if 'Link' not in full_news_df.columns and 'link' in full_news_df.columns:
        full_news_df.rename(columns={'link': 'Link'}, inplace=True)
    
    if 'Date' in full_news_df.columns:
        full_news_df['Date'] = pd.to_datetime(full_news_df['Date'], errors='coerce')
        full_news_df = full_news_df.sort_values(by='Date', ascending=False)
        
    return full_news_df