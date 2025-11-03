import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import traceback

@st.cache_data
def calculate_portfolio_performance(data, weights, tickers):
    try:
        if len(tickers) == 1:
            if 'Adj Close' in data.columns:
                price_col = data['Adj Close']
            elif 'Close' in data.columns:
                price_col = data['Close']
            else:
                st.error(f"Available columns: {data.columns.tolist()}")
                return None
            
            if isinstance(price_col, pd.Series):
                price_data = pd.DataFrame({tickers[0]: price_col})
            elif isinstance(price_col, pd.DataFrame):
                price_data = price_col.copy()
                price_data.columns = tickers
            else:
                price_data = pd.DataFrame({tickers[0]: price_col.squeeze()}, index=data.index)
            
        else:
            if isinstance(data.columns, pd.MultiIndex):
                level_0_cols = data.columns.get_level_values(0).unique().tolist()
                
                if 'Adj Close' in level_0_cols:
                    price_data = data['Adj Close'].copy()
                elif 'Close' in level_0_cols:
                    price_data = data['Close'].copy()
                else:
                    st.error(f"No price data found. Available: {level_0_cols}")
                    return None
            else:
                st.error(f"Unexpected data structure: {data.columns.tolist()}")
                return None
        
        missing_cols = [t for t in tickers if t not in price_data.columns]
        if missing_cols:
            st.error(f"Missing data for: {', '.join(missing_cols)}")
            st.write(f"Available columns: {price_data.columns.tolist()}")
            return None
            
    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.code(traceback.format_exc())
        return None
    
    price_data = price_data.dropna()
    
    if price_data.empty:
        st.error("No valid data after cleaning")
        return None
    
    normalized_data = (price_data / price_data.iloc[0]) * 100
    
    weights_array = np.array(weights)
    
    weighted_data = normalized_data.copy()
    for i, ticker in enumerate(tickers):
        weighted_data[ticker] = normalized_data[ticker] * weights_array[i]
    
    portfolio_index = weighted_data[tickers].sum(axis=1)
    
    portfolio_df = pd.DataFrame(portfolio_index, columns=['Portfolio Value'])
    portfolio_df['MA20'] = portfolio_df['Portfolio Value'].rolling(window=20).mean()
    portfolio_df['MA50'] = portfolio_df['Portfolio Value'].rolling(window=50).mean()
    
    return portfolio_df

@st.cache_data
def get_arima_forecast(data_series):
    try:
        model = ARIMA(data_series.dropna(), order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=30)
        return forecast
    except Exception as e:
        st.error(f"Error in ARIMA forecast: {e}")
        return pd.Series([data_series.iloc[-1]] * 30)