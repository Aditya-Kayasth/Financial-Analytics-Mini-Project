import streamlit as st
import numpy as np
import re

def parse_inputs(ticker_str, weight_str):
    
    tickers = [t.strip().upper() for t in re.split(r'[,\s]+', ticker_str) if t]
    
    try:
        weights = [float(w.strip()) for w in re.split(r'[,\s]+', weight_str) if w]
    except ValueError:
        st.error("Error: Weights must be numbers. Example: 0.5, 0.25, 0.25")
        return None, None

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