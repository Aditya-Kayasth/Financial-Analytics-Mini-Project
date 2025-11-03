import plotly.graph_objects as go
import pandas as pd

def plot_price_and_forecast(data, forecast, title):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Portfolio Value'],
        name='Portfolio Value',
        line=dict(color='blue')
    ))

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
    
    forecast_dates = pd.date_range(start=data.index[-1], periods=31)[1:]
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