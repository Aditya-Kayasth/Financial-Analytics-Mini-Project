

# Financial Analytics Portfolio Dashboard

**Live Application:** [Launch Dashboard](https://aditya-kayasth-financial-analytics-mini-project-app-krfobh.streamlit.app/)

## Project Overview

This project was developed as a laboratory exercise to examine the practical application of financial analytics theories and computational tools in a real-world scenario. The objective was to construct a unified dashboard that integrates quantitative market data with qualitative sentiment analysis to assess portfolio performance and forecast future trends.

The application serves as a technical demonstration of time-series forecasting, natural language processing (NLP), and portfolio optimization techniques using Python.

## Key Features

* **Portfolio Performance Normalization:** Aggregates multiple stock tickers and weights to calculate a normalized portfolio index (Base 100), allowing for comparative performance analysis regardless of individual share prices.
* **Predictive Modeling (ARIMA):** Utilizes the AutoRegressive Integrated Moving Average (ARIMA) model from the `statsmodels` library to generate a 30-day forward-looking trend forecast based on historical portfolio values.
* **Technical Analysis Indicators:** Implements 20-Day and 50-Day Moving Averages (MA) to identify short-term and medium-term market momentum and potential crossover signals.
* **Weighted Sentiment Analysis:** Scrapes recent financial news headlines via FinViz and applies NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) lexicon to compute a composite sentiment score, weighted by the portfolio's asset allocation.
* **Interactive Visualization:** Uses Plotly for dynamic plotting of price history, confidence intervals, and sentiment gauges.

## Tech Stack

## Technical Architecture

The project is built using a modular Python architecture.

* `app.py`: Main entry point and dashboard layout logic.
* `data_fetcher.py`: Handles API calls to retrieve historical stock prices and news metadata.
* `portfolio_analyzer.py`: Contains the logic for mathematical normalization, moving average calculations, and ARIMA forecasting.
* `sentiment_analyzer.py`: Processes text data to derive polarity scores.
* `plotting.py`: Encapsulates the visualization logic for charts and gauges.
* `utils.py`: Helper functions for input parsing and validation.

## Local Installation and Setup

To run this project locally, follow the steps below.

**1. Clone the repository**

```bash
git clone https://github.com/your-username/financial-analytics-mini-project.git
cd financial-analytics-mini-project

```

**2. Create and activate a virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

```

**3. Install dependencies**

```bash
pip install -r requirements.txt

```

**4. Initialize NLTK Lexicon**
The application requires the VADER lexicon for sentiment analysis. Run the setup script to download the necessary data:

```bash
python setup.py

```

**5. Run the application**

```bash
streamlit run app.py

```

The application will be accessible at `http://localhost:8501`.

## Theoretical Background

This project applies two distinct analytical approaches:

1. **Time Series Analysis (Quantitative):** The project employs ARIMA (5,1,0), a statistical model that uses time series data to better understand the data set or to predict future trends. The parameters were selected to account for autoregression, non-seasonal differencing (stationarity), and moving averages.
2. **Lexicon-Based Sentiment Analysis (Qualitative):** The VADER system is utilized specifically for its efficacy in processing microblog-like contexts (headlines), detecting polarity (positive/negative) and intensity in text without requiring extensive training datasets.

## Future Roadmap

* **Integration of Vector Databases:** Implementation of semantic search for news archives to identify historical precedents for current market events.
* **Advanced Optimization:** Implementation of Markowitz Mean-Variance Optimization to suggest optimal portfolio weights automatically.
* **Model Expansion:** Incorporation of LSTM (Long Short-Term Memory) neural networks for comparative forecasting against the ARIMA model.

## License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
