# StockCast AI — LSTM-Based Stock Price Forecasting Dashboard

An interactive stock forecasting application built with Python, PyTorch, and Streamlit. It uses an LSTM neural network to generate 30-day stock price forecasts from historical data retrieved through Yahoo Finance.

**Live Demo:** [Launch StockCast AI][(https://stockpredictorproject-cbp2mt2sv92wt7anm7kjqb.streamlit.app/)]

## 1. Project Objective

- Retrieve and preprocess historical stock market data.
- Apply LSTM-based time-series forecasting.
- Generate iterative 30-day price predictions.
- Visualize historical prices, forecast trends, and market statistics through an interactive dashboard.

## 2. Key Features

- Dynamic stock ticker selection.
- Historical data selection (1, 2, 3, or 5 years).
- LSTM-based 30-day forecasting.
- Historical and forecast price charts.
- Daily prediction table and UP/DOWN trend indication.
- Weekly highs, lows, and available market statistics.

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Streamlit | Interactive dashboard |
| PyTorch | LSTM model development |
| yfinance | Historical stock data retrieval |
| Pandas & NumPy | Data processing |
| Scikit-learn | MinMaxScaler normalization |
| Matplotlib | Visualization, if used in the implementation |

## 4. Methodology and Workflow

1. **Data Acquisition:** Fetch historical stock prices using Yahoo Finance.
2. **Data Preprocessing:** Normalize price data using MinMaxScaler.
3. **Sequence Generation:** Create sequences using a 30-day lookback window.
4. **LSTM Model Training:** Train the neural network to learn temporal patterns.
5. **Iterative Forecasting:** Predict the next price and feed predictions back into the sequence to generate a 30-day forecast.
6. **Inverse Transformation:** Convert normalized predictions back to the original price scale.
7. **Visualization:** Display historical prices, forecast trends, and market statistics in Streamlit.

## 5. Project Structure

```text
StockCast-AI/
├── app.py                 # Streamlit dashboard entry point
├── model.py               # LSTM model and forecasting pipeline
├── utils.py               # Market data helpers and chart layout
├── assets/                # Static application assets
├── data/
│   └── raw/               # Optional local data; prices are fetched live
├── models/                # Saved model artifacts, if added later
├── notebooks/             # Exploratory analysis notebooks
├── reports/
│   └── figures/           # Exported charts and figures
├── tests/                 # Automated tests
├── requirements.txt
├── .gitignore
└── README.md
```

## 6. Installation and Setup

```bash
git clone <your-repository-url>
cd StockCast-AI
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## 7. How to Use

1. Open the live application.
2. Enter a stock ticker, such as `AAPL`, `MSFT`, or `TCS.NS`.
3. Select the historical data period.
4. Run the prediction workflow.
5. Explore the forecast chart, predicted values, and market statistics.

## 8. Disclaimer

This project is for educational purposes only and does not provide financial advice. Predictions are not guaranteed to be accurate and should not be used as the sole basis for investment decisions.

## Author

**Anurag Tiwari**
