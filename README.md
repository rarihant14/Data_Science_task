# Trader Performance vs Market Sentiment Analysis

This project analyzes how **market sentiment (Fear vs Greed)** affects trader behavior and performance using Hyperliquid trading data.

The project includes:

- Data analysis and visualization
- Fear vs Greed comparison
- Predictive model for next-day profitability
- Trader clustering
- Streamlit dashboard


## Installation

Install required libraries:

    pip install -r requirements.txt


## How to Run Notebook

Open Jupyter Notebook:
  
    jupyter notebook

Open:

    trader_sentiment_project.ipynb

Run all cells.


## How to Run Streamlit App

Place files together:

app.py
fear_greed_index.csv
historical_data.csv

Run:

streamlit run app.py


## Methodology

- Loaded and cleaned datasets
- Converted timestamps to datetime
- Merged datasets using date
- Calculated daily metrics (PnL, win rate, trade size)
- Compared Fear vs Greed behavior
- Built predictive model
- Clustered traders into behavior groups


Project Workflow

Output Charts
Long vs Short Trades by Sentiment

Trade Size Comparison

PnL Distribution

## Key Insights

1. Trader profitability differs between Fear and Greed periods.

2. Trade sizes increase during Greed sentiment.

3. Trader direction changes between Fear and Greed.


## Strategy Recommendations

- Fear → Use smaller trade sizes and reduce risk

- Greed → Increase trading activity and follow trends

- Use sentiment as a trading signal to adjust exposure



