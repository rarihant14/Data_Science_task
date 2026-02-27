# Predict Next-Day Trader Profitability using Sentiment + Behavior

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


st.title("Next-Day Trader Profitability Prediction")

st.write("Predict next-day profitability using sentiment and trading behavior")


# Load datasets
sentiment = pd.read_csv("fear_greed_index.csv")

traders = pd.read_csv("historical_data.csv")


# Convert dates
sentiment['date'] = pd.to_datetime(sentiment['date'])
sentiment['date'] = sentiment['date'].dt.date


traders['Timestamp IST'] = pd.to_datetime(
    traders['Timestamp IST'],
    dayfirst=True
)

traders['date'] = traders['Timestamp IST'].dt.date


# Merge datasets
merged = pd.merge(traders, sentiment, on='date')


# Create win column
merged['win'] = merged['Closed PnL'] > 0


# Convert columns to numeric
merged['Closed PnL'] = pd.to_numeric(
    merged['Closed PnL'],
    errors='coerce'
)

merged['Size USD'] = pd.to_numeric(
    merged['Size USD'],
    errors='coerce'
)

merged['value'] = pd.to_numeric(
    merged['value'],
    errors='coerce'
)


st.header("Daily Features")


# Create daily dataset
daily = merged.groupby('date').agg({

    'Closed PnL':'sum',
    'Size USD':'mean',
    'win':'mean',
    'value':'mean'

}).reset_index()


st.write("Sample Daily Data")

st.dataframe(daily.head())


# Create next day profit column
daily['next_profit'] = daily['Closed PnL'].shift(-1)


daily['target'] = daily['next_profit'] > 0


daily = daily.dropna()


# Features
X = daily[['Size USD','win','value']]

# Target
y = daily['target']


# Split data
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
model = LogisticRegression()

model.fit(X_train,y_train)


# Predictions
pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test,pred)


st.header("Model Accuracy")

st.write(accuracy)


# Prediction Section
st.header("Try Prediction")


size_input = st.number_input(
    "Average Trade Size",
    value=500
)

win_input = st.slider(
    "Win Rate",
    0.0,
    1.0,
    0.5
)

sentiment_input = st.slider(
    "Sentiment Value",
    0,
    100,
    50
)


if st.button("Predict Next Day Profit"):

    new_data = np.array([[

        size_input,
        win_input,
        sentiment_input

    ]])

    prediction = model.predict(new_data)

    
    if prediction[0] == True:
        st.success("Next Day Likely Profitable")

    else:
        st.error("Next Day Likely Loss")