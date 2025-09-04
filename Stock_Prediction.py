import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error

# Stock Index or Symbol

index_map = {
    "S&P 500": "^GSPC",
    "Shanghai Composite": "000001.SS",
    "Nikkei 225": "^N225",
    "Deutscher Aktienindex": "^GDAXI",
    "Nifty 50": "^NSEI"
}

# Download 5 Years of Stock Data

def fetch_data(index_symbol):
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=5*250)
    data = yf.download(index_symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    return data

# Data Preprocessing 

def preprocess_data(data):

    # Feature engineering

    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Return'].rolling(window=10).std()

    # Predictors

    X = data[['Open', 'High', 'Low', 'Close', 'MA5', 'MA10', 'Volatility']]

    # Target: Next day's closing price
    
    y = data['Close'].shift(-1).dropna()

    return X[:-1], y

# Prediction

def predict(index_name, model_name):
    index_symbol = index_map[index_name]
    data = fetch_data(index_symbol)
    X, y = preprocess_data(data)

    # Model Selection 

    if model_name == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == 'XGBoost':
        model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)

    # Train on all data

    model.fit(X, y)

    # Predict tomorrow using last available features

    last_features = X.iloc[[-1]]
    predicted_price = model.predict(last_features)[0]
    predicted_high = predicted_price * 1.01
    predicted_low = predicted_price * 0.99
    next_day = data.index[-1] + pd.Timedelta(days=1)

    # Evaluate recent performance (last 30 days)

    y_pred = model.predict(X[-30:])
    recent_actual = y[-30:]
    mape = mean_absolute_percentage_error(recent_actual, y_pred) * 100
    accuracy = 100 - mape

    return predicted_price, predicted_high, predicted_low, data, next_day, recent_actual, y_pred, accuracy

# Streamlit UI

st.set_page_config(page_title="Stock Index Prediction", layout="centered")
st.title("Stock Index Prediction Using Ensemble Learning Models")

# Dropdowns

index_name = st.selectbox("Select Index:", list(index_map.keys()), index=0)
model_name = st.selectbox(
    "Select Prediction Algorithm:",
    ["Random Forest", "XGBoost"],
    index=0
)

# Prediction button

if st.button("Predict"):
    with st.spinner("Training model and forecasting next day..."):
        predicted_price, predicted_high, predicted_low, data, next_day, recent_actual, y_pred, accuracy = predict(index_name, model_name)

    # Chart 1: Historical Prices

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(data.index, data['Close'], label="Actual Prices", color='blue')
    ax1.set_title(f"{index_name} - Historic Prices")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.legend()
    st.pyplot(fig1)

    # Chart 2: Actual vs Predicted (last 30 days)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(recent_actual.index, recent_actual.values, label="Actual Closing Price", color="blue")
    ax2.plot(recent_actual.index, y_pred, label="Predicted Closing Price", color="orange")
    ax2.set_title(f"{index_name} - Last 30 Days: Actual vs Predicted")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    ax2.legend()
    st.pyplot(fig2)

    # Display results

    st.success(f"Next Day Prediction ({next_day.strftime('%Y-%m-%d')})")
    st.metric("Predicted Closing Price", f"{predicted_price:.2f}")
    st.info(f"Range: {predicted_low:.2f} - {predicted_high:.2f}")
    st.write(f"Recent Model Accuracy (last 30 days): **{accuracy:.2f}%**")

st.markdown("<center>---<center>", unsafe_allow_html=True)
st.markdown("<center>Dishant Vaidya</center>", unsafe_allow_html=True)
st.markdown("<center><small>vaidya.dishant@gmail.com</small></center>", unsafe_allow_html=True)