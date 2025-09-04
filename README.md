# Stock Index Prediction using Ensemble Learning

This project is a **Streamlit web app** that predicts the **next-day closing price** of major stock indices using **Random Forest** and **XGBoost** models.  
It downloads **5 years of historical data** via [Yahoo Finance](https://pypi.org/project/yfinance/), performs feature engineering, trains the models, and forecasts the next day’s closing price with accuracy evaluation.

---

## Features

- Select from **global stock indices**:

  1. S&P 500 (USA)
  2. Shanghai Composite (China)
  3. Nikkei 225 (Japan)
  4. DAX (Germany)
  5. Nifty 50 (India)

- Choose a prediction model: **Random Forest** or **XGBoost**

- Visualizations:

  1. Historical price trends
  2. Actual vs Predicted prices (last 30 days)

- Accuracy metric (**MAPE**-based)

- Predicted next-day **closing price with range (±1%)**

---

## Disclaimer

This project is built **for educational and demonstration purposes only**.  
Stock market predictions are inherently uncertain and this app should **not be used for actual financial, investment, or trading decisions**.  

Always perform your **own research and due diligence** or consult a licensed financial advisor before making any investment choices.

