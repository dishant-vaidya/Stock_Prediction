# Stock Index Prediction using Ensemble Learning
 
A Streamlit web app that predicts the **next-day closing price** of five major global stock indices using two **ensemble learning models**, **Random Forest (bagging)** and **XGBoost (boosting)**, and evaluates them honestly on data they have never seen, against a naive "tomorrow = today" baseline.
 
> **Disclaimer:** This project is for educational and demonstration purposes only. Stock market predictions are inherently uncertain. Do **not** use this app for real financial, investment, or trading decisions. Do your own research or consult a licensed financial advisor.
 
---
 
## Table of Contents
 
1. [Features](#features)
2. [How it works](#how-it-works)
3. [Project structure](#project-structure)
4. [Installation](#installation)
5. [Running the app](#running-the-app)
6. [Using the dashboard](#using-the-dashboard)
7. [Methodology in detail](#methodology-in-detail)
8. [Models used: bagging vs boosting](#models-used-bagging-vs-boosting)
9. [What was improved from the original version](#what-was-improved-from-the-original-version)
10. [Relationship to the dissertation report](#relationship-to-the-dissertation-report)
11. [Limitations](#limitations)
12. [Troubleshooting](#troubleshooting)
13. [Future improvements](#future-improvements)
14. [Tech stack](#tech-stack)
---
 
## Features
 
- **Five global indices** (data from Yahoo Finance via `yfinance`, last 5 years):
  1. S&P 500 (USA), `^GSPC`
  2. Shanghai Composite (China), `000001.SS`
  3. Nikkei 225 (Japan), `^N225`
  4. DAX (Germany), `^GDAXI`
  5. Nifty 50 (India), `^NSEI`
- **Two ensemble models** representing the two main ensemble families:
  - **Random Forest**: bagging
  - **XGBoost**: boosting
- **Honest evaluation:** chronological hold-out window (configurable, 30-120 trading days) with RMSE, MAE, MAPE, accuracy (100 - MAPE) and directional accuracy.
- **Naive baseline:** both models are compared against "tomorrow's close = today's close". A model that cannot beat it has no real skill.
- **Bagging vs boosting comparison:** both models and the baseline side by side for the selected index.
- **Next-day forecast** with an **80% range** derived from the model's own hold-out errors (not an arbitrary +/-1%).
- **Charts:** 5-year price history, and hold-out actual vs predicted vs naive.
- **Fast reruns:** downloads and training results are cached.
---
 
## How it works
 
```
Yahoo Finance ──► OHLC prices ──► Scale-free features ──► Chronological split
                                                          ├── Train (older days)
                                                          └── Hold-out (latest N days)
                                                                    │
        Random Forest + XGBoost trained on Train ──► predict next-day RETURN on Hold-out
                                                                    │
                          return ──► price ──► RMSE / MAE / MAPE / direction vs naive baseline
                                                                    │
            Refit on ALL data ──► predict tomorrow's return ──► price + 80% range
```
 
---
 
## Project structure
 
```
.
├── Stock_Prediction.py     # Streamlit app (all logic)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── Project_report.pdf      # M.Tech 3rd semester dissertation report (if included)
```
 
---
 
## Installation
 
### Prerequisites
 
- Python **3.9 or newer** (3.10 or 3.11 recommended)
- Internet access (Yahoo Finance data)
### Steps
 
```bash
# 1. Clone the repository
git clone <your-repository-url>
cd <repository-folder>
 
# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
 
# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```
 
---
 
## Running the app
 
```bash
streamlit run Stock_Prediction.py
```
 
Open the local URL shown in the terminal (usually `http://localhost:8501`). The first load downloads data and trains both models, which takes a few seconds. Later interactions are faster thanks to caching.
 
---
 
## Using the dashboard
 
**Sidebar controls**
 
| Control | Purpose |
|---|---|
| Select index | Which of the five indices to analyse |
| Select prediction algorithm | Random Forest (Bagging) or XGBoost (Boosting): whose forecast and chart to display |
| Hold-out window | Number of most recent trading days kept aside for testing (30-120) |
 
**Dashboard sections**
 
1. **Next-Day Forecast:** last close, predicted close for the next business day (with % change), and the 80% range.
2. **Model Comparison: Bagging vs Boosting:** metrics for both models plus the naive baseline, which model had the lower error, and a plain-language verdict on whether the selected model beat the baseline.
3. **Charts:** 5-year price history, and the hold-out chart (actual vs predicted vs naive).
4. **Method and limitations:** short expander summarising the approach.
**Reading the metrics**
 
- **RMSE / MAE:** average error in index points (lower is better).
- **MAPE %:** average percentage error of the predicted price (lower is better).
- **Accuracy % (100 - MAPE):** a simple, readable restatement of MAPE. Because daily moves are small, even the naive baseline scores around 99%, so compare models **against the baseline**, not against 100%.
- **Direction acc. %:** how often the predicted up/down move was correct. Around 50% (or the share of up-days in the window) is no better than guessing.
---
 
## Methodology in detail
 
### 1. Data
- 5 years of daily Open/High/Low/Close prices from Yahoo Finance.
- Handles `yfinance`'s newer MultiIndex columns and removes timezone information.
### 2. Target: next-day return, not raw price
Tree-based models cannot output values outside the range they saw during training. If an index keeps making new highs, a model trained on raw prices is stuck below its historical maximum. Predicting the **next-day return** (`close(t+1) / close(t) - 1`) avoids this, and the return is converted back to a price (`close x (1 + predicted return)`).
 
### 3. Features (all known at the close of day *t*)
 
| Feature | Meaning |
|---|---|
| `ret_0d_ago`, `ret_1d_ago`, `ret_2d_ago`, `ret_4d_ago` | Today's return and returns 1, 2 and 4 days ago |
| `ma5_gap`, `ma10_gap`, `ma20_gap` | Distance of the close from its 5/10/20-day moving average |
| `vol10` | 10-day rolling volatility of returns |
| `range_pct` | Intraday high-low range relative to close |
| `gap_pct` | Overnight gap (open vs previous close) |
| `body_pct` | Open-to-close move |
 
All features are ratios or percentages, so they are comparable across years and index levels.
 
### 4. Evaluation (no look-ahead)
- The data is split **chronologically**: models are trained on older days and tested on the most recent *N* days (no shuffling).
- Each hold-out prediction uses only information available at the previous close.
- Metrics are computed on predicted vs actual prices and compared with the naive baseline.
### 5. Final forecast and range
- Each model is refit on **all** available data to predict tomorrow's return.
- The **80% range** is built from the 10th and 90th percentiles of that model's hold-out errors, added to the predicted return. It reflects how wrong the model has recently been, rather than a fixed percentage.
- The "next day" is the next business day; exchange holidays are not modelled.
---
 
## Models used: bagging vs boosting
 
| | Random Forest (Bagging) | XGBoost (Boosting) |
|---|---|---|
| **Ensemble idea** | Many decision trees trained **independently** on random samples of the data and features; predictions are **averaged** | Trees trained **sequentially**, each one correcting the errors of the previous ones |
| **Main effect** | Reduces **variance** (less overfitting to noise) | Reduces **bias** (learns complex patterns step by step) |
| **Strength** | Robust and stable on noisy data like prices | Often more accurate, with built-in regularisation |
| **Risk** | Can underfit subtle patterns | Can overfit noise if not regularised |
| **Settings used** | 100 trees, minimum leaf size 5 | 100 trees, depth 3, learning rate 0.05 |
 
Both use a fixed random seed (42) for reproducibility. No hyperparameter tuning has been done. Which one wins on the hold-out window varies by index and period, and the app reports it directly.
 
---
 
## What was improved from the original version
 
| # | Issue in original | Improvement |
|---|---|---|
| 1 | "Accuracy" was measured on the last 30 days **that the model had been trained on** (in-sample), so it looked better than reality | Chronological hold-out; models are evaluated on unseen days, then refit for the final forecast |
| 2 | Target was the raw next-day price; tree models cannot extrapolate beyond the training range | Predict next-day return and convert to price |
| 3 | Raw price-level features (Open, High, Low, Close, moving averages) are non-stationary | Scale-free features (returns, ratios, volatility) |
| 4 | Rows with NaN from rolling windows were not removed | Rows with missing features are dropped |
| 5 | `5*250` calendar days is about 3.4 years, not 5 | Uses `period="5y"` |
| 6 | Newer `yfinance` MultiIndex columns could break `data['Close']` | Columns flattened |
| 7 | Prediction range was an arbitrary +/-1% | 80% empirical range from hold-out errors |
| 8 | "Next day" was current date + 1 calendar day (could be a weekend) | Next business day |
| 9 | No baseline to judge model quality | Naive baseline and a verdict message |
| 10 | Data was re-downloaded and models retrained on every click | Caching with `st.cache_data` |
| 11 | Footer HTML had an unclosed tag (`<center>---<center>`) | Tag closed |
| 12 | Unpinned requirements | Minimum versions specified |
 
---
 
## Relationship to the dissertation report
 
This app is the Streamlit successor to the Tkinter application described in the M.Tech 3rd semester report *"Predictive Analytics of Stock Prices Using Ensemble Learning Techniques"* (December 2024). The report covers five ensemble algorithms (Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM); this app focuses on the two that represent the bagging and boosting families. The accuracy figures in the report came from the earlier evaluation setup (raw-price target, date-based features) and are **not directly comparable** with the hold-out metrics shown by this app.
 
---
 
## Limitations
 
- Stock indices are close to a random walk. Expect the naive baseline to be very hard to beat on MAPE, and directional accuracy to hover near 50%.
- Daily features only: no volume, macroeconomic data, news or sentiment.
- A single hold-out window is a noisy estimate. A rolling (walk-forward) back-test would be more reliable.
- No hyperparameter tuning.
- The 80% range is empirical and based on a small sample (the hold-out window).
- Exchange holidays and time zones are not modelled.
- Predictions are for one day ahead only.
---
 
## Troubleshooting
 
| Problem | Solution |
|---|---|
| "Could not download data" | Yahoo Finance may be rate-limiting or temporarily down; wait and retry, and upgrade with `pip install -U yfinance` |
| `ModuleNotFoundError: xgboost` | Run `pip install -r requirements.txt` inside your virtual environment |
| XGBoost fails to import on macOS | Install OpenMP: `brew install libomp` |
| "Not enough price history" | Reduce the hold-out window in the sidebar |
| First run is slow | Both models are trained once per index and window; results are then cached |
| Stale data | Cached data refreshes hourly; use Streamlit's menu (Rerun / Clear cache) to force a refresh |
 
---
 
## Future improvements
 
- Walk-forward (rolling-origin) cross-validation across many windows.
- Hyperparameter tuning (e.g. Optuna) with time-series-aware validation.
- More features: volume, RSI/MACD, volatility indices, macro indicators, news sentiment.
- Additional ensemble members (AdaBoost, Gradient Boosting, LightGBM) and stacking or blending.
- Deep learning baselines (LSTM, temporal CNN) for comparison.
- Exchange-calendar-aware dates and multi-day forecasts with prediction intervals.
- Dockerfile and CI for reproducible deployment.
---
 
## Tech stack
 
Python, Streamlit, yfinance, pandas, NumPy, scikit-learn (Random Forest), XGBoost, Matplotlib.
 
---
 
## Author
 
Dishant Vaidya
 
