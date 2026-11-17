
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Stock Index Prediction", layout="wide")
st.title("Stock Index Prediction Using Ensemble Learning Models")
st.caption("Educational project. Not financial advice.")

INDEX_MAP = {
    "S&P 500": "^GSPC",
    "Shanghai Composite": "000001.SS",
    "Nikkei 225": "^N225",
    "Deutscher Aktienindex (DAX)": "^GDAXI",
    "Nifty 50": "^NSEI",
}

RF_NAME = "Random Forest (Bagging)"
XGB_NAME = "XGBoost (Boosting)"
MODEL_NAMES = [RF_NAME, XGB_NAME]


def build_model(name: str):
    """Return a fresh, untrained model (100 estimators each, fixed seed)."""
    if name == RF_NAME:
        return RandomForestRegressor(
            n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1
        )
    if name == XGB_NAME:
        return XGBRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            random_state=42, verbosity=0,
        )
    raise ValueError(f"Unknown model: {name}")


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
st.sidebar.header("Settings")
index_name = st.sidebar.selectbox("Select index", list(INDEX_MAP.keys()), index=0)
model_name = st.sidebar.selectbox("Select prediction algorithm", MODEL_NAMES, index=0)
HOLDOUT = st.sidebar.slider(
    "Hold-out window (trading days)", 30, 120, 60, 10,
    help="The most recent days kept aside to test the models on data they never saw.",
)


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Downloading price data...")
def load_data(symbol: str) -> pd.DataFrame:
    raw = yf.download(symbol, period="5y", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame()

    # Newer yfinance returns MultiIndex columns like ("Close", "^GSPC")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close"]].dropna().copy()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    return df


def make_dataset(df: pd.DataFrame):
    """
    Build scale-free features known at the close of day t and the target:
    the return from day t to day t+1.
    """
    close = df["Close"]
    ret = close.pct_change()

    feats = pd.DataFrame(index=df.index)
    for k in (0, 1, 2, 4):  # today's return and returns 1, 2, 4 days ago
        feats[f"ret_{k}d_ago"] = ret.shift(k)
    for w in (5, 10, 20):  # distance of price from its moving average
        feats[f"ma{w}_gap"] = close / close.rolling(w).mean() - 1
    feats["vol10"] = ret.rolling(10).std()                      # recent volatility
    feats["range_pct"] = (df["High"] - df["Low"]) / close        # intraday range
    feats["gap_pct"] = df["Open"] / close.shift(1) - 1           # overnight gap
    feats["body_pct"] = close / df["Open"] - 1                   # open-to-close move
    feats = feats.replace([np.inf, -np.inf], np.nan)

    target = close.shift(-1) / close - 1                         # next-day return
    return feats, target


def score(actual, pred, true_ret=None, pred_ret=None) -> dict:
    err = actual - pred
    mape = float((err.abs() / actual).mean() * 100)
    out = {
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAE": float(err.abs().mean()),
        "MAPE %": mape,
        "Accuracy % (100 - MAPE)": 100 - mape,
        "Direction acc. %": np.nan,
    }
    if true_ret is not None and pred_ret is not None:
        out["Direction acc. %"] = float(
            (np.sign(true_ret.values) == np.sign(pred_ret.values)).mean() * 100
        )
    return out


# ----------------------------------------------------------------------------
# TRAINING / EVALUATION
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Training models and evaluating on the hold-out window...")
def run_analysis(df: pd.DataFrame, holdout: int) -> dict:
    feats, target = make_dataset(df)
    latest_X = feats.dropna().iloc[[-1]]               # features of the last trading day
    data = feats.join(target.rename("target")).dropna()
    X, y = data.drop(columns="target"), data["target"]

    # Chronological split: no shuffling, test days come strictly after training days
    X_tr, X_te = X.iloc[:-holdout], X.iloc[-holdout:]
    y_tr, y_te = y.iloc[:-holdout], y.iloc[-holdout:]

    close_vals = df["Close"].values
    pos = df.index.get_indexer(X_te.index)
    next_dates = df.index[pos + 1]                     # the day each prediction is FOR
    base = pd.Series(close_vals[pos], index=next_dates)             # close on day t
    actual_next = pd.Series(close_vals[pos + 1], index=next_dates)  # close on day t+1

    last_close = float(df["Close"].iloc[-1])
    last_date = df.index[-1]

    results = {}
    for name in MODEL_NAMES:
        # 1) Evaluate on unseen days
        model = build_model(name)
        model.fit(X_tr, y_tr)
        pred_ret = pd.Series(model.predict(X_te), index=X_te.index)
        pred_next = pd.Series(base.values * (1 + pred_ret.values), index=next_dates)
        metrics = score(actual_next, pred_next, y_te, pred_ret)

        # 80% interval from the model's own hold-out errors (in return space)
        resid = (y_te - pred_ret).values
        q_lo, q_hi = np.quantile(resid, [0.10, 0.90])

        # 2) Refit on ALL data and forecast tomorrow
        final = build_model(name)
        final.fit(X, y)
        next_ret = float(final.predict(latest_X)[0])

        results[name] = {
            "pred_next": pred_next,
            "metrics": metrics,
            "forecast": last_close * (1 + next_ret),
            "low": last_close * (1 + next_ret + q_lo),
            "high": last_close * (1 + next_ret + q_hi),
        }

    return {
        "models": results,
        "naive": score(actual_next, base),            # naive: tomorrow = today
        "naive_next": base,
        "actual_next": actual_next,
        "last_close": last_close,
        "last_date": last_date,
        "train_end": X_tr.index[-1],
        "up_days_pct": float((y_te > 0).mean() * 100),
    }


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
df = load_data(INDEX_MAP[index_name])
if df.empty:
    st.error(f"Could not download data for {index_name}. Please try again later.")
    st.stop()
if len(df) < HOLDOUT * 4 + 60:
    st.error("Not enough price history for this hold-out window. Reduce it in the sidebar.")
    st.stop()

analysis = run_analysis(df, HOLDOUT)
sel = analysis["models"][model_name]
last_close = analysis["last_close"]
next_day = analysis["last_date"] + pd.offsets.BDay(1)   # next business day (holidays ignored)

# ---- Section 1: Next-day forecast ------------------------------------------
st.header(f"1. Next-Day Forecast: {index_name} ({model_name})")
c1, c2, c3 = st.columns(3)
c1.metric(f"Last close ({analysis['last_date'].date()})", f"{last_close:,.2f}")
c2.metric(
    f"Predicted close ({next_day.date()})",
    f"{sel['forecast']:,.2f}",
    delta=f"{(sel['forecast'] / last_close - 1) * 100:+.2f}%",
)
c3.metric("80% range", f"{sel['low']:,.2f} - {sel['high']:,.2f}")
st.caption(
    "The range comes from this model's errors on the hold-out window (10th to 90th percentile), "
    "not from a fixed percentage. Next trading day assumes no exchange holiday."
)

# ---- Section 2: Hold-out evaluation ----------------------------------------
st.header("2. Model Comparison: Bagging vs Boosting (out-of-sample hold-out)")
st.write(
    f"Models were trained on data up to **{analysis['train_end'].date()}** and tested on the "
    f"following **{HOLDOUT} trading days**, which they never saw. Each day's prediction uses only "
    "information available at the previous close. The naive baseline simply predicts that tomorrow's "
    "close equals today's close."
)

rows = {name: r["metrics"] for name, r in analysis["models"].items()}
rows["Naive (previous close)"] = analysis["naive"]
table = pd.DataFrame(rows).T
st.dataframe(table.style.format("{:.2f}", na_rep="-"))

mape_rf = analysis["models"][RF_NAME]["metrics"]["MAPE %"]
mape_xgb = analysis["models"][XGB_NAME]["metrics"]["MAPE %"]
better = RF_NAME if mape_rf < mape_xgb else XGB_NAME
st.caption(
    f"Bagging vs boosting on this window: {better} had the lower MAPE "
    f"(Random Forest {mape_rf:.2f}% vs XGBoost {mape_xgb:.2f}%)."
)

sel_mape = sel["metrics"]["MAPE %"]
naive_mape = analysis["naive"]["MAPE %"]
if sel_mape < naive_mape:
    st.success(
        f"{model_name} (MAPE {sel_mape:.2f}%) beat the naive baseline "
        f"(MAPE {naive_mape:.2f}%) on this hold-out window."
    )
else:
    st.warning(
        f"{model_name} (MAPE {sel_mape:.2f}%) did not beat the naive baseline "
        f"(MAPE {naive_mape:.2f}%) on this hold-out window. This is common for stock indices: "
        "day-to-day moves are small and very hard to predict."
    )
st.caption(
    f"The index closed higher on {analysis['up_days_pct']:.0f}% of days in this window. "
    "Direction accuracy close to 50% (or to that share) is no better than guessing."
)

# ---- Section 3: Charts ------------------------------------------------------
st.header("3. Charts")

fig1, ax1 = plt.subplots(figsize=(11, 4.5))
ax1.plot(df.index, df["Close"], color="blue", label="Close")
ax1.set_title(f"{index_name} - Historic Prices (5 years)")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")
ax1.legend()
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(11, 4.5))
ax2.plot(analysis["actual_next"].index, analysis["actual_next"].values,
         color="blue", label="Actual close")
ax2.plot(sel["pred_next"].index, sel["pred_next"].values,
         color="orange", linestyle="--", label=f"{model_name} prediction")
ax2.plot(analysis["naive_next"].index, analysis["naive_next"].values,
         color="grey", linestyle=":", label="Naive (previous close)")
ax2.set_title(f"{index_name} - Hold-out: Actual vs Predicted Next-Day Close")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.legend()
fig2.autofmt_xdate()
st.pyplot(fig2)

# ---- Method note ------------------------------------------------------------
with st.expander("Method and limitations"):
    st.markdown(
        """
- **Models:** Random Forest (bagging) and XGBoost (boosting), 100 trees each.
- **Target:** next-day return, converted back to a price (`close x (1 + predicted return)`).
- **Features:** today's and recent returns, gaps to 5/10/20-day moving averages, 10-day volatility,
  intraday range, overnight gap and candle body, all scale-free.
- **Evaluation:** chronological hold-out, compared with a naive baseline. The final forecast is made
  by a model refit on all available data.
- Markets are close to a random walk. A model that cannot beat the naive baseline has no real
  predictive edge, and even one that does may not survive trading costs.
- Exchange holidays are not modelled when choosing the next trading day.
        """
    )

# ---- Footer -----------------------------------------------------------------
st.markdown("<center>---</center>", unsafe_allow_html=True)
st.markdown("<center>Dishant Vaidya</center>", unsafe_allow_html=True)
st.markdown("<center><small>+91-7507790128</small></center>", unsafe_allow_html=True)
st.markdown("<center><small>vaidya.dishant@gmail.com</small></center>", unsafe_allow_html=True)
