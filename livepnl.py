from flask import Flask, send_file, render_template_string
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# TODO: put your raw CSV URL here
TRADES_URL = "https://raw.githubusercontent.com/orhansozgur/freemoneybot/main/open_trades.csv"

# Hard-coded S&P 500 start date (as you asked)
SPX_ANCHOR_DATE = datetime(2025, 11, 21)


def load_trades() -> pd.DataFrame:
    df = pd.read_csv(TRADES_URL)
    # Ensure expected columns
    expected = {"Ticker", "entry_date", "entry_price", "leverage", "alloc", "peak_price"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    return df


from typing import Optional
def get_price_history(tickers, start_date: datetime, end_date: Optional[datetime] = None) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.today() + timedelta(days=1)

    data = yf.download(
        tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1m",          # minutely data
        auto_adjust=True,
        progress=False,
    )

    if isinstance(data, pd.DataFrame) and "Close" in data.columns:
        close = data["Close"]
    else:
        close = data

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers if isinstance(tickers, str) else tickers[0])


    return close
def get_spx_series(start_date: datetime) -> pd.Series:
    """
    Get S&P 500 prices starting from start_date.
    Try 1-minute data first; if it's empty, fall back to daily.
    Returns a Series indexed by datetime.
    """
    # first try 1-minute (same as your strategy)
    spx_df = get_price_history("^GSPC", start_date=start_date)

    if spx_df is not None and not spx_df.empty:
        # yfinance with our get_price_history puts '^GSPC' as a column name
        if "^GSPC" in spx_df.columns:
            spx = spx_df["^GSPC"]
        else:
            # just take the first column as a fallback
            spx = spx_df.iloc[:, 0]
        return spx

    # fallback: daily data (always available)
    daily = yf.download(
        "^GSPC",
        start=start_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if isinstance(daily, pd.DataFrame) and "Close" in daily.columns:
        spx = daily["Close"]
    else:
        # last fallback: just make a flat dummy series so code doesn't crash
        idx = pd.date_range(start=start_date, end=datetime.today(), freq="D")
        spx = pd.Series(5000.0, index=idx)

    return spx

def compute_portfolio_pnl(trades: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """
    PnL(t) = sum_over_trades( qty * (price_t - entry_price) )
    qty = (leverage * alloc) / entry_price
    PnL is 0 before each trade's entry_date.
    """
    pnl_total = None

    for _, row in trades.iterrows():
        ticker = row["Ticker"]
        entry_price = float(row["entry_price"])
        leverage = float(row["leverage"])
        alloc = float(row["alloc"])
        entry_date = row["entry_date"]

        if ticker not in prices.columns:
            continue

        qty = (leverage * alloc) / entry_price
        px_series = prices[ticker]

        # raw PnL
        trade_pnl = qty * (px_series - entry_price)
        # zero before entry date
        trade_pnl = trade_pnl.where(trade_pnl.index >= entry_date, 0.0)

        if pnl_total is None:
            pnl_total = trade_pnl
        else:
            pnl_total = pnl_total.add(trade_pnl, fill_value=0.0)

    if pnl_total is None:
        raise ValueError("No valid tickers in price data to compute PnL.")

    return pnl_total



def make_plot_image() -> bytes:
    trades = load_trades()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], utc=True)

    # fixed start date that never moves
    start_date = SPX_ANCHOR_DATE

    # price history for all trade tickers from fixed start date
    tickers = trades["Ticker"].unique().tolist()
    prices = get_price_history(tickers, start_date=start_date)

    # strategy PnL
    portfolio_pnl = compute_portfolio_pnl(trades, prices)

    # ------- Benchmark: S&P 500 PnL with same capital --------
    # 1) get S&P prices from same fixed start date
    spx = get_spx_series(start_date)

    # 2) align S&P prices to the same index as strategy PnL (forward fill)
    spx = spx.reindex(portfolio_pnl.index, method="ffill")

    # 3) pretend we invest exactly 10M into S&P at start_date
    initial_capital = 10_000_000.0  # 10M benchmark
    spx_start = spx.iloc[0]
    spx_qty = initial_capital / spx_start

    # 4) benchmark PnL over time
    spx_pnl = spx_qty * (spx - spx_start)
    

    # --------------- Plot both PnLs ----------------
    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.plot(portfolio_pnl.index, portfolio_pnl.values, label="Strategy PnL")
    ax1.plot(spx_pnl.index, spx_pnl.values, linestyle="--",
             label=f"S&P 500 PnL")

    ax1.set_ylabel("PnL")
    ax1.set_xlabel("Time")
    ax1.set_title("Strategy PnL vs S&P 500 PnL")
    ax1.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    plt.close(fig)

    return buf.getvalue()



# ---- ROUTES ----

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Live PnL vs Flat S&P 500</title>
  <!-- Auto-refresh every 60 seconds -->
  <meta http-equiv="refresh" content="60">
  <style>
    body { font-family: sans-serif; text-align: center; background: #111; color: #eee; }
    img { max-width: 95vw; height: auto; border: 1px solid #444; margin-top: 20px; }
  </style>
</head>
<body>
  <h1>Live PnL vs Flat S&P 500</h1>
  <p>Auto-refreshes every 60 seconds.</p>
  <img src="/plot.png?ts={{ timestamp }}" alt="PnL Plot">
</body>
</html>
"""


@app.route("/")
def index():
    # timestamp query param busts browser cache so it really reloads
    ts = datetime.utcnow().timestamp()
    return render_template_string(HTML_TEMPLATE, timestamp=ts)


@app.route("/plot.png")
def plot_png():
    img_bytes = make_plot_image()
    return send_file(io.BytesIO(img_bytes), mimetype="image/png")
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT
    app.run(host="0.0.0.0", port=port, debug=False)
