from flask import Flask, send_file, render_template_string
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta

app = Flask(__name__)

# TODO: put your raw CSV URL here
TRADES_URL = "https://raw.githubusercontent.com/orhansozgur/freemoneybot/refs/heads/main/open_trades.csv?token=GHSAT0AAAAAADO44WJ7JCEYDJKZJWVVSRJG2JI3OPQ"

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


def get_price_history(tickers, start_date: datetime, end_date: datetime | None = None) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.today() + timedelta(days=1)

    data = yf.download(
        tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False
    )

    # yfinance shape handling
    if isinstance(data, pd.DataFrame) and "Close" in data.columns:
        close = data["Close"]
    else:
        close = data

    # If single ticker, may be a Series
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers if isinstance(tickers, str) else tickers[0])

    return close


def compute_portfolio_pnl(trades: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """
    PnL(t) = sum_over_trades( qty * (price_t - entry_price) )
    where qty is derived from leverage and alloc:
        qty = (leverage * alloc) / entry_price
    Assumes all trades are long.
    """
    pnl_total = None

    for _, row in trades.iterrows():
        ticker = row["Ticker"]
        entry_price = float(row["entry_price"])
        leverage = float(row["leverage"])
        alloc = float(row["alloc"])

        if ticker not in prices.columns:
            continue

        qty = (leverage * alloc) / entry_price  # number of index units
        px_series = prices[ticker]

        trade_pnl = qty * (px_series - entry_price)

        if pnl_total is None:
            pnl_total = trade_pnl
        else:
            pnl_total = pnl_total.add(trade_pnl, fill_value=0.0)

    if pnl_total is None:
        raise ValueError("No valid tickers in price data to compute PnL.")

    return pnl_total


def make_plot_image() -> bytes:
    trades = load_trades()

    # Parse entry dates and find earliest
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    earliest_entry = trades["entry_date"].min()

    # Start date is the later of earliest entry and SPX anchor date (you wanted SPX from 2025-11-21)
    start_date = max(earliest_entry, SPX_ANCHOR_DATE)

    # Get price history for all trade tickers
    tickers = trades["Ticker"].unique().tolist()
    prices = get_price_history(tickers, start_date=start_date)

    # Compute portfolio PnL over time
    portfolio_pnl = compute_portfolio_pnl(trades, prices)

    # ---- S&P 500 flat line ----
    # Get SPX level on SPX_ANCHOR_DATE and keep it constant (no movement)
    # If data before 2025-11-21 exists, we still anchor on that date.
    spx_prices = get_price_history("^GSPC", start_date=SPX_ANCHOR_DATE)
    if spx_prices.shape[0] == 0:
        # Fallback: just pick an arbitrary level if no data
        spx_level0 = 5000.0
    else:
        spx_level0 = spx_prices.iloc[0, 0]

    # Create a flat SPX series over the same index as portfolio PnL
    spx_flat = pd.Series(spx_level0, index=portfolio_pnl.index)

    # ---- Plot ----
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Portfolio PnL
    ax1.plot(portfolio_pnl.index, portfolio_pnl.values, label="Portfolio PnL")
    ax1.set_ylabel("PnL (currency)")
    ax1.set_xlabel("Date")

    # Flat SPX line on secondary axis or same, your choice
    ax1.plot(spx_flat.index, spx_flat.values, linestyle="--", label=f"S&P 500 (flat from {SPX_ANCHOR_DATE.date()})")

    ax1.set_title("Portfolio PnL vs Flat S&P 500")
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
    # Run on localhost:5000
    app.run(host="127.0.0.1", port=5000, debug=False)
