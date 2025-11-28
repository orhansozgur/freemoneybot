from flask import Flask, jsonify, render_template_string
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional
import os

app = Flask(__name__)

# Your CSV on GitHub (this stays the same)
TRADES_URL = "https://raw.githubusercontent.com/orhansozgur/freemoneybot/main/open_trades.csv"

# Fixed anchor date if you want a stable history start
SPX_ANCHOR_DATE = datetime(2025, 11, 21)


def load_trades() -> pd.DataFrame:
    df = pd.read_csv(TRADES_URL)
    expected = {"Ticker", "entry_date", "entry_price", "leverage", "alloc", "peak_price"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    return df


def get_price_history(
    tickers,
    start_date: datetime,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.today() + timedelta(days=1)

    data = yf.download(
        tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1m",          # minutely data → chart moves every minute
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
        entry_date = row["entry_date"]  # already a UTC Timestamp

        if ticker not in prices.columns:
            continue

        qty = (leverage * alloc) / entry_price
        px_series = prices[ticker]

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


def compute_pnl_series() -> pd.Series:
    """Helper: load trades, load prices, return cleaned PnL series."""
    trades = load_trades()
    # make entry_date timezone-aware UTC to match yfinance
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], utc=True)

    # fixed start date that never moves
    start_date = SPX_ANCHOR_DATE

    tickers = trades["Ticker"].unique().tolist()
    prices = get_price_history(tickers, start_date=start_date)

    portfolio_pnl = compute_portfolio_pnl(trades, prices)
    # drop NaN rows, sort index
    portfolio_pnl = portfolio_pnl.dropna().sort_index()
    return portfolio_pnl


# ---------------- HTML (Chart.js front-end) ----------------

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Live Strategy PnL</title>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      text-align: center;
      background: #050505;
      color: #f5f5f5;
      margin: 0;
      padding: 20px;
    }
    #container {
      max-width: 1100px;
      margin: 0 auto;
    }
    canvas {
      width: 100%;
      max-height: 540px;
    }
    #status {
      margin-top: 8px;
      font-size: 0.9rem;
      color: #aaaaaa;
    }
  </style>
  <!-- Chart.js + time adapter for nice time axis -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/luxon@^3"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@^1"></script>
</head>
<body>
  <div id="container">
    <h1>Live Strategy PnL</h1>
    <p>Hover to see time &amp; PnL. Updates automatically.</p>
    <canvas id="pnlChart"></canvas>
    <div id="status">Loading…</div>
  </div>

  <script>
    let chart = null;

    async function fetchPnl() {
      const res = await fetch('/data.json?ts=' + Date.now());
      if (!res.ok) {
        throw new Error('HTTP ' + res.status);
      }
      return res.json();
    }

    async function updateChart() {
      try {
        const data = await fetchPnl();
        const labels = data.timestamps;  // ISO strings
        const pnl = data.pnl;

        const ctx = document.getElementById('pnlChart').getContext('2d');

        if (!chart) {
          chart = new Chart(ctx, {
            type: 'line',
            data: {
              labels: labels,
              datasets: [{
                label: 'Strategy PnL',
                data: pnl,
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.1,
              }]
            },
            options: {
              responsive: true,
              interaction: {
                mode: 'index',
                intersect: false,
              },
              plugins: {
                legend: {
                  display: true,
                },
                tooltip: {
                  callbacks: {
                    // Show "PnL: £x.xx" on hover
                    label: function(ctx) {
                      const v = ctx.parsed.y;
                      if (v == null) return '';
                      return 'PnL: ' + v.toFixed(2);
                    }
                  }
                },
              },
              scales: {
                x: {
                  type: 'time',
                  time: {
                    unit: 'minute',
                    displayFormats: {
                      minute: 'HH:mm',
                      hour: 'HH:mm',
                    }
                  },
                  ticks: {
                    maxRotation: 0,
                  }
                },
                y: {
                  ticks: {
                    callback: function(value) {
                      return value.toLocaleString();
                    }
                  }
                }
              }
            }
          });
        } else {
          chart.data.labels = labels;
          chart.data.datasets[0].data = pnl;
          chart.update();
        }

        document.getElementById('status').textContent =
          'Last update: ' + new Date().toLocaleTimeString();
      } catch (e) {
        document.getElementById('status').textContent = 'Error: ' + e.message;
      }
    }

    // Initial load
    updateChart();
    // Refresh every 60 seconds (tweak if you want more/less)
    setInterval(updateChart, 60000);
  </script>
</body>
</html>
"""


# ---------------- ROUTES ----------------

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/data.json")
def data_json():
    pnl = compute_pnl_series()
    return jsonify({
        "timestamps": [ts.isoformat() for ts in pnl.index],
        "pnl": pnl.tolist(),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
