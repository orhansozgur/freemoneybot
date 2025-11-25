import yfinance as yf
import polars as pl
import numpy as np
import datetime
import warnings
import pandas as pd
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# ---------------- EMAIL SETTINGS ----------------
EMAIL_SENDER = "orhansozgur@gmail.com"
EMAIL_PASSWORD = os.getenv("GMAIL_APP_PASS")
EMAIL_RECEIVERS = ["orhansozgur@gmail.com", "eminozgur@gmail.com", "pasagokdemir3103@gmail.com"]
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

if not EMAIL_PASSWORD:
    raise RuntimeError(
        "Missing email password. Set GMAIL_APP_PASS env var, store in keyring, or use dotenv."
    )
    
def send_email(subject, html_body):
    """Send an HTML email via Gmail SMTP."""
    msg = MIMEMultipart("alternative")
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_SENDER
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVERS, msg.as_string())
        print("📧 Email sent successfully.")
    except Exception as e:
        print(f"⚠️ Failed to send email: {e}")

def df_to_html(df: pl.DataFrame, title: str = "") -> str:
    """Convert a Polars DataFrame to a styled HTML table."""
    if df.height == 0:
        return f"<p><strong>{title}</strong><br><em>No data</em></p>"

    html_table = df.to_pandas().to_html(index=False, border=0, classes="styled-table")
    return f"<h3>{title}</h3>{html_table}"
HTML_STYLE = """
<style>
body {
  font-family: Arial, sans-serif; color:#222; line-height:1.5; background-color:#f8fbff;
}
h2,h3 { color:#004085; }
.table-container { margin-top: 10px; margin-bottom: 25px; }
.styled-table {
  border-collapse: collapse;
  width: 100%;
  font-size: 0.9em;
  background-color: #eaf4ff; /* light blue background */
}
.styled-table th, .styled-table td {
  padding: 8px 10px;
  border: 1px solid #cfd8e3;
  text-align: left;
}
.styled-table th {
  background-color: #d9ecff;
  font-weight: bold;
}
.up { color: green; font-weight: bold; }
.down { color: red; font-weight: bold; }
a { color: #004085; text-decoration: none; }
a:hover { text-decoration: underline; }
</style>
"""
def make_market_html(market_summary, params):
    df = pl.DataFrame(market_summary).sort("Drawdown")
    rows = []
    for row in df.iter_rows(named=True):
        sym = row["Ticker"]
        close = row["Close"]
        peak = row["Rolling Peak"]
        dd = row["Drawdown"]           # already %
        volr = row["Vol Ratio"]
        thr = params[sym][0] * 100     # drop_threshold in %
        diff_from_peak = dd            # same number, % below peak
        diff_from_thr = dd - thr
        diff_color = "green" if diff_from_thr <0 else "red"
        link = f'<a href="https://finance.yahoo.com/quote/{sym}" target="_blank">{sym}</a>'
        rows.append(f"""
          <tr>
            <td>{link}</td>
            <td>{close:,.2f}</td>
            <td>{peak:,.2f}</td>
            <td>{diff_from_peak:.2f}%</td>
            <td style="color:{diff_color};">{diff_from_thr:.2f}% from thr</td>
            <td>{row["Current Vol"]:.5f}</td>
            <td>{row["Mean Vol"]:.5f}</td>
            <td>{volr:.2f}</td>
          </tr>""")

    return f"""
    <div class="table-container">
    <h3>Market Summary</h3>
    <table class="styled-table">
      <tr>
        <th>Ticker</th><th>Close</th><th>Rolling Peak</th><th>% from Peak</th>
        <th>% from Threshold</th><th>Current Vol</th><th>Mean Vol</th><th>Vol Ratio</th>
      </tr>
      {''.join(rows)}
    </table>
    </div>
    """

def make_open_trades_html(open_trades):
    if open_trades.height == 0:
        return "<p><strong>No open trades.</strong></p>"

    rows = []
    for t in open_trades.iter_rows(named=True):
        sym = t["Ticker"]
        link = f'<a href="https://finance.yahoo.com/quote/{sym}" target="_blank">{sym}</a>'
        entry = t["entry_price"]
        peak = t["peak_price"]
        alloc = t["alloc"]
        lev = t["leverage"]

        # get current price if available
        try:
            data = yf.download(sym, period="1d", interval="1h", progress=False)
            cur_price = data["Close"].iloc[-1]
            if hasattr(cur_price, "item"):
                cur_price = cur_price.item()
            else:
                cur_price = float(cur_price)
        except Exception:
            cur_price = float(entry)

        change = (cur_price / entry - 1)
        pnl = alloc * change * lev
        pnl_val = float(pnl)
        color = "green" if pnl_val > 0 else "red"

        rows.append(f"""
          <tr>
            <td>{link}</td>
            <td>{t['entry_date']}</td>
            <td>{entry:,.2f}</td>
            <td>{cur_price:,.2f}</td>
            <td>{lev}</td>
            <td>{alloc:,}</td>
            <td style="color:{color};">{pnl_val:,.0f} ({change*lev*100:+.2f}%)</td>
          </tr>""")

    return f"""
    <div class="table-container">
    <h3>Open Trades</h3>
    <table class="styled-table">
      <tr>
        <th>Ticker</th><th>Entry Date</th><th>Entry Price</th><th>Current Price</th>
        <th>Leverage</th><th>Alloc (£)</th><th>Current PnL</th>
      </tr>
      {''.join(rows)}
    </table>
    </div>
    """

def make_trade_actions_html(trades):
    if not trades:
        return ""

    # build HTML dynamically depending on action type
    buy_rows = []
    sell_rows = []

    for t in trades:
        sym = t["Ticker"]
        link = f'<a href="https://finance.yahoo.com/quote/{sym}" target="_blank">{sym}</a>'
        action = "BUY" if "ENTER" in t["Signal"].upper() else "SELL"
        action_color = "green" if action == "BUY" else "red"
        entry_date = t.get("Entry Date", "–")
        price = t.get("Current Price", 0)
        target_exit = t.get("Target Exit", 0)

        if action == "BUY":
            # show target recovery etc.
            target_recovery = t.get("Target Recovery", "–")
            reason = "Buy the dip"
            buy_rows.append(f"""
              <tr>
                <td>{link}</td>
                <td style="color:{action_color}; font-weight:bold;">{action}</td>
                <td>{entry_date}</td>
                <td>{price:,.2f}</td>
                <td>{target_exit:,.2f}</td>
                <td>{target_recovery}</td>
                <td>{reason}</td>
              </tr>""")
        else:
            # show PnL and exit reason
            pnl_abs = t.get("PnL (£)", 0)
            pnl_pct = t.get("PnL (%)", 0)
            pnl_color = "green" if pnl_abs > 0 else "red"
            reason = t.get("Reason", "Target met")
            sell_rows.append(f"""
              <tr>
                <td>{link}</td>
                <td style="color:{action_color}; font-weight:bold;">{action}</td>
                <td>{entry_date}</td>
                <td>{price:,.2f}</td>
                <td>{target_exit:,.2f}</td>
                <td style="color:{pnl_color};">£{pnl_abs:,.0f} ({pnl_pct:+.2f}%)</td>
                <td>{reason}</td>
              </tr>""")

    html = '<div class="table-container"><h3>Trade Actions</h3>'
    if buy_rows:
        html += """
        <h4 style="color:green;">Buy Signals</h4>
        <table class="styled-table">
          <tr>
            <th>Ticker</th><th>Action</th><th>Entry Date</th>
            <th>Current Price</th><th>Target Exit</th><th>Target Recovery</th><th>Reason</th>
          </tr>
          {0}
        </table>""".format("".join(buy_rows))
    if sell_rows:
        html += """
        <h4 style="color:red;">Sell Signals</h4>
        <table class="styled-table">
          <tr>
            <th>Ticker</th><th>Action</th><th>Entry Date</th>
            <th>Current Price</th><th>Exit Price</th><th>PnL</th><th>Reason</th>
          </tr>
          {0}
        </table>""".format("".join(sell_rows))
    html += "</div>"
    return html

# ---------------- CONFIG ----------------
lookback_vol = 20
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=365)
interval = "1h"
BASE_DIR = Path(__file__).resolve().parent
open_trades_file = BASE_DIR / "open_trades.csv"
trades_to_report = []

params = {
    "^AXJO": (-0.05, 2.0, 1.15, 0.25, 1800, 4, 0.0),
    "^TWII": (-0.12, 2.0, 1.15, 0.25, 900, 4, 0.0),
    "^BSESN": (-0.05, 1.5, 1.15, 0.33, 1800, 3, 0.0),
    "^MXX": (-0.08, 1.5, 1.15, 0.33, 1800, 3, 0.0),
    "^J203.JO": (-0.08, 2.0, 1.05, 0.25, 3060, 4, 0.0),
    "^GSPC": (-0.08, 1.5, 0.95, 0.25, 900, 4, 0.0),
    "^FTSE": (-0.05, 1.0, 1.05, 0.33, 3600, 3, 0.0),
    "^STOXX50E": (-0.08, 2.0, 1.15, 0.25, 900, 4, 0.0),
    "^N225": (-0.08, 2.0, 1.15, 0.25, 900, 4, 0.0),
    "^HSI": (-0.12, 2.0, 0.95, 0.25, 900, 4, 0.0),
    "XU100.IS": (-0.12, 2.0, 1.15, 0.25, 1080, 4, 0.0),
    "^DJI": (-0.08, 2.0, 1.15, 0.33, 900, 3, 0.0),
    "^GSPTSE": (-0.05, 2.0, 1.15, 0.25, 900, 3, 0.0),
    "^GDAXI": (-0.08, 1.5, 1.15, 0.25, 900, 4, 0.0),
    "^FCHI": (-0.08, 2.0, 1.15, 0.25, 900, 4, 0.0),
    "^AEX": (-0.08, 2.0, 1.15, 0.25, 900, 4, 0.0),
}

# ---------------- HELPERS ----------------
def load_open_trades():
    if open_trades_file.exists():
        df = pl.read_csv(open_trades_file)
        if df.height > 0:
            return df
    # define explicit schema to avoid Null dtype
    return pl.DataFrame(
        {
            "Ticker": pl.Series([], dtype=pl.Utf8),
            "entry_date": pl.Series([], dtype=pl.Utf8),
            "entry_price": pl.Series([], dtype=pl.Float64),
            "leverage": pl.Series([], dtype=pl.Int64),
            "alloc": pl.Series([], dtype=pl.Int64),
            "peak_price": pl.Series([], dtype=pl.Float64),
        }
    )


def save_open_trades(df):
    df.write_csv(open_trades_file)

def get_vol_and_dd(df):
    close = df["Close"].to_numpy()
    roll_peak_160 = np.array([np.max(close[max(0, i - 160):i + 1]) for i in range(len(close))])
    drawdown = close / roll_peak_160 - 1.0
    ret = np.diff(close, prepend=close[0]) / close
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        vol = np.array([np.std(ret[max(0, i - lookback_vol):i]) for i in range(len(ret))])
    return drawdown, vol, np.nanmean(vol)

# ---------------- MAIN LOOP ----------------
open_trades = load_open_trades()
signals = []
market_summary = []

for sym, p in params.items():
    drop_threshold, vol_mult, target_recovery, stop_loss, max_hold_days, leverage, trailing_stop = p

    data = yf.download(
        sym,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False
    )
    if data.empty:
        continue
    if "Datetime" in data.columns:
        data.rename(columns={"Datetime": "Date"}, inplace=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]
    df = pl.from_pandas(data.reset_index()).rename({"Datetime": "Date"}).select(["Date", "Close"]).drop_nulls()
    drawdown, vol, vol_mean = get_vol_and_dd(df)
    df = df.with_columns([
        pl.Series("drawdown", drawdown),
        pl.Series("vol", vol),
        pl.lit(vol_mean).alias("vol_mean")
    ])
    last = df[-1]
    today = last["Date"].item()
    if hasattr(today, "tzinfo") and today.tzinfo is not None:
        today = today.replace(tzinfo=None)
    price = last["Close"].item()

    roll_peak_val = df["Close"].tail(160).max()
    # handle both float and Series cases
    if isinstance(roll_peak_val, pl.Series):
        roll_peak = roll_peak_val.item()
    else:
        roll_peak = float(roll_peak_val)   # convert to Python float
    current_dd = price / roll_peak - 1
    current_vol = float(vol[-1])
    vol_ratio = current_vol / vol_mean if vol_mean != 0 else np.nan

    market_summary.append({
        "Ticker": sym,
        "Close": round(price, 2),
        "Rolling Peak": round(float(roll_peak), 2),
        "Drawdown": round(current_dd * 100, 2),
        "Current Vol": round(current_vol, 5),
        "Mean Vol": round(vol_mean, 5),
        "Vol Ratio": round(vol_ratio, 2),
    })

    # ---------------- ENTRY OR EXIT ----------------
    existing = open_trades.filter(pl.col("Ticker") == sym)
    if existing.height == 0:
        calm_val = bool(last["vol"].item() <= vol_mean * vol_mult)
        recent_dd_val = df.tail(160)["drawdown"].min()
        if isinstance(recent_dd_val, pl.Series):
            recent_dd_val = recent_dd_val.item()

        # now both are plain scalars
        if recent_dd_val <= drop_threshold and calm_val:
            entry = {
                "Ticker": sym,
                "entry_date": today.strftime("%Y-%m-%d %H:%M"),
                "entry_price": float(price),
                "leverage": leverage,
                "alloc": 1_000_000,
                "peak_price": float(price),
            }
            open_trades = pl.concat([open_trades, pl.DataFrame([entry])])
            save_open_trades(open_trades)
            aimed_exit = price * (1 + abs(drop_threshold) * target_recovery)
            trades_to_report.append({
                "Ticker": sym,
                "Signal": "ENTER",
                "Entry Date": today.strftime("%Y-%m-%d %H:%M"),
                "Current Price": price,
                "Target Exit": aimed_exit,
                "Target Recovery": f"{target_recovery*100:.1f}%",
                "Leverage": leverage,
                "Stop Loss": f"{stop_loss*100:.1f}%",
                "Reason": "Buy the dip",
            })
            signals.append(f"🟢 {sym}: ENTER trade at {price:.2f} ({today.date()})")
    else:
        trade = existing.row(0, named=True)
        entry_price = trade["entry_price"]
        entry_date = datetime.datetime.fromisoformat(trade["entry_date"])
        alloc = trade["alloc"]
        peak_price = max(trade["peak_price"], price)
        change = (price / entry_price - 1)
        holding_days = (today - entry_date).days
        pnl = alloc * change * leverage

        reason = None
        trail_dd = (price / peak_price - 1) if trailing_stop > 0 else 0
        if trailing_stop > 0 and trail_dd <= -trailing_stop:
            reason = "trailing_stop"
        elif change <= -stop_loss:
            reason = "stop_loss"
        elif change >= target_recovery * abs(drop_threshold):
            reason = "target_gain"
        elif holding_days >= max_hold_days:
            reason = "timeout"

        if reason:
            pnl_pct = change * leverage * 100
            pnl_abs = pnl
            duration_hours = (today - entry_date).total_seconds() / 3600

            # determine human reason
            if "stop" in reason:
                reason_clean = "Stop loss"
            else:
                reason_clean = "Target met"

            trades_to_report.append({
                "Ticker": sym,
                "Signal": "EXIT",
                "Entry Date": entry_date.strftime("%Y-%m-%d %H:%M"),
                "Current Price": price,
                "Target Exit": price,
                "PnL (£)": round(pnl_abs, 2),
                "PnL (%)": round(pnl_pct, 2),
                "Reason": reason_clean,
            })

            signals.append(
                f"🔴 {sym}: EXIT trade ({reason_clean})\n"
                f"    PnL = {'+' if pnl_abs >= 0 else ''}£{pnl_abs:,.0f}  "
                f"({pnl_pct:+.2f}%) after {duration_hours:.0f} hours"
            )

            open_trades = open_trades.filter(pl.col("Ticker") != sym)
            save_open_trades(open_trades)
        else:
            open_trades = open_trades.with_columns(
                pl.when(pl.col("Ticker") == sym)
                .then(pl.lit(peak_price))
                .otherwise(pl.col("peak_price"))
                .alias("peak_price")
            )

# ---------------- SAVE + OUTPUT ----------------
save_open_trades(open_trades)



has_buy = any("ENTER" in t["Signal"].upper() for t in trades_to_report)
has_sell = any("EXIT" in t["Signal"].upper() for t in trades_to_report)

if has_buy and has_sell:
    subject = "Daily Strategy Report – BUY & SELL"
elif has_buy:
    subject = "Daily Strategy Report – BUY"
elif has_sell:
    subject = "Daily Strategy Report – SELL"
else:
    subject = "Daily Strategy Report"

html_body = "<html><head>" + HTML_STYLE + "</head><body>"
html_body += f"<h2>{subject}</h2>"

# Trade actions summary
if trades_to_report:
    html_body += make_trade_actions_html(trades_to_report)
else:
    html_body += "<p>✅ No new trades today — markets stable.</p>"

# add market and open-trade sections
if market_summary:
    html_body += make_market_html(market_summary, params)
html_body += make_open_trades_html(open_trades)
html_body += f"<p style='font-size:0.8em;color:#666;'>Report generated {datetime.datetime.now():%Y-%m-%d %H:%M UTC}</p>"
html_body += "</body></html>"

send_email(subject, html_body)
