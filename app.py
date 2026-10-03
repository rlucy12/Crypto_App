"""
CoinCast Analytics - next-day crypto price forecasts from pre-trained ARIMA + TCN models.
    1) python train.py        (once, offline)
    2) streamlit run app.py
"""
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from coins import COINS, DEFAULT_TRAIN, display_name, resolve
from predictor import available_models, fetch_recent, live_quote, load_artifacts, make_forecast

APP_NAME = "CoinCast Analytics"
TAGLINE = "Next-day crypto price forecasting"

st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide")

# ---------------- design tokens ----------------
INK, MUTED, LINE = "#14213D", "#5B6577", "#E3E6EB"
UP, DOWN, FORECAST = "#15803D", "#B42318", "#B45309"
PERIODS = {"7 days": 7, "30 days": 30, "90 days": 90}
WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "LTC-USD"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
.stApp, .stMarkdown, p, label, input, h1, h2, h3, h4, button p, [data-testid="stMetricValue"] {{
  font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
#MainMenu, footer, .stDeployButton, [data-testid="stDecoration"] {{ display: none; }}
.block-container {{ padding-top: 1.4rem; max-width: 1240px; }}
.brand h1 {{ font-size: 1.55rem; font-weight: 600; margin: 0; padding: 0; color: {INK}; letter-spacing: -0.01em; }}
.brand p {{ color: {MUTED}; margin: .1rem 0 0; font-size: .95rem; }}
.tape {{ display: flex; gap: .5rem; overflow-x: auto; padding: .2rem 0 .9rem; border-bottom: 1px solid {LINE}; margin-bottom: 1.1rem; }}
.tape .t {{ flex: 0 0 auto; background: #fff; border: 1px solid {LINE}; border-radius: 8px;
            padding: .45rem .75rem; font-size: .86rem; white-space: nowrap; color: {INK};
            font-variant-numeric: tabular-nums; }}
.tape .t b {{ font-weight: 600; margin-right: .45rem; }}
.up {{ color: {UP}; }} .down {{ color: {DOWN}; }}
.hero {{ display: flex; align-items: flex-end; gap: .9rem; flex-wrap: wrap; margin: .2rem 0 1.1rem; }}
.hero .name {{ font-size: 1.05rem; color: {MUTED}; width: 100%; }}
.hero .price {{ font-size: 2.7rem; font-weight: 600; line-height: 1; color: {INK}; font-variant-numeric: tabular-nums; }}
.badge {{ border-radius: 6px; padding: .22rem .55rem; font-weight: 600; font-size: .95rem; margin-bottom: .2rem; }}
.badge.up {{ background: #E7F5EC; }} .badge.down {{ background: #FBEAEA; }}
.hero .asof {{ color: {MUTED}; font-size: .85rem; margin-bottom: .3rem; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: .75rem; margin-bottom: 1.2rem; }}
.card {{ background: #fff; border: 1px solid {LINE}; border-radius: 10px; padding: .9rem 1rem; }}
.card .k {{ font-size: .83rem; color: {MUTED}; }}
.card .v {{ font-size: 1.45rem; font-weight: 600; margin-top: .15rem; color: {INK}; font-variant-numeric: tabular-nums; }}
.card .s {{ font-size: .83rem; color: {MUTED}; margin-top: .2rem; }}
.card.fc {{ border-left: 4px solid {FORECAST}; }}
.card .v.up {{ color: {UP}; }} .card .v.down {{ color: {DOWN}; }}
.foot {{ color: {MUTED}; font-size: .82rem; border-top: 1px solid {LINE}; padding-top: .8rem; margin-top: 1.5rem; }}
</style>
""", unsafe_allow_html=True)


# ---------------- helpers ----------------
def money(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    if abs(x) >= 1e9:
        return f"${x / 1e9:,.2f}B"
    return f"${x:,.2f}" if abs(x) >= 1 else f"${x:,.6f}"


def pct(x, signed=True) -> str:
    if x is None or np.isnan(x):
        return "–"
    return f"{x:+.2%}" if signed else f"{x:.2%}"


def tone(x) -> str:
    return "" if x is None or np.isnan(x) or x == 0 else ("up" if x > 0 else "down")


def card(label, value, sub="", cls="", value_cls="") -> str:
    return (f'<div class="card {cls}"><div class="k">{label}</div>'
            f'<div class="v {value_cls}">{value}</div><div class="s">{sub}</div></div>')


def cards(items) -> None:
    st.markdown('<div class="cards">' + "".join(items) + "</div>", unsafe_allow_html=True)


def base_layout(fig, height=440):
    fig.update_layout(height=height, hovermode="x unified", plot_bgcolor="#fff", paper_bgcolor="#fff",
                      font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", color=INK),
                      legend=dict(orientation="h", y=1.08, x=0), margin=dict(t=30, b=10, l=10, r=10))
    fig.update_xaxes(showgrid=False, linecolor=LINE)
    fig.update_yaxes(gridcolor="#F0F2F5", zeroline=False)
    return fig


# ---------------- cached data ----------------
@st.cache_resource(show_spinner=False)
def get_artifacts(ticker):
    return load_artifacts(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def get_data(ticker):
    return fetch_recent(ticker)


@st.cache_data(ttl=60, show_spinner=False)
def get_quote(ticker):
    return live_quote(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def get_forecast(ticker, last_date: str):
    return make_forecast(get_data(ticker), get_artifacts(ticker))


@st.cache_data(ttl=300, show_spinner=False)
def get_closes(tickers: tuple, days: int) -> pd.DataFrame:
    """Daily closes for several coins in one request (includes today's live candle)."""
    start = (pd.Timestamp.today() - pd.Timedelta(days=days + 3)).date().isoformat()
    df = yf.download(list(tickers), start=start, interval="1d", auto_adjust=True,
                     progress=False, threads=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


# ---------------- header ----------------
if "coin" not in st.session_state:
    st.session_state.coin = "bitcoin"


def _from_search():
    if st.session_state.search.strip():
        st.session_state.coin = st.session_state.search.strip()


def _from_pick():
    if st.session_state.pick:
        st.session_state.coin = st.session_state.pick


left, right = st.columns([3, 2], vertical_alignment="bottom")
with left:
    st.markdown(f'<div class="brand"><h1>{APP_NAME}</h1><p>{TAGLINE}</p></div>', unsafe_allow_html=True)
with right:
    st.text_input("Search a cryptocurrency", key="search", on_change=_from_search,
                  placeholder="Search by name or symbol, e.g. bitcoin, ETH", label_visibility="collapsed")

st.pills("Popular", [display_name(t) for t in DEFAULT_TRAIN], key="pick",
         on_change=_from_pick, label_visibility="collapsed")

# Market tape
try:
    tape = get_closes(tuple(WATCHLIST), 7)
    items = []
    for t in WATCHLIST:
        s = tape[t].dropna() if t in tape else pd.Series(dtype=float)
        if len(s) >= 2:
            ch = s.iloc[-1] / s.iloc[-2] - 1
            items.append(f'<div class="t"><b>{t.replace("-USD", "")}</b>{money(float(s.iloc[-1]))} '
                         f'<span class="{tone(ch)}">{pct(ch)}</span></div>')
    st.markdown('<div class="tape">' + "".join(items) + "</div>", unsafe_allow_html=True)
except Exception:
    st.markdown('<div class="tape"></div>', unsafe_allow_html=True)

# ---------------- load selected coin ----------------
query = st.session_state.coin
ticker = resolve(query)
t0 = time.time()
try:
    with st.spinner(f"Loading {display_name(ticker)}…"):
        feat = get_data(ticker)
        r = get_forecast(ticker, str(feat["Date"].iloc[-1].date()))
        q = get_quote(ticker)
except Exception:
    st.error(f"No market data found for “{query}”. Check the spelling or search by symbol, "
             f"for example BTC, ETH or SOL.")
    st.stop()

price = q["price"] or r["last_close"]
ch24 = price / r["last_close"] - 1
move = r["next_pred"] / price - 1
outlook = "Bullish" if move > 0.005 else "Bearish" if move < -0.005 else "Neutral"
outlook_note = {"Bullish": "Model expects a rise above 0.5%",
                "Bearish": "Model expects a fall below −0.5%",
                "Neutral": "Expected move within ±0.5%"}[outlook]

st.markdown(f"""
<div class="hero">
  <div class="name">{display_name(ticker)} ({ticker})</div>
  <div class="price">{money(price)}</div>
  <div class="badge {tone(ch24)}">{pct(ch24)} today</div>
  <div class="asof">Live price, updated {pd.Timestamp.now(tz="UTC"):%H:%M} UTC</div>
</div>""", unsafe_allow_html=True)

cards([
    card(f"Predicted close for {r['next_date']:%d %b %Y}", money(r["next_pred"]),
         f"vs {money(r['last_close'])} close on {r['last_date']:%d %b}", cls="fc"),
    card("Expected move from now", pct(move), "Predicted close vs live price", value_cls=tone(move)),
    card("Likely range (95%)", f"{money(r['next_lo'])} – {money(r['next_hi'])}",
         "Where the close should fall 19 days in 20"),
    card("Model outlook", outlook, outlook_note,
         value_cls={"Bullish": "up", "Bearish": "down"}.get(outlook, "")),
])

if r["mode"] == "arima_quick":
    st.info(f"{display_name(ticker)} uses a quick statistical forecast. For the full deep-learning "
            f"model, run `python train.py --coins {ticker.replace('-USD', '').lower()}`.")

tab_fc, tab_mkt, tab_cmp, tab_method = st.tabs(["Forecast", "Market data", "Compare", "Methodology"])

# ---------------- Forecast ----------------
with tab_fc:
    c1, c2, _ = st.columns([2, 2, 3])
    choice = c1.segmented_control("Period", list(PERIODS), default="30 days", key="fc_period")
    style = c2.segmented_control("Chart", ["Line", "Candles"], default="Line", key="fc_style")
    days = PERIODS[choice or "30 days"]
    dates = pd.to_datetime(r["dates"])[-days:]
    actual, pred = r["actual"][-days:], r["predicted"][-days:]

    fig = go.Figure()
    if style == "Candles":
        ohlc = feat.set_index("Date").loc[dates]
        fig.add_trace(go.Candlestick(x=dates, open=ohlc["Open"], high=ohlc["High"], low=ohlc["Low"],
                                     close=ohlc["Close"], name="Actual",
                                     increasing_line_color=UP, decreasing_line_color=DOWN))
    else:
        fig.add_trace(go.Scatter(x=dates, y=actual, name="Actual close", mode="lines+markers",
                                 line=dict(color=INK, width=2), marker=dict(size=5 if days <= 30 else 3)))
    fig.add_trace(go.Scatter(x=list(dates) + [r["next_date"]], y=list(pred) + [r["next_pred"]],
                             name="Predicted close", mode="lines",
                             line=dict(color=FORECAST, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[r["next_date"], r["next_date"]], y=[r["next_lo"], r["next_hi"]],
                             mode="lines", line=dict(color=FORECAST, width=6), opacity=0.25,
                             name="Likely range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[r["next_date"]], y=[r["next_pred"]], name="Next-day forecast",
                             mode="markers+text", text=[money(r["next_pred"])], textposition="middle right",
                             marker=dict(color=FORECAST, size=13, symbol="diamond")))
    base_layout(fig)
    fig.update_layout(xaxis_rangeslider_visible=False, yaxis_title="Price (USD)")
    st.plotly_chart(fig)

    err = np.abs(pred - actual) / actual * 100
    prev = r["actual"][-days - 1:-1] if len(r["actual"]) > days else None
    hit = (np.mean(np.sign(actual - prev) == np.sign(pred - prev)) * 100
           if prev is not None and len(prev) == days else np.nan)
    best = int(np.argmin(err))
    cards([
        card(f"Average error, last {days} days", f"{err.mean():.2f}%", "Mean absolute percentage error"),
        card("Direction accuracy", "–" if np.isnan(hit) else f"{hit:.0f}%", "Days the up/down call was right"),
        card("Largest miss", f"{err.max():.2f}%", f"on {dates[int(np.argmax(err))]:%d %b}"),
        card("Closest call", f"{err[best]:.2f}%", f"on {dates[best]:%d %b}"),
    ])

    table = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Actual close": actual,
                          "Predicted close": pred, "Error %": err}).iloc[::-1]
    with st.expander("Forecast history"):
        st.dataframe(table.style.format({"Actual close": money, "Predicted close": money,
                                         "Error %": "{:.2f}"}), hide_index=True)
    report = pd.concat([pd.DataFrame({"Date": [f"{r['next_date']:%Y-%m-%d}"], "Actual close": [np.nan],
                                      "Predicted close": [r["next_pred"]], "Error %": [np.nan]}), table])
    st.download_button("Download forecast report (CSV)", report.to_csv(index=False).encode(),
                       file_name=f"{ticker}_forecast_{r['next_date']:%Y%m%d}.csv", mime="text/csv")

# ---------------- Market data ----------------
with tab_mkt:
    c = feat["Close"]

    def change(d):
        return price / c.iloc[-d] - 1 if len(c) >= d else np.nan

    year = feat.iloc[-365:]
    vol30 = feat["LogReturn"].iloc[-30:].std() * np.sqrt(365)
    rsi = float(feat["RSI"].iloc[-1])
    rsi_note = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral zone"
    trend_up = feat["MA7"].iloc[-1] > feat["MA21"].iloc[-1]

    cards([card(label, pct(change(d)), sub, value_cls=tone(change(d))) for label, d, sub in [
        ("7-day change", 7, "Live price vs 7 days ago"),
        ("30-day change", 30, "Live price vs 30 days ago"),
        ("90-day change", 90, "Live price vs 90 days ago"),
        ("1-year change", 365, "Live price vs 1 year ago")]])
    cards([
        card("Today's range", f"{money(q['day_low'])} – {money(q['day_high'])}", "Intraday low and high"),
        card("52-week range", f"{money(year['Low'].min())} – {money(year['High'].max())}", "Lowest and highest trade"),
        card("Market cap", money(q["market_cap"]), "Yahoo Finance estimate"),
        card("Avg. daily volume (30d)", money(feat["Volume"].iloc[-30:].mean()), "US dollars traded"),
    ])
    cards([
        card("Volatility (30d, annualised)", pct(vol30, signed=False), "Higher means bigger daily swings"),
        card("RSI (14)", f"{rsi:.0f}", rsi_note),
        card("Short-term trend", "Upward" if trend_up else "Downward",
             "7-day average above 21-day" if trend_up else "7-day average below 21-day",
             value_cls="up" if trend_up else "down"),
    ])

    span = st.segmented_control("Range", ["3M", "6M", "1Y", "2Y"], default="1Y", key="mkt_range")
    h = feat.iloc[-{"3M": 90, "6M": 180, "1Y": 365, "2Y": 730}[span or "1Y"]:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(x=h["Date"], open=h["Open"], high=h["High"], low=h["Low"], close=h["Close"],
                                 name="Price", increasing_line_color=UP, decreasing_line_color=DOWN), row=1, col=1)
    fig.add_trace(go.Scatter(x=h["Date"], y=h["MA7"], name="7-day avg", line=dict(color=FORECAST, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=h["Date"], y=h["MA21"], name="21-day avg", line=dict(color="#3B5BA9", width=1)), row=1, col=1)
    fig.add_trace(go.Bar(x=h["Date"], y=h["Volume"], name="Volume", marker_color="#B8C0CC"), row=2, col=1)
    base_layout(fig, height=560)
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

    raw = feat[["Date", "Open", "High", "Low", "Close", "Volume"]].iloc[::-1]
    st.download_button("Download price history (CSV)", raw.to_csv(index=False).encode(),
                       file_name=f"{ticker}_history.csv", mime="text/csv")

# ---------------- Compare ----------------
with tab_cmp:
    names = {display_name(t): t for t in COINS}
    default = [display_name(t) for t in dict.fromkeys([ticker, "BTC-USD", "ETH-USD", "SOL-USD"]) if t in COINS]
    c1, c2 = st.columns([3, 1])
    picked = c1.multiselect("Coins to compare", list(names), default=default[:4], max_selections=6)
    span = c2.segmented_control("Range", ["30D", "90D", "1Y"], default="90D", key="cmp_range")
    if picked:
        d = {"30D": 30, "90D": 90, "1Y": 365}[span or "90D"]
        try:
            closes = get_closes(tuple(names[p] for p in picked), d).iloc[-d:]
            fig = go.Figure()
            rows = []
            for p in picked:
                s = closes[names[p]].dropna()
                if len(s) < 2:
                    continue
                fig.add_trace(go.Scatter(x=s.index, y=s / s.iloc[0] * 100, name=p, mode="lines"))
                lr = np.log(s).diff().dropna()
                rows.append({"Coin": p, "Price": money(float(s.iloc[-1])), "Return": pct(s.iloc[-1] / s.iloc[0] - 1),
                             "Volatility (ann.)": pct(lr.std() * np.sqrt(365), signed=False),
                             "Best day": pct(np.exp(lr.max()) - 1), "Worst day": pct(np.exp(lr.min()) - 1)})
            fig.add_hline(y=100, line_dash="dot", line_color="#B8C0CC")
            base_layout(fig)
            fig.update_layout(yaxis_title="Growth of 100 USD")
            st.plotly_chart(fig)
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        except Exception:
            st.warning("Comparison data is unavailable right now. Try again in a minute.")
    else:
        st.info("Choose at least one coin to compare.")

# ---------------- Methodology ----------------
with tab_method:
    m = r["meta"]
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("""
#### How the forecast is made
**Data.** Daily prices and trading volumes from Yahoo Finance, refreshed every 15 minutes.
Each day closes at 00:00 UTC, and the forecast is for the close of the next daily candle.

**Statistical model (ARIMA).** Captures the short-term momentum and mean-reversion in
the price series, using trading volume as an additional signal.

**Deep-learning model (TCN).** A temporal convolutional network reads the last 60 days of
returns, moving averages, RSI and volume, and predicts the next day's return.

**Combined forecast.** The two models are blended, with the mix chosen on data neither
model was trained on. Models are retrained offline on a regular schedule; the website
only runs them, so results load in seconds.

#### Limitations
Daily crypto prices behave close to a random walk, so a one-day forecast is usually near
today's price. Use the range and outlook as context, not as trading advice.
""")
    with right:
        st.markdown("#### Backtest results")
        if m:
            met = pd.DataFrame(m["metrics"]).T.rename(index={"Naive": "Baseline (no change)"})
            met = met[["MAPE %", "RMSE", "Direction acc. %"]].rename(
                columns={"MAPE %": "Avg. error %", "Direction acc. %": "Direction %"})
            st.dataframe(met.style.format({"Avg. error %": "{:.2f}", "RMSE": "{:,.2f}",
                                           "Direction %": "{:.0f}"}, na_rep="–"))
            st.caption(f"Scored on held-out days the models never saw. "
                       f"Model trained on data up to {m['trained_until']}. The baseline simply "
                       f"repeats yesterday's close; a useful model should beat it.")
        else:
            st.write("No backtest available: this coin uses the quick statistical model.")

st.markdown(f'<div class="foot">{APP_NAME}. Market data from Yahoo Finance. Page loaded in {time.time() - t0:.1f} s. '
            f'For research and education only, not investment advice.</div>', unsafe_allow_html=True)
