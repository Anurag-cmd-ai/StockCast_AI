import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="StockCast AI", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#17211d; --muted:#748078; --green:#00a86b; --green-dark:#078655; --line:#e7ede9; --canvas:#f5f8f6; --card:#ffffff; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] { background:var(--canvas)!important; color:var(--ink)!important; font-family:'DM Sans',sans-serif!important; }
[data-testid="stHeader"] { background:rgba(245,248,246,.88)!important; }
[data-testid="stSidebar"] { background:#fff!important; border-right:1px solid var(--line); }
section[data-testid="stMain"] > div { padding-top:1.3rem; }
.block-container { max-width:1440px; padding-left:2.4rem; padding-right:2.4rem; padding-bottom:3rem; }
.brand-row { display:flex; align-items:center; gap:10px; margin:0 0 1.1rem; color:var(--ink); font-weight:800; font-family:Manrope,sans-serif; font-size:1.2rem; letter-spacing:-.04em; }
.brand-mark { display:inline-grid; place-items:center; width:34px;height:34px;border-radius:11px;background:#dff7eb;color:#008c59;font-size:18px; }
.hero { background:linear-gradient(112deg,#e8f8ef 0%,#f3fbf6 58%,#eef8f2 100%); border:1px solid #d9efe2; border-radius:22px; padding:1.65rem 2rem; margin:0 0 1.15rem; position:relative; overflow:hidden; }
.hero:after { content:'';position:absolute;width:220px;height:220px;right:5%;top:-120px;border-radius:50%;background:rgba(0,168,107,.11); }
.hero h1 { color:var(--ink); font:800 1.8rem/1.2 Manrope,sans-serif; letter-spacing:-.055em; margin:0 0 .35rem; }
.hero p { margin:0;color:#66766d;font-size:.94rem; }
.eyebrow { color:#008c59;text-transform:uppercase;font-weight:700;letter-spacing:.12em;font-size:.7rem;margin-bottom:.55rem; }
[data-testid="stWidgetLabel"] p, label { color:#536158!important;font-size:.78rem!important;font-weight:700!important;letter-spacing:.025em; }
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-testid="stSelectbox"] [data-baseweb="select"] > div { background:#fff!important;color:var(--ink)!important;border:1px solid #dce5df!important;border-radius:10px!important; }
[data-testid="stTextInput"] input:focus,[data-testid="stNumberInput"] input:focus { border-color:#00a86b!important;box-shadow:0 0 0 3px rgba(0,168,107,.12)!important; }
[data-testid="stButton"] > button { border-radius:11px!important;font-weight:700!important;border:1px solid #dce5df!important;background:#fff!important;color:#34433a!important;transition:all .18s ease!important; }
[data-testid="stButton"] > button:hover { border-color:#00a86b!important;color:#008c59!important;transform:translateY(-1px); }
[data-testid="stButton"] > button[kind="primary"] { background:#00a86b!important;border-color:#00a86b!important;color:white!important;box-shadow:0 5px 15px rgba(0,168,107,.2)!important; }
[data-testid="metric-container"] { background:#fff!important;border:1px solid var(--line)!important;border-radius:16px!important;padding:1rem 1.1rem!important;box-shadow:0 4px 16px rgba(26,53,39,.035)!important; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color:#738078!important;font-size:.76rem!important;font-weight:700!important;letter-spacing:.045em;text-transform:uppercase; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color:var(--ink)!important;font-family:Manrope,sans-serif!important;font-size:1.55rem!important;font-weight:800!important;letter-spacing:-.04em; }
[data-testid="stMetricDelta"] { font-size:.8rem!important;font-weight:700!important; }
[data-testid="stRadio"] [role="radiogroup"] { gap:.35rem; }
[data-testid="stRadio"] label { text-transform:none!important;letter-spacing:0!important;font-size:.76rem!important; }
[data-testid="stCheckbox"] label { text-transform:none!important;letter-spacing:0!important;font-size:.74rem!important; }
.ticker-head { font:800 1.45rem Manrope,sans-serif;color:var(--ink);letter-spacing:-.04em;display:flex;align-items:center;gap:.65rem;flex-wrap:wrap;margin:.2rem 0 .8rem; }
.ticker-badge { background:#e6f7ee;color:#008c59;border:1px solid #c9ebd8;border-radius:999px;padding:4px 11px;font:700 .75rem 'DM Sans',sans-serif;letter-spacing:.025em; }
.section-head { color:#35443b;font:800 .98rem Manrope,sans-serif;letter-spacing:-.02em;margin:1rem 0 .5rem; }
.helper-copy { color:#748078;font-size:.86rem; }
[data-testid="stAlert"] { border-radius:13px!important;border:1px solid var(--line)!important; }
[data-testid="stDataFrame"] { border-radius:14px!important;border:1px solid var(--line)!important;overflow:hidden; }
[data-testid="stDownloadButton"] > button { background:#fff!important;border:1px solid #cfe8da!important;color:#078655!important;border-radius:10px!important;font-weight:700!important; }
[data-testid="stDownloadButton"] > button:hover { background:#f0faf4!important;border-color:#00a86b!important; }
[data-testid="stVerticalBlockBorderWrapper"] { background:#fff;border:1px solid var(--line)!important;border-radius:18px!important;box-shadow:0 5px 20px rgba(26,53,39,.035); }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:1.4rem;border-bottom:1px solid var(--line); }
[data-testid="stTabs"] [data-baseweb="tab"] { color:#718078;font-weight:700; }
[data-testid="stTabs"] [aria-selected="true"] { color:#008c59!important; }
[data-testid="stSpinner"] p { color:#66766d!important; }
#MainMenu, footer { visibility:hidden; }
@media (max-width: 760px) { .block-container{padding-left:1rem;padding-right:1rem}.hero{padding:1.3rem}.hero h1{font-size:1.45rem} }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BRAND & SEARCH
# ─────────────────────────────────────────────
st.markdown('<div class="brand-row"><span class="brand-mark">↗</span> StockCast <span style="font-weight:500;color:#7b8981;font-size:.78rem;letter-spacing:0">MARKET STUDIO</span></div>', unsafe_allow_html=True)
st.markdown("""
<div class="hero">
  <div class="eyebrow">Your market, at a glance</div>
  <h1>Invest with a clearer view.</h1>
  <p>Explore market history, compare a model-generated scenario, and plan a recurring investment.</p>
</div>
""", unsafe_allow_html=True)
with st.container(border=True):
    st.markdown('<div class="eyebrow">Explore an asset</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.3, 1.5, 1])
    with c1:
        market = st.selectbox("Market", ["Auto-detect / Global", "US Market", "Indian Market (NSE)", "Indian Market (BSE)", "UK Market (LSE)", "Cryptocurrency"])
    with c2:
        user_input = st.text_input("Company or symbol", placeholder="Search Apple, AAPL, TCS, Bitcoin…")
    with c3:
        timeline_unit = st.selectbox("History window", ["Days", "Weeks", "Months", "Years"], index=2)
        timeline_map = {"Days": [7, 14, 30, 60, 90], "Weeks": [1, 2, 4, 8, 12, 26, 52], "Months": [1, 3, 6, 12, 24], "Years": [1, 2, 3, 5]}
        timeline_val = st.selectbox(f"Window ({timeline_unit.lower()})", timeline_map[timeline_unit], index=2 if timeline_unit == "Months" else 0)
predict_btn = st.button("View market dashboard  →", type="primary", use_container_width=True)
st.caption("Market data provided by Yahoo Finance. Forecasts are illustrative model outputs, not investment advice.")

# ─────────────────────────────────────────────
# SHARED PLOTLY LAYOUT
# ─────────────────────────────────────────────
CHART_LAYOUT = dict(
    height=430,
    margin=dict(l=8, r=8, t=28, b=8),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0)",
    font=dict(family="DM Sans, sans-serif", color="#34433a", size=12),
    xaxis=dict(
        showgrid=True, gridcolor="rgba(39,71,52,0.08)",
        gridwidth=1, zeroline=False,
        showline=True, linewidth=1, linecolor="rgba(39,71,52,0.16)", mirror=True,
        tickfont=dict(color="#718078", size=11),
        rangeslider=dict(visible=False),
        rangeselector=dict(
            bgcolor="rgba(255,255,255,0.95)",
            activecolor="#dff7eb",
            font=dict(color="#536158", size=10),
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="ALL"),
            ],
        ),
    ),
    yaxis=dict(
        showgrid=True, gridcolor="rgba(39,71,52,0.08)",
        gridwidth=1, zeroline=False,
        showline=True, linewidth=1, linecolor="rgba(39,71,52,0.16)", mirror=True,
        tickfont=dict(color="#718078", size=11),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#e3ebe5",
        borderwidth=1,
        font=dict(color="#34433a", size=12),
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#ffffff",
        bordercolor="#00a86b",
        font=dict(family="DM Sans, sans-serif", color="#17211d", size=14),
    ),
)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def to_period(unit, val):
    days = {"Days": val, "Weeks": val * 7, "Months": val * 30, "Years": val * 365}[unit]
    if days <= 30:   return "1mo"
    if days <= 90:   return "3mo"
    if days <= 180:  return "6mo"
    if days <= 365:  return "1y"
    if days <= 730:  return "2y"
    if days <= 1095: return "3y"
    return "5y"

@st.cache_data(ttl=86400, show_spinner=False)
def get_symbol_from_name(query, market_type):
    query = query.strip()
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={requests.utils.quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            quotes = resp.json().get("quotes", [])
            if quotes:
                if market_type == "Indian Market (NSE)":
                    match = next(
                        (q for q in quotes if q.get("exchange") == "NSI" or str(q.get("symbol", "")).endswith(".NS")),
                        None,
                    )
                    if match: return match["symbol"]
                elif market_type == "Indian Market (BSE)":
                    match = next(
                        (q for q in quotes if q.get("exchange") == "BSE" or str(q.get("symbol", "")).endswith(".BO")),
                        None,
                    )
                    if match: return match["symbol"]
                elif market_type == "US Market":
                    match = next((q for q in quotes if q.get("exchange") in ["NMS", "NYQ"]), None)
                    if match: return match["symbol"]
                elif market_type == "UK Market (LSE)":
                    match = next((q for q in quotes if str(q.get("symbol", "")).endswith(".L")), None)
                    if match: return match["symbol"]
                elif market_type == "Cryptocurrency":
                    match = next((q for q in quotes if q.get("quoteType") == "CRYPTOCURRENCY"), None)
                    if match: return match["symbol"]
                return quotes[0]["symbol"]
    except Exception:
        pass

    ticker = query.upper()
    if market_type == "Indian Market (NSE)" and not ticker.endswith(".NS"): return f"{ticker}.NS"
    if market_type == "Indian Market (BSE)" and not ticker.endswith(".BO"): return f"{ticker}.BO"
    if market_type == "UK Market (LSE)"     and not ticker.endswith(".L"):  return f"{ticker}.L"
    if market_type == "Cryptocurrency"      and not ticker.endswith("-USD"):return f"{ticker}-USD"
    return ticker

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, 2, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class SeqDataset(Dataset):
    def __init__(self, X, seq_len):
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, i):
        return self.X[i : i + self.seq_len], self.X[i + self.seq_len]


@st.cache_data(ttl=3600, show_spinner=False)
def train_and_predict(prices_tuple, future_days=30):
    prices = np.array(prices_tuple)
    SEQ_LEN = min(30, max(5, len(prices) // 4))
    EPOCHS  = 30
    scaler  = MinMaxScaler()
    scaled  = scaler.fit_transform(prices.reshape(-1, 1))
    split   = int(len(scaled) * 0.85)
    tr_ds   = SeqDataset(scaled[:split].flatten(), SEQ_LEN)
    tr_dl   = DataLoader(tr_ds, batch_size=32, shuffle=False)

    model   = LSTMModel()
    opt     = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad()
            loss = loss_fn(model(xb.unsqueeze(-1)).squeeze(), yb.squeeze())
            loss.backward()
            opt.step()

    last_seq   = scaled[-SEQ_LEN:].flatten().tolist()
    future_raw = []
    model.eval()

    with torch.no_grad():
        for _ in range(future_days):
            x   = torch.tensor(last_seq[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            out = model(x).item()
            future_raw.append(out)
            last_seq.append(out)

    return scaler.inverse_transform(np.array(future_raw).reshape(-1, 1)).flatten()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if predict_btn:
    if not user_input.strip():
        st.warning("Please enter a stock symbol or company name.")
        st.stop()
        
    st.session_state['active_pred'] = True
    st.session_state['p_user_input'] = user_input
    st.session_state['p_market'] = market
    st.session_state['p_period'] = to_period(timeline_unit, timeline_val)

if st.session_state.get('active_pred'):
    p_user_input = st.session_state['p_user_input']
    p_market = st.session_state['p_market']
    p_period = st.session_state['p_period']

    with st.spinner("Resolving symbol and fetching live market data…"):
        fetch_ticker = get_symbol_from_name(p_user_input, p_market)
        df = fetch_stock_data(fetch_ticker, p_period)

    if df is None or df.empty or "Close" not in df.columns:
        st.error(
            f"Could not retrieve valid historical data for **{p_user_input}** (Resolved Ticker: `{fetch_ticker}`). "
            "Please check the spelling or ensure the market selection is correct."
        )
        st.session_state['active_pred'] = False
        st.stop()

    close = df["Close"].dropna().values.flatten().astype(float)
    dates = df.index[: len(close)]

    if len(close) < 10:
        st.error(f"Not enough historical data to train the model for `{fetch_ticker}`. Try a longer timeline.")
        st.stop()

    if "US" in p_market or p_market == "Cryptocurrency": currency = "USD"
    elif "Indian" in p_market:                           currency = "INR"
    elif "UK" in p_market:                               currency = "GBP"
    else:                                                currency = ""

    # Fetch 52-week data points
    fifty2_low = close.min()
    fifty2_high = close.max()
    mkt_cap = None
    try:
        info     = yf.Ticker(fetch_ticker).info
        name     = info.get("longName") or info.get("shortName") or p_user_input.upper()
        currency = info.get("currency", currency)
        fifty2_low  = info.get("fiftyTwoWeekLow",  fifty2_low)
        fifty2_high = info.get("fiftyTwoWeekHigh", fifty2_high)
        mkt_cap     = info.get("marketCap", None)
    except Exception:
        name = p_user_input.upper()

    with st.spinner("Training LSTM model on historical data…"):
        future_prices = train_and_predict(tuple(close))

    future_dates = pd.bdate_range(start=dates[-1] + pd.Timedelta(days=1), periods=30)

    # ── Stats ──────────────────────────────────
    curr        = close[-1]
    chg         = ((close[-1] - close[-2]) / close[-2]) * 100 if len(close) > 1 else 0
    trend       = ((future_prices[-1] - curr) / curr) * 100
    week_close  = close[-5:] if len(close) >= 5 else close
    week_high   = week_close.max()
    day_high    = float(df["High"].iloc[-1])  if "High" in df.columns else curr
    day_low     = float(df["Low"].iloc[-1])   if "Low"  in df.columns else curr

    # ── Heading ────────────────────────────────
    st.markdown(
        f'<div class="ticker-head">{name}'
        f'<span class="ticker-badge">{fetch_ticker}</span></div>',
        unsafe_allow_html=True,
    )
    st.write("")

    # ── Metric cards ───────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"{currency} {curr:,.2f}",      f"{chg:+.2f}% today")
    m2.metric("High",          f"{currency} {day_high:,.2f}",  f"{((day_high - curr) / curr) * 100:+.2f}% vs now")
    m3.metric("Low",           f"{currency} {day_low:,.2f}",   f"{((day_low  - curr) / curr) * 100:+.2f}% vs now")
    m4.metric("Week High",     f"{currency} {week_high:,.2f}", f"{((week_high - curr) / curr) * 100:+.2f}% vs now")

    st.write("")

    # ── Forecast Alert & Gauge Row ──────────────────
    top_c1, top_c2 = st.columns([1.5, 1])
    
    with top_c1:
        st.markdown("<div class='section-head'> 30-Day Outlook</div>", unsafe_allow_html=True)
        st.write("")
        if future_prices[-1] > curr:
            st.success(
                f"**▲ Expected to GO UP** · {trend:+.2f}% over 30 days · "
                f"Target: {currency} {future_prices[-1]:,.2f}"
            )
        else:
            st.error(
                f"**▼ Expected to FALL DOWN** · {trend:+.2f}% over 30 days · "
                f"Target: {currency} {future_prices[-1]:,.2f}"
            )
            
    with top_c2:
        st.markdown("<div class='section-head'> Key Stats</div>", unsafe_allow_html=True)
        st.write("")
        kc1, kc2 = st.columns(2)
        kc1.metric("52-Week High", f"{currency} {fifty2_high:,.2f}",
                   f"{((fifty2_high - curr) / curr) * 100:+.2f}% vs now")
        if mkt_cap:
            if mkt_cap >= 1_000_000_000_000:
                mkt_str = f"{currency} {mkt_cap / 1_000_000_000_000:.2f}T"
            elif mkt_cap >= 1_000_000_000:
                mkt_str = f"{currency} {mkt_cap / 1_000_000_000:.2f}B"
            elif mkt_cap >= 1_000_000:
                mkt_str = f"{currency} {mkt_cap / 1_000_000:.2f}M"
            else:
                mkt_str = f"{currency} {mkt_cap:,.0f}"
        else:
            mkt_str = "N/A"
        kc2.metric("Mkt Cap", mkt_str)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Historical Chart Controls ───────────────────
    st.markdown("<div class='section-head'>Historical price movement</div>", unsafe_allow_html=True)
    controls_a, controls_b, controls_c = st.columns([1.6, 1.0, 0.72])
    with controls_a:
        chart_style = st.radio("Chart style", ["Line", "Candlestick"], horizontal=True, index=0, key="history_chart_style")
    with controls_b:
        show_averages = st.checkbox("SMA 20 / 50", value=True, key="history_show_sma")
    with controls_c:
        show_volume = st.checkbox("Volume", value=False, key="history_show_volume")

    has_ohlc = all(col in df.columns for col in ["Open", "High", "Low", "Close"])
    use_candles = chart_style == "Candlestick" and has_ohlc
    add_volume = show_volume and "Volume" in df.columns
    chart_rows = 2 if add_volume else 1
    row_heights = [0.76, 0.24] if add_volume else [1.0]
    fig1 = make_subplots(
        rows=chart_rows, cols=1, shared_xaxes=add_volume,
        vertical_spacing=0.04, row_heights=row_heights,
    )

    if use_candles:
        candle_trace = go.Candlestick(
            x=dates,
            open=df["Open"].iloc[:len(dates)],
            high=df["High"].iloc[:len(dates)],
            low=df["Low"].iloc[:len(dates)],
            close=df["Close"].iloc[:len(dates)],
            name="OHLC",
            increasing_line_color="#00a86b",
            decreasing_line_color="#e05252",
            increasing_fillcolor="rgba(0,168,107,0.78)",
            decreasing_fillcolor="rgba(224,82,82,0.78)",
            whiskerwidth=0.45,
        )
        fig1.add_trace(candle_trace, row=1, col=1)
    else:
        line_trace = go.Scatter(
            x=dates,
            y=close,
            mode="lines",
            name="Close price",
            line=dict(color="#00a86b", width=2.8),
            fill="tozeroy",
            fillcolor="rgba(0,168,107,0.10)",
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Price: " + currency + " %{y:,.2f}<extra></extra>",
        )
        fig1.add_trace(line_trace, row=1, col=1)

    if show_averages:
        close_series = pd.Series(close, index=dates)
        for window, color in [(20, "#4682c5"), (50, "#c18a2c")]:
            if len(close_series) >= window:
                sma = close_series.rolling(window, min_periods=window).mean()
                sma_trace = go.Scatter(
                    x=dates,
                    y=sma,
                    mode="lines",
                    name=f"SMA {window}",
                    line=dict(color=color, width=1.5, dash="dot"),
                    hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>SMA {window}: {currency} %{{y:,.2f}}<extra></extra>",
                )
                fig1.add_trace(sma_trace, row=1, col=1)

    if add_volume:
        volume = pd.to_numeric(df["Volume"].iloc[:len(dates)], errors="coerce").fillna(0)
        if use_candles:
            up_day = df["Close"].iloc[:len(dates)].to_numpy() >= df["Open"].iloc[:len(dates)].to_numpy()
        else:
            up_day = pd.Series(close, index=dates).diff().fillna(0).to_numpy() >= 0
        volume_colors = ["rgba(0,168,107,0.55)" if is_up else "rgba(224,82,82,0.50)" for is_up in up_day]
        volume_trace = go.Bar(
            x=dates,
            y=volume,
            marker_color=volume_colors,
            name="Volume",
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Volume: %{y:,.0f}<extra></extra>",
        )
        fig1.add_trace(volume_trace, row=2, col=1)
        fig1.update_yaxes(title_text="Volume", row=2, col=1, showgrid=False, tickfont=dict(color="#718078", size=9))

    layout1 = {**CHART_LAYOUT}
    layout1["height"] = 480 if add_volume else 430
    layout1["xaxis"] = {**layout1["xaxis"], "rangeslider": {"visible": False}, "title": None}
    layout1["yaxis"] = {**layout1["yaxis"], "title": dict(text=f"Price ({currency})", font=dict(color="#718078", size=11))}
    layout1["xaxis2"] = {"showgrid": False, "tickfont": dict(color="#718078", size=10), "rangeslider": {"visible": False}}
    layout1["showlegend"] = True
    layout1["legend"] = {**layout1["legend"], "orientation": "h", "y": 1.12}
    fig1.update_layout(**layout1)
    fig1.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": True, "displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

    # ── Forecast Chart ─────────────────────────
    st.markdown("<div class='section-head'> 30-Day Model Scenario</div>", unsafe_allow_html=True)

    fig2      = go.Figure()
    hist_x    = dates[-60:]
    hist_y    = close[-60:]
    is_up     = future_prices[-1] > curr
    color_pred = "#00a86b" if is_up else "#e05252"
    fill_rgba  = "rgba(0,168,107,0.10)" if is_up else "rgba(224,82,82,0.09)"
    glow_rgba  = "rgba(0,168,107,0.20)" if is_up else "rgba(224,82,82,0.18)"

    fig2.add_trace(go.Scatter(
        x=hist_x, y=hist_y,
        mode="lines", name="Recent Market",
        line=dict(color="#00a86b", width=2.7),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Price: " + currency + " %{y:,.2f}<extra></extra>",
    ))
    fig2.add_trace(go.Scatter(
        x=future_dates, y=future_prices,
        mode="lines",
        line=dict(color=glow_rgba, width=9),
        showlegend=False, hoverinfo="skip",
    ))
    fig2.add_trace(go.Scatter(
        x=future_dates, y=future_prices,
        mode="lines+markers", name="Forecast Trend",
        line=dict(color=color_pred, width=2.5, dash="dash"),
        marker=dict(size=5, symbol="circle", color=color_pred,
                    line=dict(width=1.5, color="rgba(255,255,255,0.25)")),
        fill="tozeroy", fillcolor=fill_rgba,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast: " + currency + " %{y:,.2f}<extra></extra>",
    ))

    today_x = str(dates[-1].date())
    fig2.add_shape(
        type="line", x0=today_x, x1=today_x, y0=0, y1=1, yref="paper",
        line=dict(color="rgba(148,163,184,0.45)", width=1.5, dash="dot"),
    )
    fig2.add_annotation(
        x=today_x, y=0.97, yref="paper", text="Today",
        showarrow=False, xanchor="left",
        font=dict(color="#94a3b8", size=11, family="DM Sans"),
    )

    layout2 = {**CHART_LAYOUT}
    layout2["yaxis"] = {**layout2["yaxis"], "title": dict(text=f"Price ({currency})", font=dict(color="#e2e8f0", size=13))}
    fig2.update_layout(**layout2)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": True, "displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

    # ── Forecast Table ─────────────────────────
    st.markdown("<div class='section-head'> Model Scenario — Next 30 Days</div>", unsafe_allow_html=True)

    pred_df = pd.DataFrame({
        "Date": future_dates.strftime("%Y-%m-%d"),
        f"Predicted ({currency})": np.round(future_prices, 2),
        "Change from today": [
            f"{'▲' if p >= curr else '▼'} {((p - curr) / curr) * 100:+.2f}%"
            for p in future_prices
        ],
    })

    def style_trend(val):
        if "▲" in str(val): return "color: #22c55e; font-weight: bold;"
        if "▼" in str(val): return "color: #ef4444; font-weight: bold;"
        return ""

    styled_df = pred_df.set_index("Date").style.map(style_trend, subset=["Change from today"])
    st.dataframe(styled_df, use_container_width=True, height=350)

    csv = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Forecast as CSV",
        data=csv,
        file_name=f"{fetch_ticker}_forecast.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Lumpsum & SIP Calculator ────────────────
    st.markdown("<div class='section-head'> Long-Term Returns Calculator</div>", unsafe_allow_html=True)
    st.markdown("Calculate potential long-term returns for this asset based on the 30-day forecasted trend.")

    calc_type = st.radio("Investment Mode", ["Lumpsum", "SIP (Monthly)"], horizontal=True)
    st.markdown("<br>", unsafe_allow_html=True)

    calc_c1, calc_c2 = st.columns(2)

    with calc_c1:
        if calc_type == "Lumpsum":
            investment = st.number_input(f"Total Investment ({currency})", min_value=100.0, value=10000.0, step=1000.0)
        else:
            investment = st.number_input(f"Monthly Investment ({currency})", min_value=100.0, value=1000.0, step=500.0)

    with calc_c2:
        duration_years = st.selectbox("Investment Duration (Years)", [1, 2, 3, 5, 10, 15, 20, 25, 30], index=3)

    # Annualize the model's 30-day forecast using CAGR: (1 + r_30d)^(365/30) - 1
    expected_return = round(((1 + trend / 100) ** (365 / 30) - 1) * 100, 2)
    return_icon = "" if expected_return >= 0 else ""
    if expected_return >= 0:
        st.success(
            f"{return_icon} **Model-Predicted Annual Return: {expected_return:+.2f}%** — "
            f"Annualized from the 30-day forecast ({trend:+.2f}% in 30 days)"
        )
    else:
        st.error(
            f"{return_icon} **Model-Predicted Annual Return: {expected_return:+.2f}%** — "
            f"Annualized from the 30-day forecast ({trend:+.2f}% in 30 days)"
        )

    # Mathematical Calculations
    if calc_type == "Lumpsum":
        total_invested = investment
        rate = expected_return / 100
        total_value = investment * ((1 + rate) ** duration_years)
        est_returns = total_value - total_invested
    else: # SIP Calculation
        total_invested = investment * 12 * duration_years
        
        if expected_return == 0:  # Division by zero safety
            total_value = total_invested
            est_returns = 0.0
        else:
            monthly_rate = expected_return / 12 / 100
            months = duration_years * 12
            total_value = investment * (((1 + monthly_rate)**months - 1) / monthly_rate) * (1 + monthly_rate)
            est_returns = total_value - total_invested

    # Metric Display
    st.markdown("<br>", unsafe_allow_html=True)
    res_c1, res_c2, res_c3 = st.columns(3)
    res_c1.metric("Total Invested", f"{currency} {total_invested:,.2f}")
    
    # Show red/green depending on profit vs loss
    sign = "+" if est_returns >= 0 else ""
    res_c2.metric("Estimated Returns", f"{currency} {est_returns:,.2f}", f"{sign}{expected_return}% CAGR")
    res_c3.metric("Total Projected Value", f"{currency} {total_value:,.2f}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    if est_returns > 0:
        st.success(
            f"### PROFIT! \n"
            f"By making a **{calc_type}** investment of **{currency} {total_invested:,.2f}** over **{duration_years} years** "
            f"at an expected rate of **{expected_return}%**, your money is projected to grow to **{currency} {total_value:,.2f}**.\n\n"
            f"**Total Profit Earned:** {currency} {est_returns:,.2f}"
        )
    elif est_returns < 0:
        st.error(
            f"### LOSS! \n"
            f"By making a **{calc_type}** investment of **{currency} {total_invested:,.2f}** over **{duration_years} years** "
            f"at an expected rate of **{expected_return}%**, your money is projected to shrink to **{currency} {total_value:,.2f}**.\n\n"
            f"**Total Loss Incurred:** {currency} {abs(est_returns):,.2f}"
        )
    else:
        st.info(
            f"### BREAK EVEN \n"
            f"By making a **{calc_type}** investment of **{currency} {total_invested:,.2f}** over **{duration_years} years**, "
            f"your money will not grow or shrink.\n\n"
            f"**Final Value:** {currency} {total_value:,.2f}"
        )
