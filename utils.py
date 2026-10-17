"""Shared market-data helpers and Plotly styling for StockCast AI."""

import pandas as pd
import requests
import streamlit as st
import yfinance as yf


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
