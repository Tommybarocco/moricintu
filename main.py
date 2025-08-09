
import os
import requests
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prophet import Prophet
from typing import Optional

API_TOKEN = os.getenv("API_TOKEN")  # optional simple auth

app = FastAPI(title="Prophet Forecast API (multi-provider)", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ForecastResponse(BaseModel):
    symbol: str
    interval: str
    periods: int
    provider: str
    trend: str
    score: int
    delta: float
    last_yhat: float
    last_yhat_upper: float
    last_yhat_lower: float

# -------- Data providers --------
def fetch_coinbase_klines(symbol: str, interval: str, limit: int = 300):
    product = symbol.replace("USDT", "USD").replace("/", "-")
    gran_map = {"1h":3600, "4h":14400, "1d":86400, "1w":604800}
    gran = gran_map.get(interval, 3600)
    url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    r = requests.get(url, params={"granularity": gran}, timeout=30, headers={"User-Agent":"prophet-api"})
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Coinbase error: {r.text}")
    data = r.json()
    if not isinstance(data, list) or len(data) == 0:
        raise HTTPException(status_code=502, detail="No data from Coinbase")
    df = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
    df["ds"] = pd.to_datetime(df["time"], unit="s")
    df["y"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.sort_values("ds")
    return df[["ds","y"]].dropna()

def fetch_kraken_klines(symbol: str, interval: str, limit: int = 300):
    pair = symbol.replace("BTCUSDT","XBTUSDT")
    interval_map = {"1h":60, "4h":240, "1d":1440}
    intr = interval_map.get(interval, 60)
    url = "https://api.kraken.com/0/public/OHLC"
    r = requests.get(url, params={"pair": pair, "interval": intr}, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Kraken error: {r.text}")
    js = r.json()
    if js.get("error"):
        raise HTTPException(status_code=502, detail=f"Kraken error: {js['error']}")
    result = next(iter(js["result"].values()))
    df = pd.DataFrame(result, columns=["time","open","high","low","close","vwap","volume","count"])
    df["ds"] = pd.to_datetime(df["time"], unit="s")
    df["y"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.sort_values("ds")
    return df[["ds","y"]].dropna()

def fetch_okx_klines(symbol: str, interval: str, limit: int = 300):
    instId = symbol.replace("BTCUSDT","BTC-USDT")
    tf_map = {"1h":"1H", "4h":"4H", "1d":"1D"}
    tf = tf_map.get(interval, "1H")
    url = "https://www.okx.com/api/v5/market/candles"
    r = requests.get(url, params={"instId": instId, "bar": tf, "limit": str(limit)}, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OKX error: {r.text}")
    js = r.json()
    if js.get("code") != "0":
        raise HTTPException(status_code=502, detail=f"OKX error: {js.get('msg','unknown')}")
    data = js.get("data", [])
    if not data:
        raise HTTPException(status_code=502, detail="No data from OKX")
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"])
    df["ds"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    df["y"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.sort_values("ds")
    return df[["ds","y"]].dropna()

def fetch_klines(provider: str, symbol: str, interval: str, limit: int = 300):
    if provider == "coinbase":
        return fetch_coinbase_klines(symbol, interval, limit)
    elif provider == "kraken":
        return fetch_kraken_klines(symbol, interval, limit)
    elif provider == "okx":
        return fetch_okx_klines(symbol, interval, limit)
    else:
        return fetch_coinbase_klines(symbol, interval, limit)

# -------- Prophet --------
def run_prophet(df: pd.DataFrame, periods: int, interval: str):
    freq = "H" if interval.endswith("h") else "D"
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    fc = m.predict(future)
    return fc

def compute_trend(fc: pd.DataFrame, horizon_last: int):
    yhat = fc["yhat"]
    base = yhat.iloc[-horizon_last-1]
    delta = float(yhat.iloc[-1] - base)
    rel = abs(delta) / (abs(base) + 1e-9)
    if delta > 0:
        trend = "bullish"; score = 3 if rel > 0.02 else 2
    elif delta < 0:
        trend = "bearish"; score = 3 if rel > 0.02 else 2
    else:
        trend = "neutral"; score = 1
    return trend, delta, score

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    symbol: str = Query(..., description="e.g., BTCUSDT"),
    interval: str = Query("1h", description="1h, 4h, 1d"),
    periods: int = Query(24, ge=1, le=365),
    provider: str = Query("coinbase", description="coinbase|kraken|okx"),
    x_api_token: Optional[str] = Header(None, convert_underscores=False)
):
    if API_TOKEN and x_api_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    df = fetch_klines(provider, symbol, interval, limit=300)
    if len(df) < 50:
        raise HTTPException(status_code=422, detail="Not enough history for forecasting")
    fc = run_prophet(df, periods=periods, interval=interval)
    horizon = min(periods, 24 if interval.endswith("h") else 10)
    trend, delta, score = compute_trend(fc, horizon_last=horizon)
    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "periods": periods,
        "provider": provider,
        "trend": trend,
        "score": int(score),
        "delta": float(delta),
        "last_yhat": float(fc["yhat"].iloc[-1]),
        "last_yhat_upper": float(fc["yhat_upper"].iloc[-1]),
        "last_yhat_lower": float(fc["yhat_lower"].iloc[-1])
    }
