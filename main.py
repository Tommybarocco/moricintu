
import os
import requests
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prophet import Prophet
from typing import Optional

BINANCE_URL = "https://api.binance.com/api/v3/klines"
API_TOKEN = os.getenv("API_TOKEN")  # optional simple auth

app = FastAPI(title="Prophet Forecast API", version="1.0.0")

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
    trend: str
    score: int
    delta: float
    last_yhat: float
    last_yhat_upper: float
    last_yhat_lower: float

def fetch_binance_klines(symbol: str, interval: str, limit: int = 500):
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(BINANCE_URL, params=params, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Binance error: {r.text}")
    data = r.json()
    if not isinstance(data, list) or len(data) == 0:
        raise HTTPException(status_code=502, detail="No data from Binance")
    cols = ["openTime","open","high","low","close","volume","closeTime","qav","ntrades","takerBase","takerQuote","ignore"]
    df = pd.DataFrame(data, columns=cols)
    df["ds"] = pd.to_datetime(df["closeTime"], unit="ms")
    df["y"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[["ds","y"]].dropna()
    if len(df) < 50:
        raise HTTPException(status_code=422, detail="Not enough history for forecasting")
    return df

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
    interval: str = Query("1h", description="Binance interval, e.g., 1h, 4h, 1d"),
    periods: int = Query(24, ge=1, le=365, description="Steps to forecast"),
    x_api_token: Optional[str] = Header(None, convert_underscores=False)
):
    if API_TOKEN and x_api_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    df = fetch_binance_klines(symbol, interval, limit=500)
    fc = run_prophet(df, periods=periods, interval=interval)
    horizon = min(periods, 24 if interval.endswith("h") else 10)
    trend, delta, score = compute_trend(fc, horizon_last=horizon)
    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "periods": periods,
        "trend": trend,
        "score": int(score),
        "delta": float(delta),
        "last_yhat": float(fc["yhat"].iloc[-1]),
        "last_yhat_upper": float(fc["yhat_upper"].iloc[-1]),
        "last_yhat_lower": float(fc["yhat_lower"].iloc[-1])
    }
