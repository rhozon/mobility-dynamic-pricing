"""Pricing API.

    uvicorn mdp.api:app --host 0.0.0.0 --port 8000

GET  /health       model version and data window
POST /quote        outcomes of a price change in an area and hour
POST /multiplier   market-clearing and smoothed multipliers for today's supply and demand
"""
from __future__ import annotations

import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import pricing
from .config import RESULTS

VERSION = "0.2.0"
app = FastAPI(title="Ride-hailing pricing", version=VERSION)
_profiles = pd.read_csv(RESULTS / "profiles.csv")
_params = pricing.load_params()
_summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))


class QuoteIn(BaseModel):
    area: str = Field(examples=["Congestion zone"])
    weekend: int = Field(0, ge=0, le=1)
    hour: int = Field(18, ge=0, le=23)
    price_change: float = Field(0.10, gt=-0.9, lt=3.0, description="0.10 means a 10% higher price")


class MultiplierIn(BaseModel):
    drivers_vs_usual: float = Field(1.0, gt=0)
    rider_share_vs_usual: float = Field(1.0, gt=0)
    demand_vs_usual: float = Field(1.0, gt=0)
    previous_multiplier: float = Field(1.0, gt=0)
    smoothing: float = Field(0.3, gt=0, le=1)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": VERSION, "months": _summary["months"], "areas": pricing.AREAS}


@app.post("/quote")
def quote(q: QuoteIn) -> dict:
    try:
        b = pricing.baseline(_profiles, q.area, q.weekend, q.hour)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    now, new = pricing.quote(b, 0.0, _params), pricing.quote(b, q.price_change, _params)
    low = pricing.quote(b, q.price_change, _params, eps=-0.6)
    high = pricing.quote(b, q.price_change, _params, eps=-0.2)
    rng = sorted([low["margin_per_hour"] / now["margin_per_hour"] - 1, high["margin_per_hour"] / now["margin_per_hour"] - 1])
    return {"baseline": now, "quote": new, "margin_change": new["margin_per_hour"] / now["margin_per_hour"] - 1,
            "margin_change_range": rng}


@app.post("/multiplier")
def multiplier(m: MultiplierIn) -> dict:
    target = pricing.clearing_multiplier(m.drivers_vs_usual, m.rider_share_vs_usual, m.demand_vs_usual, _params)
    return {"clearing": target, "smoothed": pricing.smoothed(m.previous_multiplier, target, m.smoothing)}
