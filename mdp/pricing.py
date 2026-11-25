"""Pricing quotes from the estimated model, shared by the API and the page.

For an area, a day type and an hour, the platform's baseline (trips, fare, driver pay share and wait
at the current price) comes from the observed profiles. A price change m = 1 + change moves demand
through the ride-hail elasticity, the wait through demand per driver (drivers held fixed), and the
platform's rider share through the estimated choice model; the three are solved together.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import RESULTS

AREAS = ["Congestion zone", "Manhattan north", "Brooklyn", "Queens", "Bronx and Staten Island", "Airports"]


@dataclass
class Params:
    eps: float = -0.4
    alpha: float = 0.21
    beta_p: float = 0.30
    beta_w: float = 0.08
    rho: float = 0.61
    m_min: float = 0.7
    m_max: float = 2.5


def load_params(results: Path = RESULTS) -> Params:
    cm = pd.read_csv(results / "choice_models.csv")
    s = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    get = lambda model, term: float(cm[(cm["model"] == model) & (cm["term"] == term)]["coef"].iloc[0])
    return Params(alpha=max(s["wait_elasticity"], 0.05), beta_p=-get("static", "dlogp"),
                  beta_w=-get("static", "dwait"), rho=get("dynamic", "L_lag"))


def baseline(profiles: pd.DataFrame, area: str, weekend: int, hour: int) -> dict:
    p = profiles[(profiles["area"] == area) & (profiles["weekend"] == weekend) & (profiles["hour"] == hour)]
    if p.empty:
        raise KeyError(f"no profile for {area}, weekend={weekend}, hour={hour}")
    l, u = p[p["platform"] == "lyft"].iloc[0], p[p["platform"] == "uber"].iloc[0]
    return {"Q": float(l["trips"]), "p0": float(l["fare"]), "theta": float(l["theta"]), "w0": float(l["wait"]),
            "s0": float(l["trips"] / (l["trips"] + u["trips"]))}


def quote(b: dict, change: float, par: Params, eps: float | None = None) -> dict:
    eps = par.eps if eps is None else eps
    m = 1 + change
    if m <= 0:
        raise ValueError("price change must be above -100%")
    L0 = math.log(b["s0"] / (1 - b["s0"]))
    s, w, q = b["s0"], b["w0"], b["Q"]
    for _ in range(50):
        q = b["Q"] * m ** eps * s / b["s0"]
        w = b["w0"] * (q / b["Q"]) ** par.alpha
        s = 1 / (1 + math.exp(-(L0 - par.beta_p * math.log(m) - par.beta_w * (w - b["w0"]))))
    rev = m * b["p0"] * q
    return {"price_multiplier": m, "trips_per_hour": q, "revenue_per_hour": rev,
            "margin_per_hour": (1 - b["theta"]) * rev, "driver_pay_per_hour": b["theta"] * rev,
            "wait_minutes": w, "platform_share": s}


def clearing_multiplier(drivers_vs_usual: float, share_vs_usual: float, demand_vs_usual: float, par: Params) -> float:
    """Multiplier that brings the wait back to its usual level given today's drivers and demand."""
    m = (drivers_vs_usual / share_vs_usual / demand_vs_usual) ** (1 / par.eps)
    return min(par.m_max, max(par.m_min, m))


def smoothed(previous: float, target: float, lam: float) -> float:
    return previous + lam * (target - previous)
