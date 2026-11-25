"""Drift monitoring: has the data the model sees moved away from the data it learned from?

Population stability index (PSI) of trip distance, minutes, fare per mile and wait, each month
against a reference window, on the 0.5% trip sample. Common reading: below 0.1 stable, 0.1 to 0.25
watch, above 0.25 act. Alongside, the monthly data-quality report flags changes in how the
source is filled, such as the arrival time that Lyft trips started to carry in March 2025.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = {"miles": "miles", "minutes": "minutes", "fare_per_mile": None, "wait_min": None}


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def monthly(sample: pd.DataFrame, ref_from: str = "2024-09-01", ref_to: str = "2024-12-31") -> pd.DataFrame:
    s = sample.copy()
    s["date"] = pd.to_datetime(s["date"])
    s["fare_per_mile"] = s["fare"] / s["miles"].clip(lower=0.1)
    s["wait_min"] = s["wait_s"] / 60
    s["month"] = s["date"].dt.to_period("M").astype(str)
    ref = s[(s["date"] >= ref_from) & (s["date"] <= ref_to)]
    rows = []
    for m, g in s.groupby("month"):
        for f in FEATURES:
            v = psi(ref[f].to_numpy(dtype=float), g[f].to_numpy(dtype=float))
            rows.append({"month": m, "feature": f, "psi": v,
                         "status": "stable" if v < 0.1 else ("watch" if v < 0.25 else "act")})
    return pd.DataFrame(rows)
