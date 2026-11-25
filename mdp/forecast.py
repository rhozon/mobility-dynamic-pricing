"""Demand forecast by zone and hour, made one week ahead.

Every feature is known a week before the hour it forecasts: calendar, holidays and lags of at
least 168 hours. Rolling origin: each test month is forecast by a model trained on everything
before it. Two baselines: the same hour last week, and the mean of the same hour over the last
four weeks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from pandas.tseries.holiday import USFederalHolidayCalendar

LAGS = [168, 336, 504, 672]


def hourly_panel(zone_hour: pd.DataFrame, start: str = "2024-09-01", min_trips_day: float = 50) -> pd.DataFrame:
    z = zone_hour.copy()
    z["ts"] = pd.to_datetime(z["date"]) + pd.to_timedelta(z["hour"], unit="h")
    z = z[z["ts"] >= pd.Timestamp(start)]
    y = z.groupby(["zone", "ts"])["trips"].sum()
    keep = y.groupby("zone").mean() * 24 >= min_trips_day
    y = y[y.index.get_level_values("zone").isin(keep[keep].index)]
    full = pd.MultiIndex.from_product(
        [y.index.get_level_values("zone").unique(), pd.date_range(start, z["ts"].max(), freq="h")],
        names=["zone", "ts"])
    p = y.reindex(full, fill_value=0).rename("y").reset_index()
    p = p.sort_values(["zone", "ts"])
    for L in LAGS:
        p[f"lag{L}"] = p.groupby("zone")["y"].shift(L)
    p["mean4w"] = p[[f"lag{L}" for L in LAGS]].mean(axis=1)
    p["hour"] = p["ts"].dt.hour
    p["dow"] = p["ts"].dt.dayofweek
    hol = USFederalHolidayCalendar().holidays(p["ts"].min(), p["ts"].max())
    p["holiday"] = p["ts"].dt.normalize().isin(hol).astype(int)
    p["month"] = p["ts"].dt.month
    return p.dropna(subset=[f"lag{L}" for L in LAGS])


FEATS = ["zone", "hour", "dow", "holiday", "month"] + [f"lag{L}" for L in LAGS] + ["mean4w"]


def wape(y, p) -> float:
    return float(np.abs(y - p).sum() / np.abs(y).sum())


def backtest(p: pd.DataFrame, test_months=("2025-04", "2025-05", "2025-06")) -> pd.DataFrame:
    rows = []
    for m in test_months:
        a = pd.Timestamp(m + "-01")
        b = a + pd.offsets.MonthBegin(1)
        tr, te = p[p["ts"] < a], p[(p["ts"] >= a) & (p["ts"] < b)]
        model = xgb.XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.06, subsample=0.8,
                                 colsample_bytree=0.8, objective="reg:tweedie", tree_method="hist",
                                 random_state=2026, n_jobs=8)
        model.fit(tr[FEATS], tr["y"])
        pred = model.predict(te[FEATS])
        rows.append({"month": m, "model": "XGBoost", "wape": wape(te["y"], pred)})
        rows.append({"month": m, "model": "Same hour last week", "wape": wape(te["y"], te["lag168"])})
        rows.append({"month": m, "model": "Mean of last 4 weeks", "wape": wape(te["y"], te["mean4w"])})
    return pd.DataFrame(rows)
