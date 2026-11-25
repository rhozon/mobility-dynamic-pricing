"""Fare model: the part of the price explained by the trip, and the part that moves with the market.

A log-linear base fare on distance, time, app and airport separates what any trip of that length
would cost. What is left, by area and hour, is a dynamic price index: 1.10 means trips there and
then cost 10% more than the same distance and time would cost on average.
An XGBoost model with time and place shows how much of the fare a full model explains.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import xgboost as xgb

FEATURES = ["miles", "minutes", "hour", "dow", "pu", "do_zone", "is_lyft", "airport", "cbd"]


def prepare(sample: pd.DataFrame) -> pd.DataFrame:
    s = sample[(sample["miles"] > 0.1) & (sample["minutes"] > 1) & (sample["fare"] > 2)].copy()
    s["date"] = pd.to_datetime(s["date"])
    s["log_fare"] = np.log(s["fare"])
    s["log_miles"] = np.log(s["miles"])
    s["log_minutes"] = np.log(s["minutes"])
    s["is_lyft"] = (s["platform"] == "lyft").astype(int)
    s["airport"] = (s["airport_fee"].fillna(0) > 0).astype(int)
    s["cbd"] = (s["cbd_fee"].fillna(0) > 0).astype(int)
    return s


def base_model(train: pd.DataFrame):
    return smf.ols("log_fare ~ is_lyft * (log_miles + log_minutes) + airport", data=train).fit()


def price_index(s: pd.DataFrame, base, area: pd.Series) -> pd.DataFrame:
    """Mean of fare / base fare by area, weekday or weekend, and hour."""
    s = s.copy()
    s["ratio"] = np.exp(s["log_fare"] - base.predict(s))
    s["area"] = s["pu"].map(area)
    s["weekend"] = (s["dow"].isin([0, 6])).astype(int)  # DuckDB dayofweek: 0 = Sunday
    g = s.dropna(subset=["area"]).groupby(["area", "weekend", "hour"]).agg(
        index=("ratio", "mean"), trips=("ratio", "size"), wait_min=("wait_s", lambda x: x.mean() / 60)).reset_index()
    g["index"] = g["index"] / np.average(s["ratio"], weights=None)
    return g


def full_model(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    gbm = xgb.XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.08, subsample=0.8,
                           colsample_bytree=0.9, tree_method="hist", enable_categorical=True,
                           max_cat_to_onehot=1, random_state=2026, n_jobs=8)
    cats = pd.CategoricalDtype(range(0, 266))
    X = lambda d: d[FEATURES].astype({"pu": cats, "do_zone": cats})
    gbm.fit(X(train), train["log_fare"])
    pred = np.exp(gbm.predict(X(test)))
    base = base_model(train)
    pred_base = np.exp(base.predict(test))
    mape = lambda p: float(np.mean(np.abs(p - test["fare"]) / test["fare"]))
    mae = lambda p: float(np.mean(np.abs(p - test["fare"])))
    return {"mape_base": mape(pred_base), "mape_full": mape(pred), "mae_base": mae(pred_base),
            "mae_full": mae(pred), "n_train": int(len(train)), "n_test": int(len(test))}
