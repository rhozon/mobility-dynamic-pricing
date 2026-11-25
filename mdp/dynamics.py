"""Rider-side dynamics: how fast Lyft's share moved when its price changed.

From 5 to 31 January 2025 Lyft returned the $1.50 congestion fee as a credit on rides in the
Congestion Relief Zone, each credit valid for a week. Uber did not. The fee itself applies to both
apps, so within the zone it shifts their prices alike; the credit lowers Lyft's effective price
alone. We measure the Lyft share among Uber and Lyft trips that start in each zone.

Three nested choice models, in logit share L = log(s / (1 - s)):
- static choice (rho = 0): L responds only to today's prices and waits;
- logit dynamic with inertia (0 < rho < 1): L moves part of the way to today's target;
- replicator-like drift (rho = 1): L accumulates payoff differences and never reverts.
The data pick rho.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .config import CREDIT_START, CREDIT_USE_END


def logit(s):
    s = np.clip(s, 1e-4, 1 - 1e-4)
    return np.log(s / (1 - s))


def expit(x):
    return 1 / (1 + np.exp(-x))


def daily_shares(zone_hour: pd.DataFrame, min_trips: int = 30) -> pd.DataFrame:
    """One row per zone and day with Lyft share, price per mile and mean wait of each app."""
    g = zone_hour.groupby(["date", "zone", "platform"])[
        ["trips", "fare_sum", "miles_sum", "wait_sum", "wait_n"]].sum().unstack("platform")
    out = pd.DataFrame(index=g.index)
    for p in ("uber", "lyft"):
        out[f"trips_{p}"] = g[("trips", p)].fillna(0)
        out[f"ppm_{p}"] = g[("fare_sum", p)] / g[("miles_sum", p)]
        out[f"wait_{p}"] = g[("wait_sum", p)] / g[("wait_n", p)] / 60
    out = out.reset_index()
    out["date"] = pd.to_datetime(out["date"])
    out["trips"] = out["trips_uber"] + out["trips_lyft"]
    out = out[(out["trips_uber"] >= min_trips) & (out["trips_lyft"] >= min_trips)].copy()
    out["share"] = out["trips_lyft"] / out["trips"]
    out["L"] = logit(out["share"])
    out["dlogp"] = np.log(out["ppm_lyft"] / out["ppm_uber"])
    out["dwait"] = out["wait_lyft"] - out["wait_uber"]
    out["dow"] = out["date"].dt.dayofweek
    return out.dropna(subset=["dlogp", "dwait"])


def credit_flag(dates: pd.Series, year_shift: int = 0) -> pd.Series:
    a = pd.Timestamp(CREDIT_START) - pd.DateOffset(years=year_shift)
    b = pd.Timestamp(CREDIT_USE_END) - pd.DateOffset(years=year_shift)
    return ((dates >= a) & (dates <= b)).astype(int)


def did(d: pd.DataFrame, crz: set[int], year_shift: int = 0) -> pd.DataFrame:
    """Difference in differences of Lyft's logit share, zone and date fixed effects.

    Window: December to April. `Credit` covers 5 January to 7 February, `After` the rest.
    year_shift=1 runs the same calendar a year earlier, when there was no credit (placebo).
    """
    a = pd.Timestamp("2024-12-01") - pd.DateOffset(years=year_shift)
    b = pd.Timestamp("2025-04-30") - pd.DateOffset(years=year_shift)
    w = d[(d["date"] >= a) & (d["date"] <= b)].copy()
    w["crz"] = w["zone"].isin(crz).astype(int)
    w["credit"] = credit_flag(w["date"], year_shift)
    w["after"] = (w["date"] > pd.Timestamp(CREDIT_USE_END) - pd.DateOffset(years=year_shift)).astype(int)
    w["crz_credit"] = w["crz"] * w["credit"]
    w["crz_after"] = w["crz"] * w["after"]
    m = smf.wls("L ~ crz_credit + crz_after + C(zone) + C(date)", data=w, weights=w["trips"]).fit(
        cov_type="cluster", cov_kwds={"groups": w["zone"]})
    rows = []
    for k in ("crz_credit", "crz_after"):
        lo, hi = m.conf_int().loc[k]
        rows.append({"term": k, "coef": m.params[k], "lo": lo, "hi": hi, "p": m.pvalues[k],
                     "n_obs": int(m.nobs), "window": f"{a.date()} to {b.date()}"})
    return pd.DataFrame(rows)


def event_series(d: pd.DataFrame, crz: set[int], year_shift: int = 0) -> pd.DataFrame:
    """Daily Lyft share inside minus outside the zone (trip-weighted), December to April."""
    a = pd.Timestamp("2024-12-01") - pd.DateOffset(years=year_shift)
    b = pd.Timestamp("2025-04-30") - pd.DateOffset(years=year_shift)
    w = d[(d["date"] >= a) & (d["date"] <= b)].copy()
    w["in"] = w["zone"].isin(crz)
    g = w.groupby(["date", "in"]).apply(lambda x: x["trips_lyft"].sum() / x["trips"].sum(),
                                        include_groups=False).unstack("in")
    out = pd.DataFrame({"inside": g[True], "outside": g[False]})
    out["gap"] = out["inside"] - out["outside"]
    out["day"] = (out.index - a).days
    return out.reset_index()


FORMULA_X = "credit + dlogp + dwait + C(dow) + C(zone)"


def fit_models(d: pd.DataFrame, crz: set[int], train_end: str = CREDIT_USE_END,
               train_start: str = "2024-09-01") -> dict:
    """Static and dynamic choice models for zones inside the congestion zone."""
    w = d[d["zone"].isin(crz)].sort_values(["zone", "date"]).copy()
    w["credit"] = credit_flag(w["date"])
    w["L_lag"] = w.groupby("zone")["L"].shift(1)
    gap = (w["date"] - w.groupby("zone")["date"].shift(1)).dt.days
    w.loc[gap != 1, "L_lag"] = np.nan
    tr = w[(w["date"] >= train_start) & (w["date"] <= train_end)].dropna(subset=["L_lag"])
    static = smf.ols("L ~ " + FORMULA_X, data=tr).fit(cov_type="cluster", cov_kwds={"groups": tr["zone"]})
    dyn = smf.ols("L ~ L_lag + " + FORMULA_X, data=tr).fit(cov_type="cluster", cov_kwds={"groups": tr["zone"]})
    return {"data": w, "static": static, "dynamic": dyn, "train_end": pd.Timestamp(train_end)}


def forecast(models: dict, test_end: str = "2025-04-30") -> pd.DataFrame:
    """Forecast each zone's daily share after the credit ended.

    The dynamic model runs recursively from the last observed share before the test window, so
    it never sees an observed share inside the window; both models use observed prices and waits.
    """
    w, st, dy, t0 = models["data"], models["static"], models["dynamic"], models["train_end"]
    te = w[(w["date"] > t0) & (w["date"] <= pd.Timestamp(test_end))].copy()
    te = te[te["zone"].isin(set(w[w["date"] <= t0]["zone"]))]
    te["pred_static"] = st.predict(te)
    out = []
    rho = dy.params["L_lag"]
    for z, g in te.groupby("zone"):
        g = g.sort_values("date").copy()
        last = w[(w["zone"] == z) & (w["date"] <= t0)].sort_values("date")
        if last.empty:
            continue
        prev = last["L"].iloc[-1]
        prev_date = last["date"].iloc[-1]
        base = dy.predict(g.assign(L_lag=0.0))  # everything except the lag term
        preds = []
        for (_, r), b in zip(g.iterrows(), base):
            # bridge days without data by letting the share keep adjusting toward its target
            steps = max(1, (r["date"] - prev_date).days)
            for _ in range(steps):
                prev = b + rho * prev
            preds.append(prev)
            prev_date = r["date"]
        g["pred_dynamic"] = preds
        out.append(g)
    res = pd.concat(out)
    res["share_static"] = expit(res["pred_static"])
    res["share_dynamic"] = expit(res["pred_dynamic"])
    return res


def compare(fc: pd.DataFrame) -> dict:
    """RMSE of the share, trip-weighted, and a Diebold-Mariano test on daily losses."""
    e_s = (fc["share"] - fc["share_static"]) ** 2
    e_d = (fc["share"] - fc["share_dynamic"]) ** 2
    wts = fc["trips"]
    daily = pd.DataFrame({"date": fc["date"], "ls": e_s * wts, "ld": e_d * wts, "w": wts}).groupby("date").sum()
    diff = (daily["ls"] - daily["ld"]) / daily["w"]  # positive when the dynamic model is better
    n = len(diff)
    lag = int(np.floor(n ** (1 / 3)))
    x = diff - diff.mean()
    gamma0 = (x @ x) / n
    lrv = gamma0 + 2 * sum((1 - k / (lag + 1)) * (x[k:].values @ x[:-k].values) / n for k in range(1, lag + 1))
    dm = diff.mean() / np.sqrt(lrv / n)
    from scipy.stats import norm
    return {
        "rmse_static": float(np.sqrt((e_s * wts).sum() / wts.sum())),
        "rmse_dynamic": float(np.sqrt((e_d * wts).sum() / wts.sum())),
        "dm_stat": float(dm), "dm_p_one_sided": float(1 - norm.cdf(dm)),
        "days": int(n), "zones": int(fc["zone"].nunique()),
    }


def summary(models: dict) -> pd.DataFrame:
    st, dy = models["static"], models["dynamic"]
    rows = []
    for name, m in (("static", st), ("dynamic", dy)):
        for k in ("L_lag", "credit", "dlogp", "dwait"):
            if k in m.params:
                lo, hi = m.conf_int().loc[k]
                rows.append({"model": name, "term": k, "coef": m.params[k], "lo": lo, "hi": hi,
                             "p": m.pvalues[k], "n_obs": int(m.nobs), "r2": m.rsquared})
    return pd.DataFrame(rows)
