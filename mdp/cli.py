"""Command line: download, aggregate, analyse, export.

    python -m mdp.cli download     # monthly files into MDP_RAW (about 7 GB)
    python -m mdp.cli aggregate    # DuckDB pass over every month
    python -m mdp.cli analyse      # models, tables in results/
    python -m mdp.cli export PATH  # JSON for the interactive page
"""
from __future__ import annotations

import json
import sys
import urllib.request

import numpy as np
import pandas as pd

from . import aggregate, dynamics, experiment, fare, forecast, simulate, zones
from .config import BASE_URL, DATA, DERIVED, MONTHS, RAW, RESULTS, ZONES_URL

AREAS = zones.AREAS


def cmd_download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for m in MONTHS:
        f = RAW / f"fhvhv_tripdata_{m}.parquet"
        if not f.exists():
            print("downloading", m, flush=True)
            urllib.request.urlretrieve(BASE_URL.format(month=m), f)
    urllib.request.urlretrieve(ZONES_URL, DATA / "taxi_zone_lookup.csv")


def load():
    zh = pd.read_parquet(DERIVED / "zone_hour.parquet")
    crz = zones.crz_zones(zh)
    area = zones.area_of(pd.read_csv(DATA / "taxi_zone_lookup.csv"), crz)
    return zh, crz, area


def wait_elasticity(zh: pd.DataFrame, crz: set[int]) -> float:
    """Within zone and hour of week, how much the mean wait rises with trips (2025, after Feb 7)."""
    z = zh[(pd.to_datetime(zh["date"]) > pd.Timestamp("2025-02-07"))].copy()
    z = z.groupby(["date", "hour", "zone"])[["trips", "wait_sum", "wait_n"]].sum().reset_index()
    z = z[(z["trips"] >= 20) & (z["wait_n"] > 0)]
    z["lw"] = np.log(z["wait_sum"] / z["wait_n"])
    z["lq"] = np.log(z["trips"])
    z["cell"] = z["zone"].astype(str) + "_" + pd.to_datetime(z["date"]).dt.dayofweek.astype(str) + "_" + z["hour"].astype(str)
    for c in ("lw", "lq"):
        z[c + "_d"] = z[c] - z.groupby("cell")[c].transform("mean")
    return float((z["lw_d"] * z["lq_d"]).sum() / (z["lq_d"] ** 2).sum())


def calibration(zh, area, hour: int = 18) -> pd.DataFrame:
    z = zh[(pd.to_datetime(zh["date"]) > pd.Timestamp("2025-02-07"))].copy()
    z["dow"] = pd.to_datetime(z["date"]).dt.dayofweek
    z = z[(z["hour"] == hour) & (z["dow"] < 5)]
    z["area"] = z["zone"].map(area)
    days = z["date"].nunique()
    g = z.groupby(["area", "platform"])[["trips", "fare_sum", "pay_sum", "wait_sum", "wait_n"]].sum().unstack("platform")
    out = pd.DataFrame({
        "Q": g[("trips", "lyft")] / days,
        "p0": g[("fare_sum", "lyft")] / g[("trips", "lyft")],
        "theta": g[("pay_sum", "lyft")] / g[("fare_sum", "lyft")],
        "w0": g[("wait_sum", "lyft")] / g[("wait_n", "lyft")] / 60,
        "s0": g[("trips", "lyft")] / (g[("trips", "lyft")] + g[("trips", "uber")]),
        "uber_trips": g[("trips", "uber")] / days,
    }).reindex(AREAS)
    return out


def params(cal: pd.DataFrame, models: dict, alpha: float) -> simulate.Params:
    st, dy = models["static"], models["dynamic"]
    return simulate.Params(Q=cal["Q"].values, p0=cal["p0"].values, theta=cal["theta"].values,
                           w0=cal["w0"].values, s0=cal["s0"].values, alpha=alpha,
                           beta_p=float(-st.params["dlogp"]), beta_w=float(-st.params["dwait"]),
                           rho=float(dy.params["L_lag"]))


def cmd_analyse() -> dict:
    RESULTS.mkdir(exist_ok=True)
    zh, crz, area = load()
    res = {"crz_zones": sorted(crz)}
    q = pd.read_csv(DERIVED / "quality.csv")
    q.to_csv(RESULTS / "data_quality.csv", index=False)

    d = dynamics.daily_shares(zh)
    did = pd.concat([dynamics.did(d, crz).assign(year="2024-25"), dynamics.did(d, crz, 1).assign(year="2023-24 placebo")])
    did.to_csv(RESULTS / "did_credit.csv", index=False)
    ev = pd.concat([dynamics.event_series(d, crz).assign(year="2024-25"),
                    dynamics.event_series(d, crz, 1).assign(year="2023-24")])
    ev.to_csv(RESULTS / "event_series.csv", index=False)
    models = dynamics.fit_models(d, crz)
    dynamics.summary(models).to_csv(RESULTS / "choice_models.csv", index=False)
    cmp = {h: dynamics.compare(dynamics.forecast(models, e)) for h, e in
           (("feb8_apr30", "2025-04-30"), ("feb8_jun30", "2025-06-30"))}
    res["share_forecast"] = cmp

    s = fare.prepare(pd.read_parquet(DERIVED / "trip_sample.parquet"))
    tr, te = s[(s["date"] >= "2024-09-01") & (s["date"] < "2025-06-01")], s[s["date"] >= "2025-06-01"]
    res["fare_model"] = fare.full_model(tr, te)
    base = fare.base_model(tr)
    res["fare_model"]["r2_base_log"] = float(base.rsquared)
    fare.price_index(s[s["date"] > "2025-02-07"], base, area).to_csv(RESULTS / "price_index.csv", index=False)

    p = forecast.hourly_panel(zh)
    bt = forecast.backtest(p)
    bt.to_csv(RESULTS / "forecast_backtest.csv", index=False)
    res["forecast_zones"] = int(p["zone"].nunique())

    alpha = wait_elasticity(zh, crz)
    res["wait_elasticity"] = alpha
    cal = calibration(zh, area)
    cal.to_csv(RESULTS / "calibration.csv")
    par = params(cal, models, max(alpha, 0.05))
    pol = []
    for eta in (0.5, 1.0, 2.0):
        for name, lam in (("fixed", 1.0), ("clearing", 1.0), ("smoothed", 0.3)):
            mt = simulate.metrics(simulate.run(par, name, eta=eta, lam=lam))
            pol.append({"eta": eta, "policy": name, **mt})
    pd.DataFrame(pol).to_csv(RESULTS / "policies.csv", index=False)
    pd.DataFrame(simulate.phase(par, np.round(np.linspace(0.25, 4, 16), 2), np.round(np.linspace(0.1, 1, 10), 2))).to_csv(
        RESULTS / "phase.csv", index=False)

    sw = experiment.block_sd(zh, crz)
    sw.to_csv(RESULTS / "switchback_sd.csv", index=False)
    res["trips_total"] = int(zh["trips"].sum())
    res["months"] = MONTHS
    (RESULTS / "summary.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return res


def cmd_export(path: str) -> None:
    """Everything the page needs, in one small JSON file."""
    zh, crz, area = load()
    r = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    cal = pd.read_csv(RESULTS / "calibration.csv", index_col=0)
    cm = pd.read_csv(RESULTS / "choice_models.csv")
    get = lambda model, term: float(cm[(cm["model"] == model) & (cm["term"] == term)]["coef"].iloc[0])
    z = zh[pd.to_datetime(zh["date"]) > pd.Timestamp("2025-02-07")].copy()
    z["dow"] = pd.to_datetime(z["date"]).dt.dayofweek
    z["weekend"] = (z["dow"] >= 5).astype(int)
    z["area"] = z["zone"].map(area)
    days = z.groupby("weekend")["date"].nunique()
    prof = z.groupby(["area", "weekend", "hour", "platform"])[["trips", "fare_sum", "pay_sum", "wait_sum", "wait_n"]].sum()
    prof = prof.reset_index()
    prof["trips"] = prof["trips"] / prof["weekend"].map(days)
    prof["fare"] = prof["fare_sum"] / (prof["trips"] * prof["weekend"].map(days))
    prof["theta"] = prof["pay_sum"] / prof["fare_sum"]
    prof["wait"] = prof["wait_sum"] / prof["wait_n"] / 60
    pi = pd.read_csv(RESULTS / "price_index.csv")
    ev = pd.read_csv(RESULTS / "event_series.csv")
    out = {
        "areas": AREAS,
        "summary": r,
        "calibration": cal.reset_index().rename(columns={"index": "area"}).to_dict("records"),
        "params": {"eps": -0.4, "eps_lo": -0.6, "eps_hi": -0.2, "alpha": max(r["wait_elasticity"], 0.05),
                   "beta_p": -get("static", "dlogp"), "beta_w": -get("static", "dwait"),
                   "rho": get("dynamic", "L_lag"), "m_min": 0.7, "m_max": 2.5},
        "profiles": prof[["area", "weekend", "hour", "platform", "trips", "fare", "theta", "wait"]].round(4).to_dict("records"),
        "price_index": pi.round(4).to_dict("records"),
        "event": ev.assign(date=ev["date"].astype(str)).round(5).to_dict("records"),
        "did": pd.read_csv(RESULTS / "did_credit.csv").round(5).to_dict("records"),
        "choice": cm.round(5).to_dict("records"),
        "forecast": pd.read_csv(RESULTS / "forecast_backtest.csv").round(5).to_dict("records"),
        "switchback": pd.read_csv(RESULTS / "switchback_sd.csv").round(5).to_dict("records"),
        "policies": pd.read_csv(RESULTS / "policies.csv").round(5).to_dict("records"),
        "phase": pd.read_csv(RESULTS / "phase.csv").round(5).to_dict("records"),
        "quality": pd.read_csv(RESULTS / "data_quality.csv").round(6).to_dict("records"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"), default=float)
    print("wrote", path)


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    cmd = argv[0] if argv else "analyse"
    if cmd == "download":
        cmd_download()
    elif cmd == "aggregate":
        aggregate.run(MONTHS)
    elif cmd == "analyse":
        print(json.dumps(cmd_analyse(), indent=2, default=float))
    elif cmd == "export":
        cmd_export(argv[1])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
