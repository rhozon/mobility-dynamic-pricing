import numpy as np
import pandas as pd
import pytest

from mdp import dynamics, experiment, forecast, simulate, zones
from mdp.aggregate import month_bounds


def toy_params(**kw):
    return simulate.Params(Q=np.array([300.0, 200, 100]), p0=np.array([25.0, 20, 40]),
                           theta=np.array([0.75, 0.8, 0.8]), w0=np.array([5.0, 5, 5]),
                           s0=np.array([0.27, 0.27, 0.25]), **kw)


def test_month_bounds_cross_year():
    assert month_bounds("2024-12") == ("2024-12-01", "2025-01-01")


def test_logit_round_trip():
    s = np.array([0.1, 0.27, 0.9])
    assert np.allclose(dynamics.expit(dynamics.logit(s)), s)


def test_credit_flag_window_and_placebo():
    d = pd.Series(pd.to_datetime(["2025-01-04", "2025-01-05", "2025-02-07", "2025-02-08", "2024-01-20"]))
    assert dynamics.credit_flag(d).tolist() == [0, 1, 1, 0, 0]
    assert dynamics.credit_flag(d, year_shift=1).tolist() == [0, 0, 0, 0, 1]


def test_crz_zones_uses_fee_share():
    zh = pd.DataFrame({"date": ["2025-03-01"] * 3, "zone": [1, 2, 3],
                       "trips": [5000, 5000, 50], "fee_trips": [4990, 1500, 50]})
    assert zones.crz_zones(zh) == {1}


def test_simulator_rest_point_without_shock():
    par = toy_params()
    r = simulate.run(par, "clearing", eta=1.0, shock=0.0, days=40)
    assert np.allclose(r["m"], 1.0)
    assert np.allclose(r["x"][-1], par.x0)
    assert np.allclose(r["w"][-1], par.w0)


@pytest.mark.parametrize("policy", ["fixed", "clearing", "smoothed"])
def test_simulator_shares_stay_on_simplex(policy):
    r = simulate.run(toy_params(), policy, eta=3.0, lam=0.3, days=60)
    assert np.allclose(r["x"].sum(axis=1), 1.0)
    assert (r["x"] > 0).all() and (r["s"] > 0).all() and (r["s"] < 1).all()
    assert np.isfinite(r["margin"]).all()


def test_clearing_price_restores_usual_wait_on_the_day():
    par = toy_params()
    r = simulate.run(par, "clearing", eta=0.2, shock=0.25, days=3)
    # on day 0 drivers have not moved yet, so the clearing price alone brings the wait back
    assert np.allclose(r["w"][0], par.w0)
    assert r["m"][0][0] > 1.0


def test_smoothing_damps_price_swings():
    par = toy_params()
    fast = simulate.metrics(simulate.run(par, "clearing", eta=2.0))
    calm = simulate.metrics(simulate.run(par, "smoothed", eta=2.0, lam=0.2))
    assert calm["price_swing"] < fast["price_swing"]


def test_mde_falls_with_more_weeks():
    assert experiment.mde(0.2, 4, 4) < experiment.mde(0.2, 2, 4)
    assert experiment.mde(0.2, 2, 4) == pytest.approx(np.expm1(experiment.Z * 0.2 * np.sqrt(4 / 84)))


def test_forecast_features_only_use_lags_of_a_week_or_more():
    ts = pd.date_range("2024-09-01", periods=24 * 40, freq="h")
    zh = pd.DataFrame({"date": ts.date, "hour": ts.hour, "zone": 7, "platform": "uber",
                       "trips": np.arange(len(ts)) % 24 + 100})
    p = forecast.hourly_panel(zh, min_trips_day=0)
    row = p.iloc[0]
    assert row["lag168"] == p.set_index("ts")["y"].get(row["ts"] - pd.Timedelta(hours=168), row["lag168"])
    assert min(forecast.LAGS) >= 168


def test_static_and_dynamic_models_recover_inertia():
    rng = np.random.default_rng(0)
    days = pd.date_range("2024-09-01", "2025-04-30")
    rows = []
    for z in range(1, 9):
        L = -1.0
        for t in days:
            dlogp, dwait = rng.normal(0, 0.05), rng.normal(0, 0.5)
            target = -1.0 - 0.3 * dlogp - 0.08 * dwait
            L = 0.6 * L + 0.4 * target + rng.normal(0, 0.02)
            s = dynamics.expit(L)
            rows.append({"date": t, "zone": z, "L": L, "share": s, "dlogp": dlogp, "dwait": dwait,
                         "dow": t.dayofweek, "trips": 1000, "trips_lyft": 1000 * s, "trips_uber": 1000 * (1 - s)})
    d = pd.DataFrame(rows)
    m = dynamics.fit_models(d, set(range(1, 9)))
    assert m["dynamic"].params["L_lag"] == pytest.approx(0.6, abs=0.05)
    fc = dynamics.forecast(m)
    c = dynamics.compare(fc)
    assert c["rmse_dynamic"] <= c["rmse_static"] * 1.05


def test_psi_is_zero_for_the_same_distribution_and_grows_with_a_shift():
    from mdp import drift
    rng = np.random.default_rng(5)
    a = rng.normal(0, 1, 50_000)
    assert drift.psi(a, rng.normal(0, 1, 50_000)) < 0.01
    assert drift.psi(a, rng.normal(0.5, 1, 50_000)) > 0.1
