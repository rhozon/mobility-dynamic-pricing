"""Switchback experiment planning from the observed variability of the platform's margin.

In a switchback test the whole area alternates between the old and the new price policy in
blocks of b hours, assigned at random. With n blocks split evenly, the minimum detectable effect
on log margin at 5% (two-sided) and 80% power is (1.96 + 0.84) * sd_b * sqrt(4 / n), where sd_b
is the standard deviation of log block margin after removing the usual hour-of-week level.
Longer blocks are noisier per block in absolute terms but less exposed to carryover, the
drivers who are still repositioning from the previous block.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Z = 1.959964 + 0.841621


def block_sd(zone_hour: pd.DataFrame, zones: set[int], platform: str = "lyft",
             start: str = "2025-02-08", blocks=(1, 2, 3, 4, 6, 8, 12, 24)) -> pd.DataFrame:
    z = zone_hour[(zone_hour["zone"].isin(zones)) & (zone_hour["platform"] == platform)
                  & (pd.to_datetime(zone_hour["date"]) >= pd.Timestamp(start))]
    h = z.assign(margin=z["fare_sum"] - z["pay_sum"])
    h["ts"] = pd.to_datetime(h["date"]) + pd.to_timedelta(h["hour"], unit="h")
    y = h.groupby("ts")["margin"].sum().asfreq("h", fill_value=0)
    how = y.index.dayofweek * 24 + y.index.hour
    usual = y.groupby(how).transform("mean")
    rows = []
    for b in blocks:
        t0 = y.index[0]
        idx = ((y.index - t0) / pd.Timedelta(hours=1)).astype(int) // b
        num, den = y.groupby(idx).sum(), usual.groupby(idx).sum()
        n_full = y.groupby(idx).size() == b
        r = np.log(num[n_full] / den[n_full])
        r = r[np.isfinite(r)]  # an hour with no trips (the clock change) has no log margin
        rows.append({"block_hours": b, "sd_log": float(r.std()), "blocks": int(n_full.sum())})
    return pd.DataFrame(rows)


def mde(sd_log: float, weeks: float, block_hours: int) -> float:
    n = weeks * 7 * 24 / block_hours
    return float(np.expm1(Z * sd_log * np.sqrt(4 / n)))
