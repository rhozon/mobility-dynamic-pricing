"""Taxi zones, the Congestion Relief Zone and the six areas used by the simulator."""
from __future__ import annotations

import pandas as pd

AREAS = ["Congestion zone", "Manhattan north", "Brooklyn", "Queens", "Bronx and Staten Island", "Airports"]
AIRPORT_ZONES = {1, 132, 138}  # Newark, JFK, LaGuardia


def crz_zones(zone_hour: pd.DataFrame, min_share: float = 0.9, start: str = "2025-02-01") -> set[int]:
    """Pickup zones where at least `min_share` of trips paid the congestion fee.

    Every trip that starts inside the zone pays it, so the inner zones sit near 100%. Trips that
    only end in or cross the zone also pay, which keeps zones outside well below the cut.
    """
    z = zone_hour[pd.to_datetime(zone_hour["date"]) >= pd.Timestamp(start)]
    g = z.groupby("zone")[["fee_trips", "trips"]].sum()
    share = g["fee_trips"] / g["trips"]
    return set(int(i) for i in share[(share >= min_share) & (g["trips"] >= 1000)].index)


def area_of(lookup: pd.DataFrame, crz: set[int]) -> pd.Series:
    """Map LocationID to one of the six areas."""
    def one(row) -> str | None:
        z, b = int(row["LocationID"]), row["Borough"]
        if z in AIRPORT_ZONES:
            return "Airports"
        if z in crz:
            return "Congestion zone"
        if b == "Manhattan":
            return "Manhattan north"
        if b in ("Brooklyn", "Queens"):
            return b
        if b in ("Bronx", "Staten Island"):
            return "Bronx and Staten Island"
        return None
    return pd.Series(lookup.apply(one, axis=1).values, index=lookup["LocationID"].astype(int))
