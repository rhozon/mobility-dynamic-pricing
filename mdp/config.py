"""Paths and constants shared by the pipeline."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# Raw monthly files are large (about 500 MB each), so they live outside the repository.
RAW = Path(os.environ.get("MDP_RAW", "C:/Users/rodri/data/nyc-tlc/raw"))
# Aggregates are too large for git (millions of zone-hour rows); small result tables go to results/.
DERIVED = Path(os.environ.get("MDP_DERIVED", str(RAW.parent / "derived")))

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_{month}.parquet"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

# December to June around the January 2025 change, and the same calendar a year earlier.
MONTHS = [
    "2023-12", "2024-01", "2024-02", "2024-03", "2024-04",
    "2024-09", "2024-10", "2024-11", "2024-12",
    "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
]

PLATFORMS = {"HV0003": "uber", "HV0005": "lyft"}

# Congestion Relief Zone fee started on 5 January 2025. Lyft credited riders the $1.50
# fee on rides in the zone from 5 to 31 January 2025; each credit was valid for one week.
FEE_START = "2025-01-05"
CREDIT_START = "2025-01-05"
CREDIT_EARN_END = "2025-01-31"
CREDIT_USE_END = "2025-02-07"

# Waits above one hour are treated as scheduled rides, not on-demand requests.
MAX_WAIT_S = 3600
