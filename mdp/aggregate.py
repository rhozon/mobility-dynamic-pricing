"""Monthly aggregation of the high-volume for-hire trip records with DuckDB.

Each monthly file has about 20 million trips. One pass per month writes:
- zone-hour-platform aggregates (trips, fares, driver pay, waits, fee incidence);
- a data-quality row (null and invalid shares, out-of-month pickups);
- a 0.5% Bernoulli sample of valid trips for the fare model.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .config import DERIVED, MAX_WAIT_S, PLATFORMS, RAW

VALID = (
    "hvfhs_license_num IN ('HV0003', 'HV0005') "
    "AND base_passenger_fare > 0 AND trip_miles > 0 AND trip_time > 0 "
    "AND pickup_datetime >= TIMESTAMP '{start}' AND pickup_datetime < TIMESTAMP '{end}'"
)
PLATFORM_SQL = "CASE hvfhs_license_num " + " ".join(
    f"WHEN '{k}' THEN '{v}'" for k, v in PLATFORMS.items()) + " END"
WAIT_SQL = "date_diff('second', request_datetime, pickup_datetime)"


def month_bounds(month: str) -> tuple[str, str]:
    start = pd.Timestamp(month + "-01")
    end = start + pd.offsets.MonthBegin(1)
    return str(start.date()), str(end.date())


def _columns(con: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    return set(con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}')").df()["column_name"])


def aggregate_month(month: str, raw: Path = RAW, con: duckdb.DuckDBPyConnection | None = None,
                    sample_pct: float = 0.5) -> dict[str, pd.DataFrame]:
    path = raw / f"fhvhv_tripdata_{month}.parquet"
    con = con or duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    cols = _columns(con, path)
    fee = "cbd_congestion_fee" if "cbd_congestion_fee" in cols else "0.0"
    start, end = month_bounds(month)
    where = VALID.format(start=start, end=end)
    src = f"read_parquet('{path.as_posix()}')"
    wait_ok = f"{WAIT_SQL} BETWEEN 0 AND {MAX_WAIT_S}"

    zh = con.execute(f"""
        SELECT CAST(pickup_datetime AS DATE) AS date, hour(pickup_datetime) AS hour,
               PULocationID AS zone, {PLATFORM_SQL} AS platform,
               count(*) AS trips,
               sum(base_passenger_fare) AS fare_sum,
               sum(driver_pay) AS pay_sum,
               sum(trip_miles) AS miles_sum,
               sum(trip_time) / 60.0 AS minutes_sum,
               sum(CASE WHEN {wait_ok} THEN {WAIT_SQL} END) AS wait_sum,
               count(CASE WHEN {wait_ok} THEN 1 END) AS wait_n,
               sum(CASE WHEN {fee} > 0 THEN 1 ELSE 0 END) AS fee_trips
        FROM {src} WHERE {where}
        GROUP BY ALL
    """).df()

    q = con.execute(f"""
        SELECT '{month}' AS month, count(*) AS rows,
               avg(CASE WHEN hvfhs_license_num = 'HV0003' THEN 1.0 ELSE 0 END) AS share_uber,
               avg(CASE WHEN hvfhs_license_num = 'HV0005' THEN 1.0 ELSE 0 END) AS share_lyft,
               avg(CASE WHEN request_datetime IS NULL THEN 1.0 ELSE 0 END) AS null_request,
               avg(CASE WHEN on_scene_datetime IS NULL THEN 1.0 ELSE 0 END) AS null_on_scene,
               avg(CASE WHEN hvfhs_license_num = 'HV0005' AND on_scene_datetime IS NULL THEN 1.0
                        WHEN hvfhs_license_num = 'HV0005' THEN 0 END) AS null_on_scene_lyft,
               avg(CASE WHEN base_passenger_fare <= 0 THEN 1.0 ELSE 0 END) AS nonpositive_fare,
               avg(CASE WHEN trip_miles <= 0 THEN 1.0 ELSE 0 END) AS zero_miles,
               avg(CASE WHEN {WAIT_SQL} < 0 THEN 1.0 ELSE 0 END) AS negative_wait,
               avg(CASE WHEN {WAIT_SQL} > {MAX_WAIT_S} THEN 1.0 ELSE 0 END) AS wait_over_1h,
               avg(CASE WHEN pickup_datetime < TIMESTAMP '{start}' OR pickup_datetime >= TIMESTAMP '{end}'
                        THEN 1.0 ELSE 0 END) AS out_of_month,
               avg(CASE WHEN PULocationID IN (264, 265) OR PULocationID IS NULL THEN 1.0 ELSE 0 END) AS unknown_zone,
               avg(CASE WHEN {where} THEN 1.0 ELSE 0 END) AS kept
        FROM {src}
    """).df()

    sample = con.execute(f"""
        SELECT {PLATFORM_SQL} AS platform, CAST(pickup_datetime AS DATE) AS date,
               hour(pickup_datetime) AS hour, dayofweek(pickup_datetime) AS dow,
               PULocationID AS pu, DOLocationID AS do_zone,
               trip_miles AS miles, trip_time / 60.0 AS minutes,
               base_passenger_fare AS fare, driver_pay AS pay,
               CASE WHEN {wait_ok} THEN {WAIT_SQL} END AS wait_s,
               airport_fee, {fee} AS cbd_fee
        FROM {src}
        WHERE {where}
        USING SAMPLE {sample_pct}% (bernoulli, 2026)
    """).df()
    return {"zone_hour": zh, "quality": q, "sample": sample}


def run(months: list[str], raw: Path = RAW, out: Path = DERIVED) -> None:
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    zh, q, s = [], [], []
    for m in months:
        r = aggregate_month(m, raw, con)
        zh.append(r["zone_hour"]); q.append(r["quality"]); s.append(r["sample"])
        print(m, "trips kept:", int(r["zone_hour"]["trips"].sum()), flush=True)
    pd.concat(zh).sort_values(["date", "hour", "zone", "platform"]).to_parquet(out / "zone_hour.parquet", index=False)
    pd.concat(q).to_csv(out / "quality.csv", index=False)
    pd.concat(s).to_parquet(out / "trip_sample.parquet", index=False)
