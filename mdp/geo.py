"""Taxi zone outlines for the page map, simplified and kept in the source projection.

The TLC shapefile is in New York State Plane (feet). The map only needs relative positions, so the
coordinates are simplified, rounded to 50 feet and written as they are: no reprojection needed.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import shapefile
from shapely.geometry import shape

from .config import RAW


def zones_geojson(out: Path, raw: Path = RAW, tolerance: float = 250.0) -> int:
    z = raw / "taxi_zones.zip"
    d = raw / "taxi_zones"
    if not d.exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(d)
    shp = next(d.rglob("*.shp"))
    r = shapefile.Reader(str(shp))
    fields = [f[0] for f in r.fields[1:]]
    feats = []
    for sr in r.shapeRecords():
        rec = dict(zip(fields, sr.record))
        g = shape(sr.shape.__geo_interface__).simplify(tolerance, preserve_topology=True)
        polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
        rings = [[[round(x / 50) * 50, round(y / 50) * 50] for x, y in p.exterior.coords] for p in polys if p.area > 1e5]
        feats.append({"id": int(rec["LocationID"]), "zone": rec["zone"], "borough": rec["borough"], "rings": rings})
    out.write_text(json.dumps(feats, separators=(",", ":")), encoding="utf-8")
    return len(feats)
