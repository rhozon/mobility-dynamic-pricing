import pytest
from fastapi.testclient import TestClient

from mdp.api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_zero_change_reproduces_the_baseline():
    r = client.post("/quote", json={"area": "Brooklyn", "weekend": 0, "hour": 18, "price_change": 0.0}).json()
    assert r["quote"]["trips_per_hour"] == pytest.approx(r["baseline"]["trips_per_hour"])
    assert r["margin_change"] == pytest.approx(0.0)


def test_higher_price_means_fewer_trips():
    r = client.post("/quote", json={"area": "Congestion zone", "weekend": 0, "hour": 18, "price_change": 0.2}).json()
    assert r["quote"]["trips_per_hour"] < r["baseline"]["trips_per_hour"]
    assert r["margin_change_range"][0] <= r["margin_change"] <= r["margin_change_range"][1]


def test_unknown_area_is_404():
    assert client.post("/quote", json={"area": "Atlantis", "hour": 18, "price_change": 0.1}).status_code == 404


def test_short_supply_raises_the_multiplier_and_smoothing_damps_it():
    r = client.post("/multiplier", json={"drivers_vs_usual": 0.8, "previous_multiplier": 1.0, "smoothing": 0.3}).json()
    assert r["clearing"] > 1 and 1 < r["smoothed"] < r["clearing"]
