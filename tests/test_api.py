"""API tests: fit a tiny model on synthetic data into a temp folder, then call the endpoints."""
from argparse import Namespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.fit_final import fit_final


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("art")
    fit_final(Namespace(data="synthetic", n_assets=3, n_days=600, window=22, val_len=100, epochs=2,
                        patience=2, batch=128, lr=2e-3, seed=0, device="cpu", amp=False, out=str(d)))
    res = tmp_path_factory.mktemp("res")
    pd.DataFrame({"RMSE_log": [0.7], "MAE_log": [0.5], "QLIKE": [0.25], "QLIKE_vs_HAR-X_%": [0.0],
                  "DM_stat_vs_HAR-X": [None], "DM_p_vs_HAR-X": [None]},
                 index=pd.Index(["har_x"], name="model")).to_csv(res / "summary.csv")
    return TestClient(create_app(str(d), str(res)))


def test_health_and_meta(client):
    assert client.get("/api/health").json()["status"] == "ok"
    meta = client.get("/api/meta").json()
    assert meta["assets"] == ["SIM0", "SIM1", "SIM2"] and meta["data_source"] == "synthetic"


def test_forecast_is_positive_and_complete(client):
    r = client.get("/api/forecast/SIM0")
    assert r.status_code == 200
    body = r.json()
    assert {f["model"] for f in body["forecasts"]} == {"naive", "ewma", "har", "har_x", "lstm", "transformer"}
    assert all(f["annualised_vol_pct"] > 0 for f in body["forecasts"])
    assert body["for_date"] > body["as_of"]
    # sanity: simulated assets have roughly 10-40% annualised volatility
    assert 3 < body["ensemble_annualised_vol_pct"] < 80


def test_unknown_asset_is_404(client):
    r = client.get("/api/forecast/NOPE")
    assert r.status_code == 404 and "Available" in r.json()["detail"]


def test_history_length_and_validation(client):
    assert len(client.get("/api/history/SIM1?days=50").json()) == 50
    assert client.get("/api/history/SIM1?days=5").status_code == 422


def test_models_endpoint_and_dashboard(client):
    scores = client.get("/api/models").json()["scores"]
    assert scores[0]["model"] == "har_x" and scores[0]["dm_p_value"] is None
    page = client.get("/")
    assert page.status_code == 200 and "Volatility forecasts" in page.text
