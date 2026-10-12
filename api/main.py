"""FastAPI service for next-day volatility forecasts.

    uvicorn api.main:create_app --factory --port 8000
    open http://localhost:8000          (dashboard)
    open http://localhost:8000/docs     (interactive API docs)

Environment variables:
    ARTIFACTS_DIR  folder written by `python -m src.fit_final`   (default: artifacts)
    RESULTS_DIR    folder written by `python -m src.train`       (default: results)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.inference import LABELS, Forecaster

STATIC = Path(__file__).parent / "static"
ENSEMBLE = ("lstm", "transformer", "har_x")


class ModelForecast(BaseModel):
    model: str
    label: str
    variance: float
    daily_move_pct: float
    annualised_vol_pct: float


class ForecastResponse(BaseModel):
    asset: str
    as_of: str
    for_date: str
    ensemble_annualised_vol_pct: float
    forecasts: list[ModelForecast]


class HistoryPoint(BaseModel):
    date: str
    vol_pct: float
    vol_ma5_pct: float | None


class ModelScore(BaseModel):
    model: str
    label: str
    rmse_log: float
    mae_log: float
    qlike: float
    qlike_vs_har_x_pct: float
    dm_p_value: float | None


class ModelsResponse(BaseModel):
    backtest_data_source: str
    scores: list[ModelScore]


def create_app(artifacts_dir: str | None = None, results_dir: str | None = None) -> FastAPI:
    art = artifacts_dir or os.environ.get("ARTIFACTS_DIR", "artifacts")
    res = Path(results_dir or os.environ.get("RESULTS_DIR", "results"))
    fc = Forecaster(art)

    app = FastAPI(title="Volatility Forecast API", version="1.0.0",
                  description="Next-day volatility forecasts from an LSTM, a Transformer and "
                              "classical baselines. See the README for how they were validated.")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def check_asset(asset: str) -> str:
        if asset not in fc.assets:
            raise HTTPException(404, f"Unknown asset '{asset}'. Available: {', '.join(fc.assets)}")
        return asset

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "assets": len(fc.assets), "trained_through": fc.meta["trained_through"]}

    @app.get("/api/meta")
    def meta():
        m = fc.meta
        return {"data_source": m["data_source"], "assets": m["assets"], "window_days": m["window"],
                "trained_through": m["trained_through"], "fitted_at": m["fitted_at"],
                "forecast_for": fc.next_day()}

    @app.get("/api/forecast/{asset}", response_model=ForecastResponse)
    def forecast(asset: str):
        check_asset(asset)
        out = fc.forecast(asset)
        ens = sum(out[k]["annualised_vol_pct"] for k in ENSEMBLE) / len(ENSEMBLE)
        return ForecastResponse(asset=asset, as_of=str(fc.dates[-1].date()), for_date=fc.next_day(),
                                ensemble_annualised_vol_pct=ens,
                                forecasts=[ModelForecast(**v) for v in out.values()])

    @app.get("/api/forecasts")
    def forecasts_all():
        """One-line summary per asset: the ensemble forecast, for the asset list."""
        rows = []
        for a in fc.assets:
            out = fc.forecast(a)
            rows.append({"asset": a, "ensemble_annualised_vol_pct":
                         sum(out[k]["annualised_vol_pct"] for k in ENSEMBLE) / len(ENSEMBLE)})
        return rows

    @app.get("/api/history/{asset}", response_model=list[HistoryPoint])
    def history(asset: str, days: int = Query(120, ge=10, le=250)):
        return fc.history(check_asset(asset), days)

    @app.get("/api/models", response_model=ModelsResponse)
    def models():
        f = res / "summary.csv"
        if not f.exists():
            raise HTTPException(404, "No backtest results found. Run `python -m src.train` first.")
        df = pd.read_csv(f, index_col="model")
        source = "unknown"
        cfg = res / "run_config.json"
        if cfg.exists():
            source = json.loads(cfg.read_text()).get("data", "unknown")
        scores = []
        for key, r in df.iterrows():
            p = r.get("DM_p_vs_HAR-X")
            scores.append(ModelScore(
                model=key, label=LABELS.get(key, key), rmse_log=r["RMSE_log"], mae_log=r["MAE_log"],
                qlike=r["QLIKE"], qlike_vs_har_x_pct=r["QLIKE_vs_HAR-X_%"],
                dm_p_value=None if pd.isna(p) else float(p)))
        return ModelsResponse(backtest_data_source=source, scores=scores)

    return app
