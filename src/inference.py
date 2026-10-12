"""Load saved artifacts and produce next-day volatility forecasts."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from . import baselines
from .data import Panel, Scaler, build_features
from .models import build_model

TRADING_DAYS = 252
LABELS = {"naive": "Naive (yesterday)", "ewma": "EWMA (0.94)", "har": "HAR (Ridge)",
          "har_x": "HAR-X (Ridge + market)", "lstm": "LSTM", "transformer": "Transformer"}


class Forecaster:
    def __init__(self, art_dir: str | Path):
        self.dir = Path(art_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        s = self.meta["scaler"]
        self.sc = Scaler(np.array(s["mu"]), np.array(s["sd"]), np.array(s["ret_sd"]))
        self.assets: list = self.meta["assets"]
        self.L: int = self.meta["window"]

        df = pd.read_csv(self.dir / "recent.csv", index_col="date", parse_dates=True)
        self.dates = df.index
        ret = df[[f"ret:{a}" for a in self.assets]].to_numpy()
        rv = df[[f"rv:{a}" for a in self.assets]].to_numpy()
        self.panel = Panel(ret, rv, self.dates, self.assets)
        self.feat, _ = build_features(self.panel, self.sc)       # (T, N, 3)

        self.har = {k: joblib.load(self.dir / f"{k}.joblib") for k in ("har", "har_x")}
        self.nets = {}
        for name in ("lstm", "transformer"):
            net = build_model(name, self.feat.shape[2], self.L)
            net.load_state_dict(torch.load(self.dir / f"{name}.pt", map_location="cpu"))
            self.nets[name] = net.eval()

    # ------------------------------------------------------------------
    def next_day(self) -> str:
        return str((self.dates[-1] + pd.offsets.BDay(1)).date())

    def forecast(self, asset: str) -> dict:
        """Return {model: {variance, daily_move_pct, annualised_vol_pct}} for the next trading day."""
        a = self.assets.index(asset)
        X = self.feat[-self.L:, a][None]                          # (1, L, F)
        ai = np.array([a])
        z = {"naive": baselines.naive(X), "ewma": baselines.ewma(X, ai, self.sc),
             "har": self.har["har"].predict(X), "har_x": self.har["har_x"].predict(X)}
        with torch.no_grad():
            for name, net in self.nets.items():
                z[name] = net(torch.from_numpy(X)).numpy()
        out = {}
        for key in self.meta["models"]:
            log_var = float(z[key][0] * self.sc.sd[a] + self.sc.mu[a])
            var = float(np.exp(log_var) * self.meta["calibration"][key])
            out[key] = {"model": key, "label": LABELS[key], "variance": var,
                        "daily_move_pct": 100 * var ** 0.5,
                        "annualised_vol_pct": 100 * (var * TRADING_DAYS) ** 0.5}
        return out

    def history(self, asset: str, days: int = 120) -> list:
        a = self.assets.index(asset)
        rv = self.panel.rv[-days:, a]
        vol = 100 * np.sqrt(rv * TRADING_DAYS)
        ma5 = pd.Series(100 * np.sqrt(self.panel.rv[:, a] * TRADING_DAYS)).rolling(5).mean().to_numpy()[-days:]
        return [{"date": str(d.date()), "vol_pct": float(v), "vol_ma5_pct": None if np.isnan(m) else float(m)}
                for d, v, m in zip(self.dates[-days:], vol, ma5)]
