"""Fit the final models on all available data and save serving artifacts.

    python -m src.fit_final                     # synthetic demo data
    python -m src.fit_final --data yfinance     # real data; re-run daily to refresh

Layout of the data: |------- train -------|-- val --|   (val = last 200 days)
The validation window is used for early stopping and for the QLIKE scale
calibration, exactly as in the walk-forward evaluation in train.py.

Artifacts written to --out (default: artifacts/):
    meta.json         asset names, window, scaler, calibration factors, dates, data source
    recent.csv        last 260 days of returns and variance proxy for every asset
    lstm.pt, transformer.pt, har.joblib, har_x.joblib
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from . import baselines
from .data import build_features, fit_scaler, load_yfinance, make_windows, simulate_panel
from .metrics import calibration_factor
from .train import NN_MODELS, fit_nn, predict_nn

MODEL_KEYS = ["naive", "ewma", "har", "har_x", "lstm", "transformer"]


def fit_final(args) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    panel = load_yfinance() if args.data == "yfinance" else simulate_panel(
        args.n_assets, args.n_days, seed=args.seed)
    L, T, N = args.window, panel.T, panel.N
    train_end = T - args.val_len
    print(f"data={args.data} days={T} assets={N} device={device}")

    sc = fit_scaler(panel, train_end)
    feat, lrv_z = build_features(panel, sc)
    Xtr, ytr, _, _ = make_windows(feat, lrv_z, range(L, train_end), L)
    Xva, yva, dva, ava = make_windows(feat, lrv_z, range(train_end, T), L)
    rv_val = panel.rv[dva, ava]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    preds_val = {"naive": baselines.naive(Xva), "ewma": baselines.ewma(Xva, ava, sc)}
    for key, wm in [("har", False), ("har_x", True)]:
        m = baselines.HAR(with_market=wm).fit(Xtr, ytr)
        joblib.dump(m, out / f"{key}.joblib")
        preds_val[key] = m.predict(Xva)
    for name in NN_MODELS:
        net = fit_nn(name, Xtr, ytr, Xva, yva, device, args, seed=args.seed)
        torch.save(net.state_dict(), out / f"{name}.pt")
        preds_val[name] = predict_nn(net, Xva, device)

    calib = {}
    for key in MODEL_KEYS:
        lp = preds_val[key] * sc.sd[ava] + sc.mu[ava]
        calib[key] = calibration_factor(rv_val, np.exp(lp))

    n_recent = 260
    cols = {}
    for i, nm in enumerate(panel.names):
        cols[f"ret:{nm}"] = panel.ret[-n_recent:, i]
        cols[f"rv:{nm}"] = panel.rv[-n_recent:, i]
    pd.DataFrame(cols, index=panel.dates[-n_recent:]).to_csv(out / "recent.csv", index_label="date")

    meta = {
        "data_source": args.data, "assets": panel.names, "window": L,
        "trained_through": str(panel.dates[-1].date()),
        "fitted_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "scaler": {"mu": sc.mu.tolist(), "sd": sc.sd.tolist(), "ret_sd": sc.ret_sd.tolist()},
        "calibration": calib, "models": MODEL_KEYS,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved artifacts to {out}/")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["synthetic", "yfinance"], default="synthetic")
    p.add_argument("--n-assets", type=int, default=8)
    p.add_argument("--n-days", type=int, default=3000)
    p.add_argument("--window", type=int, default=22)
    p.add_argument("--val-len", type=int, default=200)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out", default="artifacts")
    fit_final(p.parse_args())


if __name__ == "__main__":
    main()
