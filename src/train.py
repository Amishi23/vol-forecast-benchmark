"""Walk-forward evaluation of volatility forecasters.

Expanding-window scheme (nothing from the future is ever used):

    fold k:  |---------- train ----------|-- val --|-- test --|
             0                          e-val     e          e+test_len

 * train part : fit models (Ridge fit / neural net weights / scaler statistics)
 * val part   : early stopping (nets) and QLIKE scale calibration (all models)
 * test part  : out-of-sample scores, then the window expands and we refit.

Usage:
    python -m src.train                        # synthetic data
    python -m src.train --data yfinance        # real data (needs internet)
    python -m src.train --device cuda --amp    # GPU + mixed precision
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import baselines
from .data import build_features, fit_scaler, load_yfinance, make_windows, simulate_panel
from .metrics import calibration_factor, diebold_mariano, qlike
from .models import build_model

NN_MODELS = ["lstm", "transformer"]
ALL_MODELS = ["naive", "ewma", "har", "har_x"] + NN_MODELS
LABELS = {"naive": "Naive (yesterday)", "ewma": "EWMA (0.94)", "har": "HAR (Ridge)",
          "har_x": "HAR-X (Ridge + market)", "lstm": "LSTM", "transformer": "Transformer"}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_nn(model, X, device, bs: int = 4096):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out)


def fit_nn(name, Xtr, ytr, Xva, yva, device, args, seed):
    set_seed(seed)
    model = build_model(name, Xtr.shape[2], Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    Xt, yt = torch.from_numpy(Xtr).to(device), torch.from_numpy(ytr).to(device)
    best, best_state, bad = float("inf"), None, 0

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = nn.functional.mse_loss(model(Xt[idx]).float(), yt[idx])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        val = float(np.mean((predict_nn(model, Xva, device) - yva) ** 2))
        if val < best - 1e-5:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model


def run(args):
    t_start = time.time()
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    panel = (load_yfinance() if args.data == "yfinance"
             else simulate_panel(args.n_assets, args.n_days, seed=args.seed))
    L, T, N = args.window, panel.T, panel.N
    print(f"data={args.data}  days={T}  assets={N}  window={L}  device={device}")

    rows = []                                                # one row per test sample
    fold_starts = list(range(args.init_train, T - 1, args.test_len))
    for k, e in enumerate(fold_starts):
        test_days = range(e, min(e + args.test_len, T))
        val_days = range(e - args.val_len, e)
        train_days = range(L, e - args.val_len)

        sc = fit_scaler(panel, e - args.val_len)             # statistics from train part only
        feat, lrv_z = build_features(panel, sc)
        Xtr, ytr, _, _ = make_windows(feat, lrv_z, train_days, L)
        Xva, yva, _, ava = make_windows(feat, lrv_z, val_days, L)
        Xte, yte, dte, ate = make_windows(feat, lrv_z, test_days, L)

        preds_val, preds_te = {}, {}
        preds_val["naive"], preds_te["naive"] = baselines.naive(Xva), baselines.naive(Xte)
        preds_val["ewma"], preds_te["ewma"] = baselines.ewma(Xva, ava, sc), baselines.ewma(Xte, ate, sc)
        for key, wm in [("har", False), ("har_x", True)]:
            m = baselines.HAR(with_market=wm).fit(Xtr, ytr)
            preds_val[key], preds_te[key] = m.predict(Xva), m.predict(Xte)
        for name in NN_MODELS:
            ps_v, ps_t = [], []
            for s in range(args.seeds):
                m = fit_nn(name, Xtr, ytr, Xva, yva, device, args, seed=args.seed + s)
                ps_v.append(predict_nn(m, Xva, device))
                ps_t.append(predict_nn(m, Xte, device))
            preds_val[name], preds_te[name] = np.mean(ps_v, 0), np.mean(ps_t, 0)

        rv_val = panel.rv[np.repeat(list(val_days), N), np.tile(np.arange(N), len(val_days))]
        rv_te = panel.rv[dte, ate]
        for name in ALL_MODELS:
            to_log = lambda z, a: z * sc.sd[a] + sc.mu[a]
            lp_val, lp_te = to_log(preds_val[name], ava), to_log(preds_te[name], ate)
            c = calibration_factor(rv_val, np.exp(lp_val))
            f_te = np.exp(lp_te) * c
            rows.append(pd.DataFrame({
                "fold": k, "model": name, "day": dte, "asset": ate,
                "log_true": np.log(rv_te), "log_pred": lp_te,
                "qlike": qlike(rv_te, f_te)}))
        print(f"fold {k + 1}/{len(fold_starts)} done  (test days {test_days.start}-{test_days.stop - 1})"
              f"  elapsed {time.time() - t_start:.0f}s")

    df = pd.concat(rows, ignore_index=True)
    return panel, df, time.time() - t_start


def summarise(df, out: Path):
    df = df.assign(se=(df.log_true - df.log_pred) ** 2, ae=(df.log_true - df.log_pred).abs())
    summary = df.groupby("model").agg(RMSE_log=("se", lambda s: np.sqrt(s.mean())),
                                      MAE_log=("ae", "mean"), QLIKE=("qlike", "mean"))
    # Diebold-Mariano on the cross-sectional mean loss per day, against HAR-X
    daily = df.groupby(["model", "day"]).qlike.mean().unstack(0)
    ref = "har_x"
    dm_stat, dm_p = {}, {}
    for m in ALL_MODELS:
        dm_stat[m], dm_p[m] = (np.nan, np.nan) if m == ref else diebold_mariano(daily[m], daily[ref])
    summary["QLIKE_vs_HAR-X_%"] = 100 * (summary.QLIKE / summary.loc[ref, "QLIKE"] - 1)
    summary["DM_stat_vs_HAR-X"] = pd.Series(dm_stat)
    summary["DM_p_vs_HAR-X"] = pd.Series(dm_p)
    summary = summary.loc[ALL_MODELS]
    by_fold = df.groupby(["fold", "model"]).qlike.mean().unstack(1)[ALL_MODELS]

    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv")
    by_fold.to_csv(out / "qlike_by_fold.csv")
    return summary, by_fold


def plot(panel, df, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(9, 3.6))
    a = 0
    base = df[(df.asset == a) & (df.model == "naive")].sort_values("day")
    ax.plot(base.day, base.log_true, color="0.8", lw=0.8, label="log variance proxy (truth)")
    for m, c in [("har_x", "tab:blue"), ("lstm", "tab:red")]:
        s = df[(df.asset == a) & (df.model == m)].sort_values("day")
        ax.plot(s.day, s.log_pred, color=c, lw=1.2, label=LABELS[m])
    ax.set_title(f"Out-of-sample 1-day-ahead forecasts, asset {panel.names[a]}")
    ax.set_xlabel("day index"); ax.set_ylabel("log variance"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "forecasts.png", dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["synthetic", "yfinance"], default="synthetic")
    p.add_argument("--n-assets", type=int, default=8)
    p.add_argument("--n-days", type=int, default=3000)
    p.add_argument("--window", type=int, default=22)
    p.add_argument("--init-train", type=int, default=1500)
    p.add_argument("--val-len", type=int, default=200)
    p.add_argument("--test-len", type=int, default=250)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seeds", type=int, default=1, help="ensemble size for neural nets")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    panel, df, secs = run(args)
    out = Path(args.out)
    summary, by_fold = summarise(df, out)
    plot(panel, df, out)
    (out / "run_config.json").write_text(json.dumps({**vars(args), "seconds": round(secs, 1)}, indent=2))
    pd.set_option("display.width", 220, "display.max_columns", None, "display.float_format", "{:.4f}".format)
    print("\n=== Out-of-sample summary (lower is better) ===")
    print(summary.rename(index=LABELS))
    print("\n=== QLIKE by fold ===")
    print(by_fold.rename(columns=LABELS))


if __name__ == "__main__":
    main()
