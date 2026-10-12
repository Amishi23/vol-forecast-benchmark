"""Data utilities: a synthetic multi-asset volatility panel, a Yahoo Finance
loader, and leakage-safe windowing.

Target of the whole project: forecast tomorrow's log daily variance proxy
(log rv_t) for every asset, using only information up to today.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "XLE", "XLF", "XLK"]


@dataclass
class Panel:
    """Daily panel. ret and rv have shape (T, N)."""
    ret: np.ndarray          # daily log returns
    rv: np.ndarray           # daily variance proxy (> 0)
    dates: pd.DatetimeIndex
    names: list

    @property
    def T(self) -> int:
        return self.ret.shape[0]

    @property
    def N(self) -> int:
        return self.ret.shape[1]


# --------------------------------------------------------------------------
# Synthetic data (used when there is no internet, and for unit tests)
# --------------------------------------------------------------------------
def simulate_panel(n_assets: int = 8, n_days: int = 3000, seed: int = 0) -> Panel:
    """GJR-GARCH(1,1) returns for each asset, scaled by a shared two-state
    market regime (calm / stress). The variance proxy is the true conditional
    variance times chi-square noise, which mimics a noisy realised-variance
    estimator. Volatility is therefore forecastable, but only with noise.
    """
    rng = np.random.default_rng(seed)

    # shared market regime (Markov chain): 0 = calm, 1 = stress
    stay = np.array([0.995, 0.97])
    regime = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        regime[t] = regime[t - 1] if rng.random() < stay[regime[t - 1]] else 1 - regime[t - 1]
    mult = np.where(regime == 1, 2.5, 1.0)

    ret = np.zeros((n_days, n_assets))
    rv = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        uncond = rng.uniform(0.8e-4, 2.5e-4)               # daily variance level
        alpha, gamma, beta = 0.04, 0.06, rng.uniform(0.86, 0.90)
        persistence = alpha + gamma / 2 + beta
        omega = uncond * (1 - persistence)
        h = uncond
        r_prev = 0.0
        for t in range(n_days):
            h = omega + (alpha + gamma * (r_prev < 0)) * r_prev ** 2 + beta * h
            sigma2 = h * mult[t]
            z = rng.standard_t(6) / np.sqrt(6 / 4)         # unit-variance Student-t
            r = np.sqrt(sigma2) * z
            ret[t, i] = r
            rv[t, i] = sigma2 * rng.chisquare(6) / 6       # noisy variance proxy
            r_prev = r
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    return Panel(ret, rv, dates, [f"SIM{i}" for i in range(n_assets)])


# --------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------
def ohlc_to_panel(frames: dict) -> Panel:
    """Convert {ticker: OHLC DataFrame} to a Panel.

    Variance proxy = Parkinson range estimator: (ln(High/Low))^2 / (4 ln 2).
    """
    rets, rvs = {}, {}
    for name, df in frames.items():
        close, hi, lo = df["Close"], df["High"], df["Low"]
        rets[name] = np.log(close).diff()
        rvs[name] = (np.log(hi / lo) ** 2) / (4 * np.log(2))
    ret = pd.DataFrame(rets).dropna()
    rv = pd.DataFrame(rvs).reindex(ret.index).clip(lower=1e-8)   # avoid log(0) on flat days
    ok = rv.notna().all(axis=1)
    ret, rv = ret[ok], rv[ok]
    return Panel(ret.values, rv.values, pd.DatetimeIndex(ret.index), list(ret.columns))


def load_yfinance(tickers=None, start="2005-01-01", end=None) -> Panel:
    import yfinance as yf

    tickers = tickers or DEFAULT_TICKERS
    frames = {}
    for tk in tickers:
        df = yf.download(tk, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):          # newer yfinance versions
            df.columns = df.columns.get_level_values(0)
        if len(df) > 500:
            frames[tk] = df[["Close", "High", "Low"]]
    return ohlc_to_panel(frames)


# --------------------------------------------------------------------------
# Features + windowing (the part where leakage bugs hide)
# --------------------------------------------------------------------------
@dataclass
class Scaler:
    mu: np.ndarray       # (N,) mean of log rv over TRAIN days only
    sd: np.ndarray       # (N,) std of log rv over TRAIN days only
    ret_sd: np.ndarray   # (N,) std of returns over TRAIN days only


def fit_scaler(panel: Panel, train_end: int) -> Scaler:
    """Statistics are computed on days [0, train_end) only."""
    lrv = np.log(panel.rv[:train_end])
    return Scaler(lrv.mean(0), lrv.std(0) + 1e-9, panel.ret[:train_end].std(0) + 1e-9)


def build_features(panel: Panel, sc: Scaler):
    """Return feat (T, N, 3) and normalised log-variance lrv_z (T, N).

    feat[t] only contains information known at the END of day t:
      0: own normalised log variance
      1: own scaled return (signed, lets the model learn the leverage effect)
      2: market factor = cross-sectional mean of normalised log variance
    """
    lrv_z = (np.log(panel.rv) - sc.mu) / sc.sd
    ret_z = panel.ret / sc.ret_sd
    mkt = np.repeat(lrv_z.mean(1, keepdims=True), panel.N, axis=1)
    feat = np.stack([lrv_z, ret_z, mkt], axis=-1).astype(np.float32)
    return feat, lrv_z.astype(np.float32)


def make_windows(feat: np.ndarray, lrv_z: np.ndarray, days, L: int):
    """One sample per (target day d, asset).

    X = feat[d-L : d]  -> days d-L .. d-1  (strictly before the target day)
    y = lrv_z[d]       -> the day being forecast
    Returns X (n, L, F), y (n,), day index (n,), asset index (n,).
    """
    days = [d for d in days if d >= L]
    T, N, F = feat.shape
    X = np.stack([feat[d - L:d] for d in days])             # (D, L, N, F)
    X = X.transpose(0, 2, 1, 3).reshape(-1, L, F)           # (D*N, L, F)
    y = lrv_z[days].reshape(-1)
    day_idx = np.repeat(days, N)
    asset_idx = np.tile(np.arange(N), len(days))
    return X, y.astype(np.float32), day_idx, asset_idx
