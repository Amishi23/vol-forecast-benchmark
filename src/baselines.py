"""Classical volatility baselines. All work on the same windows X (n, L, F) as the
neural nets and return predictions in normalised log-variance space."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def naive(X, asset_idx=None, sc=None):
    """Tomorrow's variance = today's variance."""
    return X[:, -1, 0].copy()


def ewma(X, asset_idx, sc, lam: float = 0.94):
    """RiskMetrics EWMA of the variance proxy, truncated to the window length."""
    L = X.shape[1]
    w = lam ** np.arange(L - 1, -1, -1)                    # most recent day gets weight 1
    w = w / w.sum()
    lrv = X[:, :, 0] * sc.sd[asset_idx, None] + sc.mu[asset_idx, None]
    f = (np.exp(lrv) * w).sum(1)
    return ((np.log(f) - sc.mu[asset_idx]) / sc.sd[asset_idx]).astype(np.float32)


def _har_features(X, with_market: bool):
    own = X[:, :, 0]
    cols = [own[:, -1], own[:, -5:].mean(1), own.mean(1)]   # daily / weekly / monthly
    if with_market:
        mkt = X[:, :, 2]
        cols += [mkt[:, -1], mkt[:, -5:].mean(1), mkt.mean(1)]
    return np.stack(cols, axis=1)


class HAR:
    """Heterogeneous AutoRegressive model (Corsi 2009) fitted with Ridge.
    with_market=True adds the same daily/weekly/monthly terms for the market
    factor, so the baseline sees the same information as the neural nets."""

    def __init__(self, with_market: bool = False, alpha: float = 1.0):
        self.with_market = with_market
        self.model = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(_har_features(X, self.with_market), y)
        return self

    def predict(self, X):
        return self.model.predict(_har_features(X, self.with_market)).astype(np.float32)
