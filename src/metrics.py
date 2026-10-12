"""Forecast evaluation: loss functions, scale calibration, Diebold-Mariano test."""
from __future__ import annotations

import numpy as np
from scipy import stats


def qlike(rv, fcst):
    """QLIKE loss (Patton 2011): robust to noise in the variance proxy."""
    r = rv / fcst
    return r - np.log(r) - 1.0


def calibration_factor(rv_val, fcst_val):
    """Multiplicative scale c minimising QLIKE on validation data (c = mean(rv/f)).
    Fixes the Jensen bias that appears when forecasting in log space."""
    return float(np.mean(rv_val / fcst_val))


def diebold_mariano(loss_a, loss_b, lag: int = 5):
    """DM test on a loss differential series d_t = loss_a - loss_b.
    Negative statistic -> model A has lower loss. Newey-West variance."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    dm = d - d.mean()
    var = np.sum(dm * dm) / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        var += 2 * w * np.sum(dm[k:] * dm[:-k]) / n
    stat = d.mean() / np.sqrt(var / n)
    p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return float(stat), float(p)
