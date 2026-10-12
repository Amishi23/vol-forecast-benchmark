import numpy as np
import pandas as pd

from src.data import build_features, fit_scaler, make_windows, ohlc_to_panel, simulate_panel
from src.metrics import diebold_mariano, qlike


def test_window_never_contains_target_day():
    """Changing the target day's data must not change its own input window."""
    panel = simulate_panel(n_assets=3, n_days=200, seed=1)
    sc = fit_scaler(panel, 150)
    feat, lrv = build_features(panel, sc)
    X1, y1, d1, _ = make_windows(feat, lrv, range(100, 110), L=22)

    panel.rv[100:110] *= 50.0          # corrupt the days we are forecasting
    panel.ret[100:110] *= 50.0
    feat2, lrv2 = build_features(panel, sc)
    X2, y2, d2, _ = make_windows(feat2, lrv2, range(100, 101), L=22)

    first = d1 == 100
    # window for day 100 uses days 78..99 only -> unchanged
    assert np.allclose(X1[first][:, :, :2], X2[:, :, :2])
    assert not np.allclose(y1[first], y2)  # the target itself did change


def test_scaler_uses_only_training_days():
    panel = simulate_panel(n_assets=2, n_days=300, seed=2)
    sc_a = fit_scaler(panel, 200)
    panel.rv[200:] *= 1000.0           # change the "future"
    sc_b = fit_scaler(panel, 200)
    assert np.allclose(sc_a.mu, sc_b.mu) and np.allclose(sc_a.sd, sc_b.sd)


def test_window_shapes_and_alignment():
    panel = simulate_panel(n_assets=4, n_days=120, seed=3)
    feat, lrv = build_features(panel, fit_scaler(panel, 100))
    X, y, d, a = make_windows(feat, lrv, range(30, 40), L=22)
    assert X.shape == (10 * 4, 22, 3)
    assert np.allclose(y, lrv[d, a])
    assert np.allclose(X[0, -1, 0], lrv[29, 0])      # last input day is the day before the target


def test_ohlc_to_panel():
    idx = pd.bdate_range("2020-01-01", periods=50)
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 50))), idx)
    df = pd.DataFrame({"Close": close, "High": close * 1.01, "Low": close * 0.99})
    p = ohlc_to_panel({"A": df, "B": df})
    assert p.N == 2 and p.T == 49 and (p.rv > 0).all()


def test_dm_detects_better_forecast():
    rng = np.random.default_rng(0)
    rv = np.exp(rng.normal(size=500))
    good, bad = rv * np.exp(rng.normal(0, 0.1, 500)), rv * np.exp(rng.normal(0, 1.0, 500))
    stat, p = diebold_mariano(qlike(rv, good), qlike(rv, bad))
    assert stat < 0 and p < 0.01
