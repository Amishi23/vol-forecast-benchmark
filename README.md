# Volatility Forecasting Benchmark: Deep Learning vs. Classical Models

A PyTorch project that forecasts **next-day volatility** for a panel of assets and asks a
simple, honest question: *do LSTM and Transformer models actually beat strong classical
baselines once everything is validated properly?*

It focuses on the parts that matter in quantitative ML work: leakage-safe features,
walk-forward validation, robust loss functions, significance testing, and GPU training
(mixed precision + throughput benchmarking).

## What it does

- **Task:** predict tomorrow's log variance for each asset from the previous 22 days.
- **Features:** own log variance, own signed return (captures the leverage effect), and a
  cross-asset market volatility factor.
- **Models**
  - Baselines: Naive (yesterday), EWMA (RiskMetrics, 0.94), HAR (Ridge), and **HAR-X**
    (HAR plus the same market-factor terms, so the baseline sees the same information as the nets).
  - Neural: 1-layer LSTM and a 2-layer encoder-only Transformer.
- **Validation:** expanding-window walk-forward. Each fold trains on the past, uses the last
  200 days for early stopping and calibration, and tests on the next 250 days.
- **Metrics:** RMSE/MAE of log variance, **QLIKE** (the standard robust loss for variance
  forecasts), and a **Diebold-Mariano test** against HAR-X.
- **Leakage guards:** scaler statistics come from training days only, windows end strictly
  before the target day, and unit tests enforce both (`tests/`).
- **GPU:** `--device cuda --amp` for mixed-precision training, plus `src/benchmark.py` for
  CPU vs GPU and fp32 vs fp16 throughput.

## Product layer: API and dashboard

On top of the research pipeline there is a small serving stack:

- `src/fit_final.py` trains every model on all available data and saves artifacts
  (`artifacts/`: weights, scaler, calibration factors, the last 260 days of data).
- `api/main.py` is a **FastAPI** service that loads the artifacts and answers
  `GET /api/forecast/{asset}`, `/api/history/{asset}`, `/api/forecasts`, `/api/models`,
  `/api/meta` and `/api/health`. Interactive docs are at `/docs`.
- `api/static/index.html` is a dashboard (Chart.js) that shows an asset's recent volatility,
  the next-day forecast from each model, and the walk-forward backtest comparison.

```bash
python -m src.fit_final --data yfinance          # trains + saves artifacts/ (minutes on CPU)
uvicorn api.main:create_app --factory --port 8000
# dashboard: http://localhost:8000     API docs: http://localhost:8000/docs
```

The `artifacts/` folder in this repo holds models trained on real ETF data (see
`artifacts/meta.json` for the training date). To refresh them, re-run `fit_final` and
restart the server. Running it on a schedule (cron or a GitHub Action) would keep forecasts
current. That scheduler is not included.

```bash
curl localhost:8000/api/forecast/SPY
```

A `Dockerfile` is included (CPU build). It has not been tested yet.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q tests                 # leakage, pipeline and API tests
python -m src.train                       # synthetic data, runs anywhere
python -m src.train --data yfinance       # real data (10 ETFs, Parkinson variance proxy)
python -m src.train --device cuda --amp   # GPU + mixed precision
python -m src.benchmark                   # CPU vs GPU throughput (use a GPU runtime)
```

Outputs are written to `results/`: `summary.csv`, `qlike_by_fold.csv`, `forecasts.png`,
`run_config.json`.

### Running on Google Colab (free GPU)

1. Open a new notebook at colab.research.google.com.
2. Runtime > Change runtime type > T4 GPU.
3. Run:

```
!git clone https://github.com/Amishi23/vol-forecast-benchmark.git
%cd vol-forecast-benchmark
!pip install -q yfinance joblib
!python -m src.benchmark
```

## Results

### Synthetic data (CPU, 6 walk-forward folds, 8 assets, 3,000 days)

The data are simulated from GJR-GARCH(1,1) processes with a shared calm/stress regime, so
volatility is forecastable but noisy. **These numbers validate the pipeline. They are not
evidence about real markets.**

| Model | RMSE (log) | MAE (log) | QLIKE | QLIKE vs HAR-X | DM p-value |
|---|---|---|---|---|---|
| Naive (yesterday) | 0.910 | 0.713 | 0.434 | +81.8% | <0.001 |
| EWMA (0.94) | 0.781 | 0.600 | 0.270 | +13.4% | <0.001 |
| HAR (Ridge) | 0.722 | 0.564 | 0.253 | +5.9% | <0.001 |
| HAR-X (Ridge + market) | 0.711 | 0.556 | 0.239 | baseline | n/a |
| **LSTM** | **0.672** | **0.525** | **0.205** | **-13.9%** | <0.001 |
| Transformer | 0.683 | 0.534 | 0.216 | -9.4% | <0.001 |

The LSTM had the lowest QLIKE in all 6 folds of the synthetic run.

### Real data (10 ETFs, 5,476 trading days, 16 walk-forward folds, CPU)

```
python -m src.train --data yfinance
```

| Model | RMSE (log) | MAE (log) | QLIKE | QLIKE vs HAR-X | DM p-value |
|---|---|---|---|---|---|
| Naive (yesterday) | 0.995 | 0.791 | 0.524 | +42.2% | <0.001 |
| EWMA (0.94) | 0.910 | 0.730 | 0.412 | +11.8% | <0.001 |
| HAR (Ridge) | 0.810 | 0.644 | 0.378 | +2.5% | <0.001 |
| HAR-X (Ridge + market) | 0.803 | 0.638 | 0.369 | baseline | n/a |
| **LSTM** | **0.789** | **0.628** | **0.350** | **-5.0%** | <0.001 |
| Transformer | 0.794 | 0.631 | 0.357 | -3.3% | 0.012 |

The LSTM beats HAR-X by about 5% in QLIKE, a small but statistically significant gain. It had
lower QLIKE than HAR-X in 14 of 16 walk-forward folds (the Transformer in 13 of 16).
Single seed, limited tuning, and the variance proxy is the noisy Parkinson high-low estimator,
so treat the ranking of the two networks as tentative.

### GPU benchmark (fill in after running on a GPU)

```
python -m src.benchmark
```

| Model | Device | Precision | Samples/sec | Speedup vs CPU |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

## Design decisions

- **Why volatility, not returns?** Daily returns are close to unpredictable, so any "accuracy"
  there is usually leakage. Volatility is persistent and genuinely forecastable, which makes
  model comparisons meaningful.
- **Why HAR-X as the main baseline?** HAR is the standard strong baseline in the literature.
  Giving it the same market-factor information as the nets keeps the comparison fair.
- **Why QLIKE?** The variance proxy is noisy. QLIKE gives the same model ranking in expectation
  whether you use the true variance or a noisy proxy, unlike plain squared error.
- **Why calibrate?** Models predict in log space, which biases variance forecasts low
  (Jensen's inequality). A single scale factor fitted on the validation window fixes this
  for every model equally.
- **Why walk-forward?** Random train/test splits leak future regimes into training.

## Limitations

- Synthetic results depend on the simulator I wrote. The real-data results above are the ones
  that matter.
- The real-data variance proxy is the Parkinson high-low estimator. It is noisier than
  intraday realised variance and ignores overnight gaps, so volatility levels run lower than
  close-to-close or VIX-style figures. Changes over time and comparisons between models
  are still meaningful.
- The served models are trained on data up to the date in `artifacts/meta.json`. Re-run
  `fit_final` to refresh them.
- The dashboard loads Chart.js and a font from public CDNs, so it needs internet in your browser.
- Single seed by default. Use `--seeds 5` to ensemble the networks and reduce variance.
- Hyperparameters were not tuned extensively.
- This forecasts volatility only. It is not a trading strategy and ignores costs.

## Possible extensions

- Add intraday realised variance, or option-implied volatility, as features.
- Multi-horizon forecasts (1, 5, 22 days).
- Temporal convolutional network or gated linear baseline.
- Hyperparameter search (Optuna) with the validation window.
- Distributed or multi-GPU training with `torch.nn.parallel.DistributedDataParallel`.

## Project layout

```
src/data.py        synthetic panel, yfinance loader, leakage-safe windowing
src/models.py      LSTM and Transformer forecasters
src/baselines.py   naive, EWMA, HAR, HAR-X
src/metrics.py     QLIKE, calibration, Diebold-Mariano test
src/train.py       walk-forward training and evaluation
src/fit_final.py   fit on all data and save serving artifacts
src/inference.py   load artifacts and produce next-day forecasts
src/benchmark.py   CPU/GPU and mixed-precision throughput benchmark
api/main.py        FastAPI service
api/static/        dashboard (index.html)
artifacts/         saved models trained on real ETF data
results/           backtest outputs
tests/             leakage, pipeline and API tests
```