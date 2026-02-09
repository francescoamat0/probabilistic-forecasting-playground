# Probabilistic Forecasting Playground

A single notebook that walks through modern probabilistic forecasting methods end-to-end: fitting models, getting prediction intervals, evaluating calibration, and comparing everything side by side.

The goal is not to build a production system but to have a self-contained reference you can run, poke at, and learn from. Each section fits a different model on the same dataset, extracts quantile forecasts (P10/P50/P90), and evaluates them with the same metrics so you can see where each approach shines and where it breaks down.

## Dataset

By default the notebook runs on a **synthetic daily time series** (900 observations) that is deliberately complex: heteroscedastic noise, heavy tails (Student-t), outliers, a structural break, holiday effects, and covariate interactions. This is useful because it forces different methods to reveal genuine strengths and weaknesses; a clean sinusoid would make everything look good.

You can flip `USE_REAL_DATA = True` to switch to **ETTh1** (electricity transformer temperature), a standard research benchmark, if you want to see how things behave on real data.

## What's covered

The notebook is organized as a progression of several models:

1. **SARIMAX** — Classical state-space model via statsmodels. Solid interpretable baseline. Prediction intervals come from the Gaussian state-space machinery, which means they can be miscalibrated when residuals aren't well-behaved.

2. **LightGBM quantile regression** — The industry workhorse. Train one model per quantile using lag/rolling features + covariates. Tends to be accurate but can suffer on coverage.

3. **NGBoost** — Distributional boosting: instead of predicting quantiles, it fits the parameters of a distribution (Normal) via natural-gradient boosting. Gives a coherent distribution (no quantile crossing) but distributional assumptions matter.

4. **Split-conformal prediction** — A calibration layer on top of any point model (here HistGradientBoostingRegressor). Distribution-free coverage guarantees in ~20 lines of code.

5. **DeepAR** (GluonTS) — Canonical deep probabilistic model. Designed for training across many series simultaneously; the notebook runs it on a single series to show where it struggles (and why).

6. **Temporal Fusion Transformer** (PyTorch Forecasting) — Deep model purpose-built for multi-horizon forecasting with known/unknown covariates and quantile outputs. Heavier than GBDT but handles complex covariate structures.

7. **Chronos-2** (Amazon) — Pretrained foundation model, zero-shot. No training needed. Slightly less sharp than task-specific models but well-calibrated out of the box.

8. **Ensemble via Vincentization** — Average quantile predictions across all models at each quantile level. Simple, often effective, but equal-weight averaging can dilute signal when the model pool includes weak members.

## Evaluation

Every model is evaluated with the same set of metrics:

- **Pinball loss** at P10, P50, P90 (measures quantile accuracy)
- **Empirical coverage** of the 80% interval (P10–P90) — the most important diagnostic for whether the intervals are trustworthy
- **Mean interval width** (sharpness)
- **Weighted Interval Score** (WIS) — a proper scoring rule that balances accuracy and calibration
- **CRPS** (approximate, from quantile predictions)

The final comparison section adds:

- **Reliability diagrams** — observed vs. nominal quantile frequencies (perfect calibration = diagonal)
- **PIT histograms** — Probability Integral Transform; uniform = well-calibrated
- **Rolling coverage curves** — how coverage varies over time, which is more informative than a single number

Rolling-origin backtests are included for the faster models (SARIMAX, LightGBM, NGBoost, Conformal).

## Setup

Install dependencies in a virtual env:

```
pip install -r requirements.txt
```

Key packages: `statsmodels`, `lightgbm`, `ngboost`, `scikit-learn`, `gluonts`, `torch`, `pytorch-forecasting`, `lightning`, `chronos-forecasting`.

On Apple Silicon the notebook sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so PyTorch ops not yet implemented for Metal fall back to CPU transparently.

## Notes

- This is a playground / learning notebook, not a library - copy what's useful.
- The synthetic DGP is seeded (`rng = np.random.default_rng(7)`) so results are reproducible.
- Deep models (DeepAR, TFT) are trained with modest hyperparameters to keep runtime reasonable. They'd do better with more data and tuning.
