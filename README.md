# Semiconductor Demand Forecast

Forecasts monthly US semiconductor output 12 months ahead with **SARIMA** and **Prophet**, and tests whether they beat simple baselines using a rolling origin backtest.

## Data
FRED series [IPG3344S](https://fred.stlouisfed.org/series/IPG3344S), Industrial Production: Semiconductors and Other Electronic Components (index, 2017 = 100, monthly), 2000 to 2026. It measures production volume, not dollar sales, so it is a public proxy for semiconductor demand.

## Method
- 10 backtests. At each origin a model sees only past data and forecasts the next 12 months. Origins step forward one year, so no future data leaks into training.
- Baselines: **Naive** (repeat last month) and **Seasonal naive** (repeat the same month last year).
- Models: **SARIMA** (log scale, order picked by AIC at every origin) and **Prophet** (log scale, yearly seasonality).
- Metrics: MAE, RMSE, MAPE, and MASE (scaled against seasonal naive in training).

## Results (mean over 10 backtests, lower is better)

| Model | MAPE % | MAE | MASE | MAPE vs Naive | Folds beating Naive |
|---|---|---|---|---|---|
| SARIMA | 4.10 | 5.00 | 0.80 | 5.4% better | 5 of 10 |
| Prophet | 4.13 | 5.61 | 0.87 | 4.6% better | 6 of 10 |
| Naive | 4.33 | 5.98 | 0.94 | n/a | n/a |
| Seasonal naive | 7.26 | 9.72 | 1.53 | 67.8% worse | 2 of 10 |

![MAPE](results/mape.png)
![Last window](results/last_window.png)

## What this says
- SARIMA and Prophet beat seasonal naive by about 43%, but that baseline is weak here because the series is trend driven with little seasonality.
- Against the stronger Naive baseline the gain is only about 5%, and neither model wins in most folds. This series is hard to beat with a simple model, and the honest takeaway is that the extra complexity buys a small improvement.
- In the last window SARIMA tracked the 2026 upswing closely, while Prophet underestimated the trend.

## Limits
Industrial production is not revenue, the backtest has 10 origins (a small sample), and no external drivers such as capex or memory prices are used. Adding them is the obvious next step.

## Run
```
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python forecast.py
```
The script downloads the data from FRED (no key needed) and writes `results/`.
