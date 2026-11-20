"""Semiconductor demand forecast: Prophet and SARIMA against simple baselines.

Data: FRED series IPG3344S, Industrial Production: Semiconductors and Other
Electronic Components (index, 2017=100, monthly). It measures output, not dollar
sales, and is used here as a public proxy for semiconductor demand.

Method: rolling origin backtest. For each origin the models see only data up to
that month and forecast the next 12 months. Scores are averaged over all origins.
"""
import itertools, logging, urllib.request, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

warnings.filterwarnings("ignore"); logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
HERE = Path(__file__).parent
SERIES, START, H, N_FOLDS = "IPG3344S", "2000-01-01", 12, 10

def load():
    f = HERE / "data" / f"{SERIES}.csv"
    if not f.exists():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={SERIES}"
        f.write_bytes(urllib.request.urlopen(url, timeout=30).read())
    d = pd.read_csv(f, parse_dates=["observation_date"]).rename(columns={"observation_date": "ds", SERIES: "y"})
    d["y"] = pd.to_numeric(d["y"], errors="coerce")
    return d.dropna().query("ds >= @START").reset_index(drop=True)

def naive(train, h):          return np.repeat(train.y.iloc[-1], h)
def seasonal_naive(train, h): return np.resize(train.y.iloc[-12:].to_numpy(), h)

def sarima(train, h):
    y = np.log(train.set_index("ds").y.asfreq("MS")); best = None
    for p, q, P, Q in itertools.product([0, 1, 2], [0, 1], [0, 1], [0, 1]):
        try:
            m = SARIMAX(y, order=(p, 1, q), seasonal_order=(P, 1, Q, 12)).fit(disp=False, maxiter=60)
            if best is None or m.aic < best[0]: best = (m.aic, m)
        except Exception: pass
    return np.exp(best[1].forecast(h).to_numpy())

def prophet(train, h):
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(train.assign(y=np.log(train.y)))
    fut = pd.DataFrame({"ds": pd.date_range(train.ds.iloc[-1], periods=h + 1, freq="MS")[1:]})
    return np.exp(m.predict(fut)["yhat"].to_numpy())

MODELS = {"Naive": naive, "Seasonal naive (baseline)": seasonal_naive, "SARIMA": sarima, "Prophet": prophet}

def main():
    df = load(); print(f"{len(df)} months, {df.ds.min():%Y-%m} to {df.ds.max():%Y-%m}")
    origins = [len(df) - H - 12 * k for k in range(N_FOLDS)][::-1]
    rows, last, fc = [], {}, []
    for i in origins:
        train, test = df.iloc[:i], df.iloc[i:i + H]
        scale = np.mean(np.abs(train.y.to_numpy()[12:] - train.y.to_numpy()[:-12]))  # MASE scale
        for name, fn in MODELS.items():
            f = fn(train, H); e = test.y.to_numpy() - f
            rows.append(dict(origin=train.ds.iloc[-1], model=name, MAE=np.abs(e).mean(),
                             RMSE=np.sqrt((e ** 2).mean()), MAPE=np.mean(np.abs(e / test.y.to_numpy())) * 100,
                             MASE=np.abs(e).mean() / scale))
            last[name] = (test.ds.to_numpy(), f)
            fc += [dict(origin=train.ds.iloc[-1], model=name, ds=d, actual=a, forecast=x) for d, a, x in zip(test.ds, test.y, f)]
        print("done origin", train.ds.iloc[-1].date())
    pd.DataFrame(fc).to_csv(HERE / "results" / "backtest_forecasts.csv", index=False)
    r = pd.DataFrame(rows); r.to_csv(HERE / "results" / "backtest_by_origin.csv", index=False)
    s = r.groupby("model")[["MAE", "RMSE", "MAPE", "MASE"]].mean().sort_values("MAPE").round(3)
    base = s.loc["Seasonal naive (baseline)", "MAPE"]
    s["MAPE vs seasonal naive %"] = ((s["MAPE"] / base - 1) * 100).round(1)
    wins = r.pivot(index="origin", columns="model", values="MAPE")
    s["MAPE vs Naive %"] = ((s["MAPE"] / s.loc["Naive", "MAPE"] - 1) * 100).round(1)
    s["Folds beating Naive"] = [int((wins[m] < wins["Naive"]).sum()) if m != "Naive" else "-" for m in s.index]
    s.to_csv(HERE / "results" / "summary.csv"); print(s.to_string())

    fig, ax = plt.subplots(figsize=(11, 4.6)); tail = df.iloc[-(H + 60):]
    ax.plot(tail.ds, tail.y, color="#222", lw=2, label="Actual")
    for (n, (ds, f)), c in zip(last.items(), ["#aaa", "#e08a00", "#2a7de1", "#16a34a"]):
        if n != "Naive": ax.plot(ds, f, "--", color=c, lw=2, label=n)
    ax.set_title("Last backtest window: 12 month forecast vs actual"); ax.set_ylabel("Index (2017=100)")
    ax.legend(frameon=False); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(HERE / "results" / "last_window.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 3.8)); s["MAPE"].sort_values().plot.barh(ax=ax, color="#2a7de1")
    ax.set_xlabel("Mean MAPE % over 10 backtests (lower is better)"); fig.tight_layout()
    fig.savefig(HERE / "results" / "mape.png", dpi=150)

if __name__ == "__main__": main()
