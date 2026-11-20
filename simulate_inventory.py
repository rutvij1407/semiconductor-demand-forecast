"""Run the inventory simulation on the SARIMA forecast for the next 12 months and write results/."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from forecast import load, sarima, H
from inventory import simulate, best_policy

HERE = Path(__file__).parent
UNITS_PER_POINT, RUNS, TARGET = 1000, 10000, 0.95   # 1 index point = 1,000 units a month (assumption)

df = load()
F_index = sarima(df, H)                              # SARIMA refit on all data, next 12 months
F = F_index * UNITS_PER_POINT
errs = pd.read_csv(HERE / "results" / "backtest_forecasts.csv").query("model == 'SARIMA'")
errors = np.log(errs.actual / errs.forecast).to_numpy()
print(f"{len(errors)} SARIMA backtest errors, spread (std of log error) {errors.std():.3f}, bias {errors.mean():+.3f}")

grid = [simulate(F, errors, z, q, runs=RUNS, K=10000.0) for z in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.645, 2.33] for q in [0.1, 0.25, 0.5, 1.0]]
r = pd.DataFrame(grid)
best = best_policy(grid, TARGET)
r["meets_95"] = r.service_level >= TARGET
r["best"] = [best is not None and (x["z"], x["q_months"]) == (best["z"], best["q_months"]) for x in grid]
r.to_csv(HERE / "results" / "inventory_simulation.csv", index=False)
pd.DataFrame({"month": pd.date_range(df.ds.iloc[-1], periods=H + 1, freq="MS")[1:], "forecast_index": F_index}) \
  .to_csv(HERE / "results" / "next_12_forecast.csv", index=False)

show = r.assign(service_level=(r.service_level * 100).round(1), fill_rate=(r.fill_rate * 100).round(1), holding_cost=r.holding_cost.round(0), ordering_cost=r.ordering_cost.round(0), lost_sales_cost=r.lost_sales_cost.round(0), total_cost=r.total_cost.round(0))
print(show[["z", "q_months", "service_level", "fill_rate", "holding_cost", "ordering_cost", "lost_sales_cost", "total_cost", "meets_95", "best"]].to_string(index=False))
print("BEST:", None if best is None else {k: round(float(v), 3) for k, v in best.items()})

fig, ax = plt.subplots(figsize=(8, 4.6))
for q, c in zip([0.1, 0.25, 0.5, 1.0], ["#7c3aed", "#2a7de1", "#16a34a", "#e08a00"]):
    d = r[r.q_months == q].sort_values("z")
    ax.plot(d.service_level * 100, d.total_cost / 1e3, "o-", color=c, label=f"Order lot {q} of a month of demand")
ax.axvline(TARGET * 100, ls="--", color="#999"); ax.text(TARGET * 100 + .3, ax.get_ylim()[1] * .95, "95% target", color="#666")
if best is not None:
    ax.scatter([best["service_level"] * 100], [best["total_cost"] / 1e3], s=180, facecolors="none", edgecolors="#c1121f", lw=2, label="Chosen policy")
ax.set_xlabel("Service level (share of months with no stockout, %)"); ax.set_ylabel("Expected yearly cost (thousands)")
ax.set_title(f"Inventory policies, {RUNS:,} simulated years each"); ax.legend(frameon=False); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(HERE / "results" / "inventory_tradeoff.png", dpi=150)
