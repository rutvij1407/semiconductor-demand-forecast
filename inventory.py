"""Monte Carlo simulation of a reorder point and order quantity inventory policy.

Demand each month is the SARIMA forecast times a random error. The errors are resampled from the
log errors SARIMA made in the rolling backtests, so the spread of demand is the spread of real
forecast misses, not a made up distribution.

Policy (s, Q) with a monthly review and a lead time of one month:
  at the end of each month, if inventory on hand is at or below the reorder point s, order the
  smallest number of lots of size Q that lifts inventory back above s. The order arrives at the
  start of next month. Unmet demand is lost.
  s = next month's forecast + z * (error spread in units). z sets the safety stock.
"""
import numpy as np

def simulate(F, errors, z, q_months, runs=10000, seed=0, h=1.0, K=10000.0, p=19.0):
    """Simulate `runs` years. F: forecast units per month. errors: pool of log errors to resample.
    z: safety factor. q_months: order lot size in months of average demand.
    h: holding cost per unit per month, K: cost per order, p: cost per lost unit.
    Returns a dict of metrics averaged over runs."""
    rng = np.random.default_rng(seed)
    F = np.asarray(F, float); months = len(F)
    sigma = float(np.std(errors))
    demand = F * np.exp(rng.choice(np.asarray(errors, float), size=(runs, months)))
    Q = q_months * F.mean()
    nxt = np.append(F[1:], F[-1])                  # forecast of next month, used for the reorder point
    s = nxt * (1 + z * sigma)
    on_hand = np.full(runs, s[0] + Q)              # start stocked just above the reorder point
    arriving = np.zeros(runs)
    stockout = np.zeros((runs, months), bool); lost = np.zeros(runs); held = np.zeros(runs)
    orders = np.zeros(runs); sold = np.zeros(runs)
    for t in range(months):
        on_hand = on_hand + arriving
        sales = np.minimum(on_hand, demand[:, t])
        stockout[:, t] = demand[:, t] > on_hand
        lost += demand[:, t] - sales; sold += sales
        on_hand = on_hand - sales
        held += on_hand
        lots = np.where(on_hand <= s[t], np.ceil((s[t] - on_hand + 1e-9) / Q), 0)
        arriving = lots * Q; orders += lots > 0
    total = h * held + K * orders + p * lost
    return {"z": z, "q_months": q_months,
            "service_level": 1 - stockout.mean(),            # share of months with no stockout
            "fill_rate": sold.sum() / (sold.sum() + lost.sum()),
            "holding_cost": (h * held).mean(), "ordering_cost": (K * orders).mean(),
            "lost_sales_cost": (p * lost).mean(), "total_cost": total.mean(),
            "total_p5": np.percentile(total, 5), "total_p95": np.percentile(total, 95),
            "min_on_hand": float(on_hand.min())}

def best_policy(results, target=0.95):
    """Cheapest policy whose service level meets the target. None if nobody meets it."""
    ok = [r for r in results if r["service_level"] >= target]
    return min(ok, key=lambda r: r["total_cost"]) if ok else None
