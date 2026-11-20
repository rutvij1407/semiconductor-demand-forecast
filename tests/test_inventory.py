import inspect
import numpy as np
from inventory import simulate, best_policy

F = np.full(12, 170000.0)
NOISE = np.random.default_rng(1).normal(0, 0.05, 500)

def test_default_runs_is_at_least_10000():
    assert inspect.signature(simulate).parameters["runs"].default >= 10000

def test_no_forecast_error_means_no_stockouts():
    r = simulate(F, np.zeros(50), z=0.0, q_months=0.5, runs=500)
    assert r["service_level"] == 1.0 and r["lost_sales_cost"] == 0

def test_more_safety_stock_never_lowers_service():
    s = [simulate(F, NOISE, z, 0.25, runs=4000, seed=3)["service_level"] for z in [-1, 0, 1, 2]]
    assert s == sorted(s) and s[0] < s[-1]

def test_more_safety_stock_costs_more_to_hold():
    lo = simulate(F, NOISE, 0.0, 0.25, runs=2000)["holding_cost"]
    hi = simulate(F, NOISE, 2.0, 0.25, runs=2000)["holding_cost"]
    assert hi > lo

def test_inventory_never_negative_and_costs_add_up():
    r = simulate(F, NOISE, 0.5, 0.25, runs=2000)
    assert r["min_on_hand"] >= 0
    assert abs(r["holding_cost"] + r["ordering_cost"] + r["lost_sales_cost"] - r["total_cost"]) < 1e-6 * r["total_cost"]
    assert 0 <= r["fill_rate"] <= 1 and 0 <= r["service_level"] <= 1

def test_same_seed_gives_same_answer():
    assert simulate(F, NOISE, 1.0, 0.5, runs=1000, seed=9) == simulate(F, NOISE, 1.0, 0.5, runs=1000, seed=9)

def test_best_policy_is_cheapest_that_meets_target():
    rs = [{"service_level": .90, "total_cost": 1}, {"service_level": .96, "total_cost": 5}, {"service_level": .99, "total_cost": 3}]
    assert best_policy(rs, 0.95)["total_cost"] == 3
    assert best_policy(rs, 0.999) is None
