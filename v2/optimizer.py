"""
Stage 3: Bayesian Optimization over the GP Surrogate.

The full 3-stage pipeline:
  1. doe_sampler.py  → Latin Hypercube sweep → doe_results.csv
  2. surrogate.py    → Train GP on doe_results.csv → surrogate.pkl
  3. optimizer.py    → Optuna Bayesian optimization queries surrogate (not the sim)

The optimizer calls surrogate.predict() which takes microseconds, not the full
sim which takes milliseconds. This lets Optuna run 10,000+ trials instantly,
while the expensive sim was only run the 200 times needed to build the surrogate.
"""

import os
import numpy as np
import optuna
import pandas as pd
from dataclasses import replace
from pathlib import Path

from config import VehicleConfig
from doe_sampler import (run_lhs_sweep, PARAM_NAMES, LOWER, UPPER,
                         snap_to_buildable, N_PER_KG_MM)
from surrogate import LapTimeSurrogate
from vehicle_model import NCMiata

optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_pipeline(n_doe_samples: int = 200, surrogate_path: str = "surrogate.pkl",
                   doe_path: str = "doe_results.csv", force_retrain: bool = False,
                   base_cfg: VehicleConfig = None):
    """Run stages 1 and 2 if not already cached."""
    if base_cfg is None:
        base_cfg = VehicleConfig()

    if not Path(doe_path).exists() or force_retrain:
        print("Stage 1: Running DoE (Latin Hypercube Sampling)...")
        df = run_lhs_sweep(n_samples=n_doe_samples, base_cfg=base_cfg)
        df.to_csv(doe_path, index=False)
    else:
        print(f"Stage 1: Loading cached DoE results from {doe_path}")
        df = pd.read_csv(doe_path)

    if not Path(surrogate_path).exists() or force_retrain:
        print("\nStage 2: Training GP surrogate...")
        surrogate = LapTimeSurrogate()
        surrogate.fit(df)
        surrogate.cross_validate(df)
        surrogate.save(surrogate_path)
    else:
        print(f"Stage 2: Loading cached surrogate from {surrogate_path}")
        surrogate = LapTimeSurrogate.load(surrogate_path)

    return surrogate


def make_objective(surrogate: LapTimeSurrogate):
    def objective(trial):
        spring_k_f = trial.suggest_float("spring_k_f", LOWER[0], UPPER[0])
        spring_k_r = trial.suggest_float("spring_k_r", LOWER[1], UPPER[1])
        arb_k_f    = trial.suggest_float("arb_k_f",    LOWER[2], UPPER[2])
        arb_k_r    = trial.suggest_float("arb_k_r",    LOWER[3], UPPER[3])

        X = np.array([[spring_k_f, spring_k_r, arb_k_f, arb_k_r]])
        lap_time, std = surrogate.predict(X, return_std=True)

        # Lower Confidence Bound: explore uncertain regions AND exploit good ones
        # This is the core of Bayesian Optimization
        kappa = 1.5  # exploration/exploitation tradeoff
        return float(lap_time[0] - kappa * std[0])
    return objective


def run_optimization(surrogate: LapTimeSurrogate, n_trials: int = 2000,
                     base_cfg: VehicleConfig = None):
    if base_cfg is None:
        base_cfg = VehicleConfig()

    print(f"\nStage 3: Bayesian Optimization ({n_trials} surrogate queries)...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(make_objective(surrogate), n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    buildable = snap_to_buildable(best)
    print("\n=== OPTIMAL SUSPENSION SETUP ===")
    print(f"  {'Parameter':<14} {'Theoretical':>12}  {'Buildable':>10}")
    print(f"  {'-'*40}")
    labels = {"spring_k_f": "Front Spring",
              "spring_k_r": "Rear Spring",
              "arb_k_f":    "Front ARB",
              "arb_k_r":    "Rear ARB"}
    for k in PARAM_NAMES:
        unit = "kg/mm" if "spring" in k else "kg/mm equiv"
        print(f"  {labels[k]:<14} {best[k]/N_PER_KG_MM:>10.2f}  "
              f"→  {buildable[k]/N_PER_KG_MM:.2f} {unit}")

    # Verify with actual simulator (not surrogate), using the loaded car's base params
    from track import Track
    from simulator import LapSimulator
    opt_cfg = replace(base_cfg, **{k: buildable[k] for k in PARAM_NAMES})
    car = NCMiata(opt_cfg)
    sim = LapSimulator(car, Track())
    actual_time, _ = sim.solve()
    print(f"\n  Surrogate predicted: {study.best_value:.3f}s")
    print(f"  Actual sim result:   {actual_time:.3f}s")
    return study


if __name__ == "__main__":
    surrogate = build_pipeline(n_doe_samples=200)
    study = run_optimization(surrogate, n_trials=2000)

