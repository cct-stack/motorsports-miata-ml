"""
Stage 1: Design of Experiments (DoE) using Latin Hypercube Sampling.

Latin Hypercube Sampling (LHS) divides each parameter axis into equal-sized
buckets, then picks exactly one sample per bucket per axis — so you get
even, non-clustered coverage of the whole parameter space. Much better than
random sampling for building a surrogate with limited compute budget.
"""

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube, scale
from vehicle_model import NCMiata
from simulator import LapSimulator
from track import Track


# --- Parameter space definition ---
# (name, lower_bound, upper_bound) in SI units (N/m)
PARAM_BOUNDS = [
    ("spring_k_f", 40_000,  120_000),  # Front spring: 4 to 12 kg/mm
    ("spring_k_r", 20_000,   80_000),  # Rear spring:  2 to 8 kg/mm
    ("arb_k_f",        0,   30_000),   # Front ARB: 0 to 3 kg/mm equiv
    ("arb_k_r",        0,   15_000),   # Rear ARB:  0 to 1.5 kg/mm equiv
]

PARAM_NAMES = [p[0] for p in PARAM_BOUNDS]
LOWER       = np.array([p[1] for p in PARAM_BOUNDS])
UPPER       = np.array([p[2] for p in PARAM_BOUNDS])


def run_lhs_sweep(n_samples: int = 200, track: Track = None, seed: int = 42) -> pd.DataFrame:
    """
    Generates n_samples setups via LHS and runs each through the lap sim.
    Returns a DataFrame of parameters + lap_time.
    """
    if track is None:
        track = Track()

    sampler = LatinHypercube(d=len(PARAM_BOUNDS), seed=seed)
    unit_samples = sampler.random(n=n_samples)                # [0, 1]^d
    samples      = scale(unit_samples, l_bounds=LOWER, u_bounds=UPPER)

    rows = []
    for i, params in enumerate(samples):
        car = NCMiata()
        car.spring_k_f = params[0]
        car.spring_k_r = params[1]
        car.arb_k_f    = params[2]
        car.arb_k_r    = params[3]

        sim = LapSimulator(car, track)
        lap_time, _ = sim.solve()

        row = dict(zip(PARAM_NAMES, params))
        row["lap_time"] = lap_time
        rows.append(row)

        if (i + 1) % 50 == 0:
            print(f"  Completed {i + 1}/{n_samples} DoE runs...")

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    print("Running Latin Hypercube Sampling sweep (200 samples)...")
    df = run_lhs_sweep(n_samples=200)
    df.to_csv("doe_results.csv", index=False)
    print(f"\nDone. Lap time range: {df['lap_time'].min():.3f}s - {df['lap_time'].max():.3f}s")
    print(df.describe())
