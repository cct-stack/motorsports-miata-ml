"""
Stage 1: Design of Experiments (DoE) using Latin Hypercube Sampling.

Latin Hypercube Sampling (LHS) divides each parameter axis into equal-sized
buckets, then picks exactly one sample per bucket per axis — so you get
even, non-clustered coverage of the whole parameter space. Much better than
random sampling for building a surrogate with limited compute budget.
"""

import numpy as np
import pandas as pd
from dataclasses import replace
from scipy.stats.qmc import LatinHypercube, scale
from config import VehicleConfig
from vehicle_model import NCMiata
from simulator import LapSimulator
from track import Track


# --- Parameter space definition ---
# (name, lower_bound, upper_bound) in SI units (N/m)
PARAM_BOUNDS = [
    ("spring_k_f", 40_000,  120_000),  # Front spring: ~4 to 12 kgf/mm
    ("spring_k_r", 20_000,   80_000),  # Rear spring:  ~2 to 8 kgf/mm
    ("arb_k_f",        0,   30_000),   # Front ARB: 0 to ~3 kgf/mm equiv
    ("arb_k_r",        0,   15_000),   # Rear ARB:  0 to ~1.5 kgf/mm equiv
]

PARAM_NAMES = [p[0] for p in PARAM_BOUNDS]
LOWER       = np.array([p[1] for p in PARAM_BOUNDS])
UPPER       = np.array([p[2] for p in PARAM_BOUNDS])

# ── Unit conversion ──────────────────────────────────────────────────────────
# 1 kgf/mm = 9.81 N/mm = 9 810 N/m  (using g = 9.81 m/s²)
N_PER_KG_MM: float = 9_810.0

# Purchasable snap steps per parameter (kgf/mm).
# Springs: 0.5 kgf/mm matches standard coilover-spring catalogue increments
#          (Swift, Hyperco, Eibach ERS, most 65 mm-ID generic springs).
# ARBs: 0.1 kgf/mm equivalent – finer because adjustable bars give several
#       discrete positions; caller should note the target, not a part number.
_SNAP_STEPS_KG_MM: dict = {
    "spring_k_f": 0.5,
    "spring_k_r": 0.5,
    "arb_k_f":    0.1,
    "arb_k_r":    0.1,
}


def snap_to_buildable(params: dict) -> dict:
    """Round optimizer output to nearest purchasable increment.

    Input and output are both in SI (N/m), same keys as *params*.
    Springs snap to 0.5 kgf/mm; ARBs snap to 0.1 kgf/mm equivalent.
    The returned config is what actually gets simulated and exported so
    the reported lap-time gain reflects a setup you can physically build.
    """
    out = {}
    for k, v_si in params.items():
        step_si = _SNAP_STEPS_KG_MM.get(k, 0.5) * N_PER_KG_MM
        out[k] = round(v_si / step_si) * step_si
    return out


def run_lhs_sweep(n_samples: int = 200, track: Track = None, seed: int = 42,
                  base_cfg: VehicleConfig = None) -> pd.DataFrame:
    """Generates n_samples setups via LHS and runs each through the lap sim.

    Non-suspension parameters (mass, aero, power, tire, …) are taken from
    *base_cfg* (defaults to ``VehicleConfig()`` if not supplied).  Only the
    four suspension params are swept.  Returns a DataFrame of parameters +
    lap_time.
    """
    if track is None:
        track = Track()
    if base_cfg is None:
        base_cfg = VehicleConfig()

    sampler = LatinHypercube(d=len(PARAM_BOUNDS), seed=seed)
    unit_samples = sampler.random(n=n_samples)                # [0, 1]^d
    samples      = scale(unit_samples, l_bounds=LOWER, u_bounds=UPPER)

    rows = []
    for i, params in enumerate(samples):
        cfg = replace(base_cfg,
                      spring_k_f=params[0], spring_k_r=params[1],
                      arb_k_f=params[2],    arb_k_r=params[3])
        car = NCMiata(cfg)

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
