"""transient_lap.py — couple the 7-DOF ride model back into lap time.

The quasi-steady-state ``LapSimulator`` treats every point as already settled,
so dampers are invisible to it. This module adds a one-pass weak coupling:

  1. Run the QSS solver for a baseline speed profile v(s) and lap time.
  2. Derive the lap's excitation from it: lateral ay(s)=v^2*kappa and
     longitudinal ax(s)=dv/dt, plus the cumulative-time grid t(s).
  3. Drive the time-domain 7-DOF model with that excitation, repeated over a
     few laps so it reaches a periodic state, and read the *dynamic* tire
     normal loads back at each track point.
  4. Compare the dynamic load distribution to the **damping-independent**
     steady distribution (springs/ARB only): unevenly loaded or momentarily
     lifted tires lose grip through tyre load-sensitivity. The ratio of total
     grip capacity (sum of mu(N)*N) gives a per-point grip multiplier <= 1.
  5. Re-run the QSS solver with that grip scaling for a damper-sensitive lap
     time. The delta vs the baseline is the transient penalty.

This is intentionally one-directional (the excitation comes from the unscaled
QSS profile, not iterated): enough to make dampers register a cost without the
expense of a fully coupled time-domain lap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulator import LapSimulator
from transient_model import G, RideModelParams, SevenDOFRideModel


@dataclass
class TransientLapResult:
    lap_time_qss: float            # baseline QSS lap time [s]
    lap_time_transient: float      # grip-scaled (damper-sensitive) lap time [s]
    delta: float                   # transient - qss penalty [s] (>= 0)
    grip_scale: np.ndarray         # per-point lateral-grip multiplier (<= 1)
    s: np.ndarray                  # distance along the lap [m] (track points)
    v_qss: np.ndarray              # baseline speed profile [m/s]
    v_transient: np.ndarray        # grip-scaled speed profile [m/s]
    ax: np.ndarray                 # longitudinal accel excitation [m/s^2]
    ay: np.ndarray                 # lateral accel excitation [m/s^2]
    t_lap: np.ndarray              # cumulative time at each track point [s]
    tire_load_dyn: np.ndarray      # dynamic tyre loads, shape (n, 4) [N]
    tire_load_steady: np.ndarray   # steady reference loads, shape (n, 4) [N]


def _lap_time_grid(track, v):
    """Cumulative-time grid t[i] (starting at 0) and lap period from v(s)."""
    s = track.s
    ds = np.diff(s)
    if track.config == "Closed":
        wrap = track.length - s[-1] + s[0]
        ds = np.concatenate(([max(wrap, 1e-6)], ds))
    else:
        ds = np.concatenate(([ds[0]], ds))
    v_prev = np.roll(v, 1)
    if track.config != "Closed":
        v_prev[0] = v[0]
    v_avg = np.maximum(0.5 * (v_prev + v), 1e-3)
    dt = ds / v_avg
    period = float(np.sum(dt))
    t = np.cumsum(dt)
    t = t - t[0]                       # point 0 sits at time 0
    return t, period


def _steady_loads(model: SevenDOFRideModel, ax, ay, t_settle=6.0):
    """Settled tyre loads at constant (ax, ay) — damper coeffs cancel here."""
    r = model.simulate(t_settle, ax=ax, ay=ay, method="LSODA", max_step=0.05)
    return r.tire_load[-1]


def _grip_capacity(tire, loads):
    """Total lateral grip capacity = sum_k mu(N_k) * N_k over the 4 tyres.

    Captures load-sensitivity (peak mu falls with load, so uneven loading or a
    lifted wheel lowers the total) — the mechanism by which damper-driven load
    variation costs grip."""
    loads = np.asarray(loads, dtype=float)
    mu = np.asarray(tire.peak_mu(loads), dtype=float)
    return np.sum(mu * loads, axis=-1)


def run_transient_lap(vehicle, track, *, n_laps: int = 3, method: str = "LSODA",
                      max_step: float = 0.01, grip_floor: float = 0.3,
                      k_tire: float = 200_000.0) -> TransientLapResult:
    """Run the QSS -> 7-DOF -> grip-scaled-QSS coupling. See module docstring."""
    # 1. QSS baseline -------------------------------------------------------- #
    base = LapSimulator(vehicle, track)
    t_qss, v_qss = base.solve()

    # 2. Excitation from the baseline profile -------------------------------- #
    t_lap, period = _lap_time_grid(track, v_qss)
    ay = v_qss ** 2 * np.asarray(track.curvature)          # signed (+left)
    ax = np.gradient(v_qss, t_lap)                          # dv/dt

    # 3. Drive the time-domain ride model periodically ----------------------- #
    params = RideModelParams.from_config(vehicle.config, k_tire=k_tire)
    model = SevenDOFRideModel(params)
    ax_fn = lambda tt: np.interp(tt, t_lap, ax, period=period)
    ay_fn = lambda tt: np.interp(tt, t_lap, ay, period=period)
    t_end = n_laps * period
    t_eval = t_lap + (n_laps - 1) * period                 # sample final lap
    ride = model.simulate(t_end, ax=ax_fn, ay=ay_fn, t_eval=t_eval,
                          method=method, max_step=max_step)
    load_dyn = ride.tire_load                              # (n, 4), aligned to s

    # 4. Damping-independent steady reference + grip multiplier --------------- #
    a_ref = 0.5 * G                                        # linear, no liftoff
    n_static = params.n_static
    dN_dax = (_steady_loads(model, a_ref, 0.0) - n_static) / a_ref
    dN_day = (_steady_loads(model, 0.0, a_ref) - n_static) / a_ref
    load_steady = np.maximum(
        n_static + np.outer(ax, dN_dax) + np.outer(ay, dN_day), 0.0)

    g_dyn = _grip_capacity(vehicle.tire, load_dyn)
    g_steady = np.maximum(_grip_capacity(vehicle.tire, load_steady), 1e-6)
    grip_scale = np.clip(g_dyn / g_steady, grip_floor, 1.0)

    # 5. Re-run QSS with the grip scaling ------------------------------------ #
    scaled = LapSimulator(vehicle, track, grip_scale=grip_scale)
    t_trans, v_trans = scaled.solve()

    return TransientLapResult(
        lap_time_qss=t_qss, lap_time_transient=t_trans, delta=t_trans - t_qss,
        grip_scale=grip_scale, s=np.asarray(track.s, float), v_qss=v_qss,
        v_transient=v_trans, ax=ax, ay=ay, t_lap=t_lap,
        tire_load_dyn=load_dyn, tire_load_steady=load_steady)


if __name__ == "__main__":
    import time
    from dataclasses import replace as _replace
    from config import VehicleConfig
    from vehicle_model import NCMiata
    from track import Track

    soft = VehicleConfig(damper_bump_f=400, damper_rebound_f=700,
                         damper_bump_r=400, damper_rebound_r=700)
    stiff = VehicleConfig(damper_bump_f=3000, damper_rebound_f=5000,
                          damper_bump_r=3000, damper_rebound_r=5000)

    for tk in (Track(), Track.monza()):
        print(f"\n=== {tk.name} ===")
        for label, cfg in (("soft", soft), ("stiff", stiff)):
            t0 = time.time()
            res = run_transient_lap(NCMiata(cfg), tk)
            print(f"  {label:5s}  qss={res.lap_time_qss:7.3f}s  "
                  f"transient={res.lap_time_transient:7.3f}s  "
                  f"penalty={res.delta:+.3f}s  "
                  f"min_scale={res.grip_scale.min():.3f}  "
                  f"[{time.time() - t0:.1f}s wall]")
