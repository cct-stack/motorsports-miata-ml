"""Setup-comparison analysis and plots.

Turns the raw output of :class:`simulator.LapSimulator` (a lap time plus a
speed profile at every track point) into the telemetry-style comparisons an
engineer actually reads: a speed-trace overlay, a cumulative time-delta trace
(where on the lap one setup gains or loses time vs another), and a lap-time
bar chart with a min/max-speed summary.

The module never selects a Matplotlib backend itself, so the caller decides
whether to pop up interactive windows (``run_pipeline.py``) or render head-
less to PNG (the test-suite uses the ``Agg`` backend).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class LapResult:
    """One simulated lap, ready to plot/compare.

    Build with :meth:`from_sim` so the cumulative-time integration matches the
    simulator's own segment lengths exactly (single source of truth).
    """

    label: str          # e.g. "Baseline" / "Optimized"
    track_name: str
    s: np.ndarray       # distance along the lap [m]
    v: np.ndarray       # speed at each point [m/s]
    ds: np.ndarray      # segment length ending at each point [m]
    curvature: np.ndarray  # signed track curvature [1/m] at each point
    lap_time: float     # total lap time [s]
    closed: bool        # closed loop vs point-to-point

    @classmethod
    def from_sim(cls, label: str, sim) -> "LapResult":
        lap_time, v = sim.solve()
        ds = sim._segment_lengths()
        track = sim.track
        return cls(label, track.name, np.asarray(track.s, float), v, ds,
                   np.asarray(track.curvature, float),
                   float(lap_time), track.config == "Closed")

    # --- derived quantities ------------------------------------------- #
    @property
    def v_kph(self) -> np.ndarray:
        return self.v * 3.6

    @property
    def v_min(self) -> float:
        return float(self.v.min())

    @property
    def v_max(self) -> float:
        return float(self.v.max())

    @property
    def cum_time(self) -> np.ndarray:
        """Elapsed time at each point, integrated like the solver does:
        ``dt[i] = ds[i] / mean(v[i-1], v[i])`` then a running sum."""
        v_prev = np.roll(self.v, 1)
        if not self.closed:
            v_prev[0] = self.v[0]
        v_avg = np.maximum(0.5 * (self.v + v_prev), 1e-3)
        return np.cumsum(self.ds / v_avg)

    @property
    def lat_g(self) -> np.ndarray:
        """Lateral acceleration at each point [g], from ``v**2 * |kappa|``."""
        return self.v * self.v * np.abs(self.curvature) / 9.81

    @property
    def long_g(self) -> np.ndarray:
        """Longitudinal acceleration [g] from the speed profile, derived the
        same way the solver integrates speed: ``ax = (v[i]**2 - v[i-1]**2) /
        (2*ds[i])``. Positive = accelerating, negative = braking."""
        v_prev = np.roll(self.v, 1)
        if not self.closed:
            v_prev[0] = self.v[0]
        ax = (self.v * self.v - v_prev * v_prev) / (2.0 * np.maximum(self.ds, 1e-6))
        return ax / 9.81


def build_lap_channels(result: LapResult, vehicle) -> "pd.DataFrame":
    """Build a per-point telemetry DataFrame (OptimumLap-style channels).

    Parameters
    ----------
    result:
        A solved :class:`LapResult` (call ``LapResult.from_sim`` first).
    vehicle:
        The :class:`~vehicle_model.NCMiata` instance used for the simulation.

    Returns
    -------
    pandas.DataFrame with columns:

    ``distance_m``, ``time_s``, ``speed_ms``, ``speed_kph``,
    ``lat_accel_g``, ``long_accel_g``, ``combined_g``,
    ``corner_radius_m``,
    ``gear``, ``engine_rpm``, ``engine_torque_Nm``,
    ``engine_power_kW``, ``engine_power_hp``,
    ``wheel_tractive_force_N``, ``drag_force_N``,
    ``rolling_resistance_N``, ``downforce_N``,
    ``lat_grip_used_pct``,
    ``driver_state``  (``"accelerating"`` / ``"braking"`` / ``"cornering"``).
    """
    import pandas as pd

    v   = result.v
    s   = result.s
    ds  = result.ds
    kap = result.curvature

    # --- time integration (matches solver) --------------------------------
    v_prev = np.roll(v, 1)
    if not result.closed:
        v_prev[0] = v[0]
    v_avg   = np.maximum(0.5 * (v + v_prev), 1e-3)
    cum_t   = np.cumsum(ds / v_avg)

    # --- longitudinal / lateral accelerations [g] -------------------------
    ax_raw  = (v * v - v_prev * v_prev) / (2.0 * np.maximum(ds, 1e-6))
    long_g  = ax_raw / 9.81
    lat_g   = v * v * np.abs(kap) / 9.81
    comb_g  = np.sqrt(long_g ** 2 + lat_g ** 2)

    # --- corner radius (cap at 5 km on near-straights) --------------------
    radius  = np.where(np.abs(kap) > 1e-6, 1.0 / np.abs(kap), 5000.0)

    # --- engine / driveline channels (vectorised) -------------------------
    eng = vehicle.engine_state(v)

    # --- resistive forces -------------------------------------------------
    drag    = np.array([vehicle.drag_force(vi)         for vi in v])
    df_aero = np.array([vehicle.downforce(vi)          for vi in v])
    rr      = np.full(len(v), vehicle.rolling_resistance())

    # --- lateral grip utilisation [%] ------------------------------------
    ay_max  = vehicle.max_lat_accel(v)
    lat_pct = np.where(ay_max > 0, np.minimum(lat_g * 9.81 / ay_max, 1.0) * 100, 0.0)

    # --- driver state (discrete) ------------------------------------------
    accel_thresh = 0.05   # [g]  above this → on throttle
    brake_thresh = -0.10  # [g]  below this → braking
    state = np.where(
        long_g > accel_thresh, "accelerating",
        np.where(long_g < brake_thresh, "braking", "cornering")
    )

    return pd.DataFrame({
        "distance_m":           s,
        "time_s":               cum_t,
        "speed_ms":             v,
        "speed_kph":            v * 3.6,
        "lat_accel_g":          lat_g,
        "long_accel_g":         long_g,
        "combined_g":           comb_g,
        "corner_radius_m":      radius,
        "gear":                 eng["gear"],
        "engine_rpm":           eng["rpm"],
        "engine_torque_Nm":     eng["torque_Nm"],
        "engine_power_kW":      eng["power_kW"],
        "engine_power_hp":      eng["power_kW"] * 1.341,
        "wheel_tractive_force_N": eng["wheel_force_N"],
        "drag_force_N":         drag,
        "rolling_resistance_N": rr,
        "downforce_N":          df_aero,
        "lat_grip_used_pct":    lat_pct,
        "driver_state":         state,
    })


def lap_summary(df: "pd.DataFrame") -> dict:
    """Headline statistics from a :func:`build_lap_channels` DataFrame."""
    total_dist = float(df["distance_m"].iloc[-1])
    total_time = float(df["time_s"].iloc[-1])

    by_state = df.groupby("driver_state")["distance_m"]
    def _pct(key):
        if key in by_state.groups:
            span = float(by_state.get_group(key).iloc[-1] -
                         by_state.get_group(key).iloc[0]) if len(
                by_state.get_group(key)) > 1 else float(
                df.loc[df["driver_state"] == key, "speed_ms"].count())
            # simpler: fraction of *points* in each state
        return float((df["driver_state"] == key).mean() * 100.0)

    return {
        "lap_time_s":       total_time,
        "distance_m":       total_dist,
        "v_max_kph":        float(df["speed_kph"].max()),
        "v_min_kph":        float(df["speed_kph"].min()),
        "v_avg_kph":        float(df["speed_kph"].mean()),
        "max_lat_g":        float(df["lat_accel_g"].max()),
        "max_accel_g":      float(df["long_accel_g"].max()),
        "max_brake_g":      float(-df["long_accel_g"].min()),
        "max_combined_g":   float(df["combined_g"].max()),
        "max_rpm":          float(df["engine_rpm"].max()),
        "max_power_kW":     float(df["engine_power_kW"].max()),
        "max_power_hp":     float(df["engine_power_hp"].max()),
        "gears_used":       sorted(df["gear"].unique().tolist()),
        "pct_accelerating": _pct("accelerating"),
        "pct_braking":      _pct("braking"),
        "pct_cornering":    _pct("cornering"),
    }


def gg_diagram_figure(df: "pd.DataFrame", label: str = ""):
    """G-G diagram: lateral vs longitudinal acceleration scatter, coloured by
    speed.  A unit friction-circle (combined_g = 1 g) is overlaid as a visual
    reference.  Returns the Matplotlib ``Figure``."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(
        df["lat_accel_g"], df["long_accel_g"],
        c=df["speed_kph"], cmap="plasma", s=4, alpha=0.7,
        vmin=df["speed_kph"].min(), vmax=df["speed_kph"].max()
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Speed [km/h]", fontsize=9)

    # mirror for left/right corners
    ax.scatter(
        -df["lat_accel_g"], df["long_accel_g"],
        c=df["speed_kph"], cmap="plasma", s=4, alpha=0.7,
        vmin=df["speed_kph"].min(), vmax=df["speed_kph"].max()
    )

    # friction-circle envelope
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.4,
            label="1 g friction circle")

    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Lateral acceleration [g]")
    ax.set_ylabel("Longitudinal acceleration [g]\n(+ accel / − brake)")
    title = f"G-G diagram  —  {label}" if label else "G-G diagram"
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def channels_figure(df: "pd.DataFrame", label: str = ""):
    """Stacked 4-panel channel plot vs distance (OptimumLap-style).

    Panels: speed (km/h), engine RPM (coloured by gear), wheel power (kW),
    and a driver-state band (green=accel, red=brake, grey=cornering).
    Returns the Matplotlib ``Figure``."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.cm import get_cmap

    dist = df["distance_m"].values
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 2, 1]})
    title = f"Channel data  —  {label}" if label else "Channel data"
    fig.suptitle(title, fontsize=12)

    # Panel 1: speed
    axes[0].plot(dist, df["speed_kph"], color="#1f77b4", lw=1.4)
    axes[0].set_ylabel("Speed [km/h]")
    axes[0].grid(True, alpha=0.25)

    # Panel 2: RPM coloured by gear
    gears = df["gear"].values
    max_g = int(gears.max()) if len(gears) else 1
    cmap_g = plt.cm.get_cmap("tab10", max_g)
    for g in range(1, max_g + 1):
        mask = gears == g
        if mask.any():
            axes[1].fill_between(dist, 0, df["engine_rpm"].values,
                                 where=mask, alpha=0.5, color=cmap_g(g - 1),
                                 label=f"Gear {g}")
    axes[1].plot(dist, df["engine_rpm"], color="#333", lw=0.8)
    axes[1].set_ylabel("RPM")
    axes[1].legend(loc="upper right", fontsize=7, ncol=max_g)
    axes[1].grid(True, alpha=0.25)

    # Panel 3: power
    axes[2].plot(dist, df["engine_power_kW"], color="#2ca02c", lw=1.4)
    axes[2].set_ylabel("Wheel power [kW]")
    axes[2].grid(True, alpha=0.25)

    # Panel 4: driver state
    state_colors = {"accelerating": "#2ca02c", "braking": "#d62728",
                    "cornering": "#aaaaaa"}
    for state, color in state_colors.items():
        mask = df["driver_state"].values == state
        axes[3].fill_between(dist, 0, 1, where=mask, color=color,
                             alpha=0.8, label=state.capitalize())
    axes[3].set_ylim(0, 1)
    axes[3].set_yticks([])
    axes[3].set_ylabel("Driver\nstate")
    axes[3].set_xlabel("Distance [m]")
    axes[3].legend(loc="upper right", fontsize=7, ncol=3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def channels_to_csv(df: "pd.DataFrame", path: str) -> None:
    """Write the channels DataFrame to a CSV file at *path*."""
    df.to_csv(path, index=False)


def channels_to_excel(df: "pd.DataFrame", path: str) -> None:
    """Write the channels DataFrame to an Excel file at *path*
    (requires ``openpyxl`` to be installed)."""
    df.to_excel(path, index=False, engine="openpyxl")


def time_delta(baseline: LapResult, optimized: LapResult) -> np.ndarray:
    """Cumulative time gained/lost by *optimized* vs *baseline* at each point.
    Negative => optimized is ahead (faster). Both must share the track mesh."""
    return optimized.cum_time - baseline.cum_time


def summary(results: Sequence[LapResult]) -> List[Dict[str, float]]:
    """Per-setup headline metrics (lap time, min/max speed)."""
    return [
        {
            "label": r.label,
            "track": r.track_name,
            "lap_time": r.lap_time,
            "v_min_kph": r.v_min * 3.6,
            "v_max_kph": r.v_max * 3.6,
        }
        for r in results
    ]


def compare_figure(baseline: LapResult, optimized: LapResult):
    """Two-panel comparison for a single track.

    Top: speed (km/h) vs distance for both setups, with the slowest point of
    each marked. Bottom: cumulative time-delta vs distance (optimized minus
    baseline). Returns the Matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    fig, (ax_v, ax_d) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(f"{baseline.track_name}  -  setup comparison", fontsize=13)

    for r, color in ((baseline, "#888888"), (optimized, "#1f77b4")):
        ax_v.plot(r.s, r.v_kph, color=color, lw=1.6,
                  label=f"{r.label}  ({r.lap_time:.3f}s)")
        i_min = int(np.argmin(r.v))
        ax_v.plot(r.s[i_min], r.v_kph[i_min], "o", color=color, ms=5)
    ax_v.set_ylabel("Speed [km/h]")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(loc="lower right", fontsize=9)

    delta = time_delta(baseline, optimized)
    ax_d.plot(baseline.s, delta, color="#d62728", lw=1.4)
    ax_d.axhline(0.0, color="k", lw=0.8)
    ax_d.fill_between(baseline.s, delta, 0.0, where=delta <= 0,
                      color="#2ca02c", alpha=0.25, interpolate=True)
    ax_d.fill_between(baseline.s, delta, 0.0, where=delta > 0,
                      color="#d62728", alpha=0.25, interpolate=True)
    gain = baseline.lap_time - optimized.lap_time
    ax_d.set_ylabel("Δt vs baseline [s]")
    ax_d.set_xlabel("Distance [m]")
    ax_d.set_title(f"Optimized gains {gain:+.3f}s over the lap "
                   f"(green = ahead)", fontsize=10)
    ax_d.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def find_corners(curvature: np.ndarray, frac: float = 0.15) -> List[tuple]:
    """Locate corners as contiguous runs where ``|curvature|`` exceeds a
    fraction of its peak. Returns a list of ``(start, stop)`` index slices in
    track order (``stop`` exclusive). Straights fall below the threshold and
    are skipped, so each returned slice brackets one corner's apex."""
    kappa = np.abs(np.asarray(curvature, float))
    kmax = float(kappa.max()) if kappa.size else 0.0
    if kmax <= 1e-9:
        return []
    thresh = max(frac * kmax, 1e-4)
    mask = kappa > thresh
    corners: List[tuple] = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            corners.append((i, j))
            i = j
        else:
            i += 1
    return corners


def gforce_figure(baseline: LapResult, optimized: LapResult):
    """Two-panel G-force comparison: lateral g (cornering load) on top and
    longitudinal g (braking/acceleration) below, both vs distance, overlaying
    the two setups. Returns the Matplotlib ``Figure``."""
    import matplotlib.pyplot as plt

    fig, (ax_lat, ax_lon) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(f"{baseline.track_name}  -  G-force traces", fontsize=13)

    for r, color in ((baseline, "#888888"), (optimized, "#1f77b4")):
        ax_lat.plot(r.s, r.lat_g, color=color, lw=1.4, label=r.label)
        ax_lon.plot(r.s, r.long_g, color=color, lw=1.4, label=r.label)

    ax_lat.set_ylabel("Lateral g")
    ax_lat.grid(True, alpha=0.3)
    ax_lat.legend(loc="upper right", fontsize=9)

    ax_lon.axhline(0.0, color="k", lw=0.8)
    ax_lon.set_ylabel("Longitudinal g\n(+accel / -brake)")
    ax_lon.set_xlabel("Distance [m]")
    ax_lon.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def corner_speed_figure(baseline: LapResult, optimized: LapResult,
                        frac: float = 0.15):
    """Grouped bar chart of the minimum speed in each detected corner,
    baseline vs optimized. Corners are taken from the (shared) track curvature
    so both setups are compared at the same apexes. Returns the ``Figure``."""
    import matplotlib.pyplot as plt

    corners = find_corners(baseline.curvature, frac=frac)
    if not corners:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No corners detected on this track",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    base_speeds = [float(baseline.v_kph[i:j].min()) for (i, j) in corners]
    opt_speeds = [float(optimized.v_kph[i:j].min()) for (i, j) in corners]
    labels = [f"T{n + 1}" for n in range(len(corners))]
    x = np.arange(len(corners))
    w = 0.4

    fig, ax = plt.subplots(figsize=(max(8, len(corners) * 0.6), 5))
    ax.bar(x - w / 2, base_speeds, w, color="#888888", label="Baseline")
    ax.bar(x + w / 2, opt_speeds, w, color="#1f77b4", label="Optimized")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Min speed [km/h]")
    ax.set_xlabel("Corner (track order)")
    ax.set_title(f"{baseline.track_name}  -  minimum speed per corner")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def transient_report_figure(result, *, title: str | None = None):
    """Damper-sensitivity report from ``transient_lap.run_transient_lap``.

    Top panel: the QSS baseline speed trace (grey) vs the grip-scaled,
    damper-sensitive trace (blue) — where they diverge is where the transient
    load swing forced the car to slow. Bottom panel: the per-point lateral grip
    *retained* (``grip_scale``, 1.0 = no loss), red-filled below 1.0 so the
    grip-losing zones line up with the speed gap above. The lap-time penalty
    (transient minus QSS) is reported in the title — bigger means the current
    dampers let load transfer cost more lap time. Returns the ``Figure``.
    """
    import matplotlib.pyplot as plt

    s = np.asarray(result.s, float)
    fig, (ax_v, ax_g) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]})
    head = title or "Damper transient analysis"
    fig.suptitle(f"{head}  —  penalty {result.delta:+.3f}s "
                 f"(QSS {result.lap_time_qss:.3f}s → "
                 f"transient {result.lap_time_transient:.3f}s)", fontsize=12)

    ax_v.plot(s, np.asarray(result.v_qss) * 3.6, color="#888888", lw=1.6,
              label=f"QSS baseline ({result.lap_time_qss:.3f}s)")
    ax_v.plot(s, np.asarray(result.v_transient) * 3.6, color="#1f77b4", lw=1.6,
              label=f"Damper-sensitive ({result.lap_time_transient:.3f}s)")
    ax_v.set_ylabel("Speed [km/h]")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(loc="lower right", fontsize=9)

    g = np.asarray(result.grip_scale, float)
    ax_g.plot(s, g, color="#d62728", lw=1.4)
    ax_g.axhline(1.0, color="k", lw=0.8)
    ax_g.fill_between(s, g, 1.0, where=g < 1.0, color="#d62728", alpha=0.25,
                      interpolate=True)
    ax_g.set_ylim(min(0.95 * float(g.min()), 0.99), 1.005)
    ax_g.set_ylabel("Lateral grip retained\n(1.0 = no loss)")
    ax_g.set_xlabel("Distance [m]")
    ax_g.set_title(f"min retained grip {g.min():.3f}", fontsize=10)
    ax_g.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def laptime_bar_figure(results: Sequence[LapResult]):
    """Bar chart of total lap time per setup (grouped by track)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{r.label}\n{r.track_name}" for r in results]
    times = [r.lap_time for r in results]
    bars = ax.bar(labels, times, color="#1f77b4", alpha=0.8)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, t, f"{t:.3f}s",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Lap time [s]")
    ax.set_title("Lap-time comparison")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
