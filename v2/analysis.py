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
