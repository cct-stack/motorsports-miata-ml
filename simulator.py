from typing import Tuple

import numpy as np

from interfaces import TrackModel, VehicleModel


class LapSimulator:
    """Quasi-steady-state lap-time solver.

    Uses the same method as OpenLAP: build a per-point cornering speed limit
    from the lateral-grip envelope, then make a forward pass (acceleration,
    engine- and tyre-limited) and a backward pass (braking, tyre-limited). The
    final speed at each point is the minimum the car can satisfy in both
    directions. Longitudinal and lateral grip are coupled with a friction
    circle inside the vehicle model.

    The solver depends only on the ``VehicleModel`` / ``TrackModel`` protocols
    (see interfaces.py), so any compatible vehicle or track can be simulated.
    """

    def __init__(self, vehicle: VehicleModel, track: TrackModel,
                 v_cap: float = 100.0):
        self.vehicle = vehicle
        self.track = track
        self.v_cap = v_cap  # upper bound on straight-line speed search [m/s]

    # ------------------------------------------------------------------ #
    def _corner_speed_limit(self) -> np.ndarray:
        """Max speed at each point from pure lateral grip.

        Solves the implicit relation v = sqrt(ay_max(v)/kappa): with aero
        downforce the grip limit ay_max rises with speed, so the cornering
        speed and the grip it relies on are mutually dependent. A few
        fixed-point iterations (vectorised through the cached grip curve)
        converge quickly. With no downforce ay_max is constant and this
        reduces to the single-shot v = sqrt(ay_max/kappa). On near-straight
        sections (kappa ~ 0) the limit is the search cap."""
        kappa = np.maximum(np.abs(self.track.curvature), 1e-9)
        ay = self.vehicle.max_lat_accel(0.0)
        v = np.minimum(np.sqrt(ay / kappa), self.v_cap)
        for _ in range(8):
            v_new = np.minimum(np.sqrt(self.vehicle.max_lat_accel(v) / kappa),
                               self.v_cap)
            if np.max(np.abs(v_new - v)) < 1e-3:
                v = v_new
                break
            v = v_new
        return v

    def _segment_lengths(self) -> np.ndarray:
        """Distance to the previous point at each index. For a closed track the
        first entry wraps around from the last point."""
        ds = np.diff(self.track.s)
        if self.track.config == "Closed":
            wrap = self.track.length - self.track.s[-1] + self.track.s[0]
            ds = np.concatenate(([max(wrap, 1e-6)], ds))
        else:
            ds = np.concatenate(([ds[0]], ds))
        return ds

    def solve(self) -> Tuple[float, np.ndarray]:
        """Return (lap_time [s], speed_profile [m/s] at each track point)."""
        # Build the speed-dependent grip curve up front so it reflects the
        # current setup/aero, then look up ay_max(v) at each point.
        self.vehicle.build_grip_curve(v_top=self.v_cap)
        v_corner = self._corner_speed_limit()
        ds = self._segment_lengths()
        n = len(v_corner)
        closed = self.track.config == "Closed"

        v = v_corner.copy()

        # --- Forward pass (acceleration out of corners) ---------------- #
        passes = 2 if closed else 1
        for _ in range(passes):
            for i in range(n):
                j = i - 1  # previous point (wraps to n-1 when i==0 on closed)
                if i == 0 and not closed:
                    continue
                vp = v[j]
                ay = vp * vp * abs(self.track.curvature[j])
                ay_max = self.vehicle.max_lat_accel(vp)
                ax = self.vehicle.get_max_long_accel(vp, ay, ay_max)
                # Integrate forward. ax may be negative when aero drag exceeds
                # available thrust (above the engine/drag equilibrium speed),
                # which correctly caps straight-line top speed.
                v_reach = np.sqrt(max(vp * vp + 2.0 * ax * ds[i], 0.0))
                v[i] = min(v[i], v_reach)

        # --- Backward pass (braking into corners) ---------------------- #
        for _ in range(passes):
            for i in range(n - 1, -1, -1):
                j = (i + 1) % n  # next point
                if i == n - 1 and not closed:
                    continue
                vn = v[j]
                ay = vn * vn * abs(self.track.curvature[j])
                ay_max = self.vehicle.max_lat_accel(vn)
                ax = self.vehicle.get_max_decel(vn, ay, ay_max)
                v_reach = np.sqrt(vn * vn + 2.0 * ax * ds[j])
                v[i] = min(v[i], v_reach)

        # --- Lap time: integrate ds / v_avg over each segment ---------- #
        lap_time = 0.0
        for i in range(n):
            v0 = v[i - 1] if (i > 0 or closed) else v[i]
            v1 = v[i]
            v_avg = max(0.5 * (v0 + v1), 1e-3)
            lap_time += ds[i] / v_avg

        return lap_time, v


if __name__ == "__main__":
    from vehicle_model import NCMiata
    from track import Track

    car = NCMiata()

    for track in (Track(), Track.monza()):
        sim = LapSimulator(car, track)
        t, v = sim.solve()
        print(f"{track.name:30s} len={track.length:7.1f} m  "
              f"lap={t:7.3f} s  v_max={v.max():5.1f} m/s  v_min={v.min():4.1f} m/s")
