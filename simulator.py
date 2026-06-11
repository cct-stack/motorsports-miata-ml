import numpy as np
from vehicle_model import NCMiata
from track import Track


class LapSimulator:
    """Quasi-steady-state lap-time solver.

    Uses the same method as OpenLAP: build a per-point cornering speed limit
    from the lateral-grip envelope, then make a forward pass (acceleration,
    engine- and tyre-limited) and a backward pass (braking, tyre-limited). The
    final speed at each point is the minimum the car can satisfy in both
    directions. Longitudinal and lateral grip are coupled with a friction
    circle inside the vehicle model.
    """

    def __init__(self, vehicle: NCMiata, track: Track, v_cap=100.0):
        self.vehicle = vehicle
        self.track = track
        self.v_cap = v_cap  # upper bound on straight-line speed search [m/s]

    # ------------------------------------------------------------------ #
    def _corner_speed_limit(self, ay_max):
        """Max speed at each point from pure lateral grip: v = sqrt(ay_max/kappa).
        On near-straight sections (kappa ~ 0) the limit is the search cap."""
        kappa = np.abs(self.track.curvature)
        with np.errstate(divide="ignore"):
            v = np.sqrt(ay_max / np.maximum(kappa, 1e-9))
        return np.minimum(v, self.v_cap)

    def _segment_lengths(self):
        """Distance to the previous point at each index. For a closed track the
        first entry wraps around from the last point."""
        ds = np.diff(self.track.s)
        if self.track.config == "Closed":
            wrap = self.track.length - self.track.s[-1] + self.track.s[0]
            ds = np.concatenate(([max(wrap, 1e-6)], ds))
        else:
            ds = np.concatenate(([ds[0]], ds))
        return ds

    def solve(self):
        """Return (lap_time [s], speed_profile [m/s] at each track point)."""
        v_corner = self._corner_speed_limit(self.vehicle.get_max_lat_accel(0.0))
        ds = self._segment_lengths()
        n = len(v_corner)
        ay_max = self.vehicle.get_max_lat_accel(0.0)
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
    car = NCMiata()

    for track in (Track(), Track.monza()):
        sim = LapSimulator(car, track)
        t, v = sim.solve()
        print(f"{track.name:30s} len={track.length:7.1f} m  "
              f"lap={t:7.3f} s  v_max={v.max():5.1f} m/s  v_min={v.min():4.1f} m/s")
