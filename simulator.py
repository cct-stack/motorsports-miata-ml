import numpy as np
from vehicle_model import NCMiata
from track import Track

class LapSimulator:
    def __init__(self, vehicle: NCMiata, track: Track):
        self.vehicle = vehicle
        self.track = track

    def solve(self):
        """
        Runs the quasi-steady-state simulation.
        1. Find max cornering speed at every point.
        2. Integrate forward (accel) and backward (braking).
        """
        v_max_corner = np.sqrt(self.vehicle.get_max_lat_accel(0) / np.maximum(self.track.curvature, 1e-6))
        
        # For v1, let's just assume we can stay at v_max_corner (pure cornering test)
        # or do a very simple integration.
        
        # Simple lap time: sum(ds / v)
        ds = np.diff(self.track.s, prepend=0)
        v = v_max_corner
        
        # Limit v by powertrain (simplified top speed)
        v_limit_power = 50.0 # m/s (~180 km/h)
        v = np.minimum(v, v_limit_power)
        
        lap_time = np.sum(ds / np.maximum(v, 1.0))
        return lap_time, v

if __name__ == "__main__":
    car = NCMiata()
    track = Track()
    sim = LapSimulator(car, track)
    t, v = sim.solve()
    print(f"Lap Time on {track.name}: {t:.3f}s")
