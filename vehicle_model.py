import numpy as np
from tire_model import PacejkaTire

class NCMiata:
    """
    Physical model of an NC Mazda MX-5 Miata for lap time simulation.
    Units: SI (kg, m, s, N, etc.)
    """
    def __init__(self):
        # --- Basic Dimensions ---
        self.mass = 1130.0         # kg (car + driver)
        self.wheelbase = 2.330      # m
        self.track_width = 1.490    # m (average front/rear)
        self.cg_height = 0.450      # m (estimate for a lowered NC)
        self.weight_dist_f = 0.52   # 52% front (typical NC)
        
        # --- Suspension (Tunable Parameters) ---
        # Springs (N/m) - Defaulting to a common street/track setup (~7k/4k)
        self.spring_k_f = 70000.0   
        self.spring_k_r = 40000.0
        
        # Anti-Roll Bars (Equivalent stiffness at wheel, N/m)
        self.arb_k_f = 15000.0
        self.arb_k_r = 5000.0
        
        # --- Aerodynamics ---
        self.cl_a = 0.0             # Lift coefficient * Area (negligible for street NC)
        self.cd_a = 0.35 * 1.8      # Drag coefficient * Frontal Area (approx)
        self.rho = 1.225            # Air density (kg/m^3)

        # --- Powertrain ---
        self.power_max = 125000     # Watts (~167 hp)
        self.efficiency = 0.85      # Drivetrain efficiency

        # --- Tire ---
        # Pacejka Magic Formula tire. Default is the Bridgestone RE-71RS preset
        # (the tire currently on the car), shaped to its documented behavior.
        # Swap in real TTC/Calspan-fit coefficients via PacejkaTire.fit_from_raw
        # or PacejkaTire.from_dict when you have measured data.
        self.tire = PacejkaTire.re71rs()
        self.tyre_radius = 0.306    # m (205/50R16 rolling radius)
        self.Cr = 0.015             # rolling resistance coefficient

        # --- Engine / Drivetrain (NC2 2.0L MZR, 6-speed) ---
        # Torque curve (rpm, Nm) at the crank. Peak power ~125 kW @ 7000 rpm.
        self.torque_curve = [
            (1000, 130), (2000, 160), (3000, 176), (4000, 184),
            (5000, 188), (6000, 182), (7000, 170), (7500, 150), (7750, 120),
        ]
        self.gear_ratios  = [3.136, 1.888, 1.330, 1.000, 0.814, 0.657]
        self.final_drive  = 4.10
        self.primary_ratio = 1.0
        self.primary_eff  = 1.0
        self.gear_eff     = 0.97
        self.final_eff    = 0.97
        self.shift_time   = 0.2     # s (unused by the QSS solver, kept for export)

        # Tractive-force-vs-speed lookup (built from torque curve + gearing),
        # so the lap solver does not redo the gear optimisation every step.
        self._build_engine_curve()

    def get_static_loads(self):
        """Returns static vertical load [F, R] in Newtons."""
        g = 9.81
        f_load = self.mass * g * (1 - self.weight_dist_f)
        r_load = self.mass * g * self.weight_dist_f
        # Note: weight_dist_f is fraction of weight on front. 
        # Actually dist_f = 0.52 means 52% on front.
        f_load = self.mass * g * self.weight_dist_f
        r_load = self.mass * g * (1 - self.weight_dist_f)
        return np.array([f_load / 2, f_load / 2, r_load / 2, r_load / 2])

    def tire_model(self, vertical_load):
        """
        Peak friction coefficient (mu) at a given vertical load.
        Delegates to the Pacejka Magic Formula tire, which captures real,
        nonlinear load sensitivity (peak mu falls as the tire is overloaded).
        """
        return float(self.tire.peak_mu(vertical_load))

    def get_max_lat_accel(self, velocity):
        """
        Calculates maximum lateral acceleration (m/s^2) for a given velocity.
        Iteratively finds the limit where tire grip equals required centripetal force.
        """
        g = 9.81

        # Roll Stiffness (Simplified)
        # For a real model, we'd use motion ratios, but we'll use wheel rates here.
        k_phi_f = self.spring_k_f + self.arb_k_f
        k_phi_r = self.spring_k_r + self.arb_k_r
        roll_dist_f = k_phi_f / (k_phi_f + k_phi_r)

        # Iterate to find max ay
        ay = 1.0 # start at 1g
        for _ in range(10):
            # Total Lateral Load Transfer
            delta_fz_total = (self.mass * ay * self.cg_height) / self.track_width

            # Distributed to front and rear
            delta_fz_f = delta_fz_total * roll_dist_f
            delta_fz_r = delta_fz_total * (1 - roll_dist_f)

            # Individual Tire Loads
            static = self.get_static_loads() # [FL, FR, RL, RR]
            loads = np.array([
                static[0] + delta_fz_f / 2, # Outside Front
                static[1] - delta_fz_f / 2, # Inside Front
                static[2] + delta_fz_r / 2, # Outside Rear
                static[3] - delta_fz_r / 2  # Inside Rear
            ])

            # Grip per axle
            mu_f = (self.tire_model(loads[0]) * loads[0] + self.tire_model(loads[1]) * loads[1]) / (loads[0] + loads[1])
            mu_r = (self.tire_model(loads[2]) * loads[2] + self.tire_model(loads[3]) * loads[3]) / (loads[2] + loads[3])

            # The car is limited by the weakest axle (simplified)
            # In reality, it would understeer or oversteer.
            ay_new = min(mu_f, mu_r) * g

            if abs(ay_new - ay) < 0.01:
                break
            ay = ay_new

        return ay

    # ------------------------------------------------------------------ #
    # Longitudinal performance (acceleration / braking)
    # ------------------------------------------------------------------ #
    def _build_engine_curve(self, v_top=95.0, n=400):
        """Precompute max tractive force at the wheels vs. vehicle speed.

        For each speed, the best gear is the one giving the highest wheel
        force without over-revving the engine (same logic as OpenVEHICLE's
        driveline model). Below the torque curve's lowest rpm we clamp to it
        (launch). Result is cached as a (v_grid, F_grid) lookup table.
        """
        rpm_pts = np.array([p[0] for p in self.torque_curve], dtype=float)
        tq_pts  = np.array([p[1] for p in self.torque_curve], dtype=float)
        rpm_min, rpm_max = rpm_pts[0], rpm_pts[-1]
        eff = self.primary_eff * self.gear_eff * self.final_eff

        v_grid = np.linspace(0.1, v_top, n)
        F_grid = np.zeros_like(v_grid)
        for k, v in enumerate(v_grid):
            best = 0.0
            for ratio in self.gear_ratios:
                total = ratio * self.final_drive * self.primary_ratio
                rpm = v / self.tyre_radius * total * 60.0 / (2.0 * np.pi)
                if rpm > rpm_max:
                    continue  # would over-rev in this gear
                tq = np.interp(max(rpm, rpm_min), rpm_pts, tq_pts)
                f_wheel = tq * total * eff / self.tyre_radius
                if f_wheel > best:
                    best = f_wheel
            F_grid[k] = best
        self._v_grid = v_grid
        self._F_grid = F_grid

    def tractive_force(self, v):
        """Max engine tractive force at the driven wheels (N) at speed v."""
        return float(np.interp(abs(v), self._v_grid, self._F_grid))

    def drag_force(self, v):
        """Aerodynamic drag force (N)."""
        return 0.5 * self.rho * self.cd_a * v * v

    def rolling_resistance(self):
        """Rolling resistance force (N)."""
        return self.Cr * self.mass * 9.81

    def get_max_long_accel(self, v, ay, ay_max):
        """Max forward acceleration (m/s^2) at speed v while using lateral
        acceleration ay. Couples lateral and longitudinal grip with a friction
        circle, then takes the lesser of tyre-limited and engine-limited drive,
        minus drag and rolling resistance."""
        if ay_max <= 0.0:
            return 0.0
        frac = np.sqrt(max(1.0 - min(ay / ay_max, 1.0) ** 2, 0.0))
        ax_tyre   = ay_max * frac
        ax_engine = self.tractive_force(v) / self.mass
        ax = min(ax_engine, ax_tyre)
        ax -= (self.drag_force(v) + self.rolling_resistance()) / self.mass
        return ax

    def get_max_decel(self, v, ay, ay_max):
        """Max braking deceleration (m/s^2, positive) at speed v while using
        lateral acceleration ay. All four tyres brake (friction circle); drag
        and rolling resistance assist."""
        if ay_max <= 0.0:
            ay_max = 1e-6
        frac = np.sqrt(max(1.0 - min(ay / ay_max, 1.0) ** 2, 0.0))
        ax_tyre = ay_max * frac
        return ax_tyre + (self.drag_force(v) + self.rolling_resistance()) / self.mass
