import numpy as np

from config import VehicleConfig


class NCMiata:
    """
    Physical model of an NC Mazda MX-5 Miata for lap time simulation.
    Units: SI (kg, m, s, N, etc.)

    All scalar defaults live in ``VehicleConfig`` (see config.py). Pass a
    ``VehicleConfig`` to describe a different car or setup variant; the fields
    are copied onto the instance so the flat attribute API (``car.spring_k_f``,
    ``car.cl_a``, ...) that the DoE sampler, optimizer and exporter rely on is
    unchanged.
    """
    def __init__(self, config: VehicleConfig | None = None):
        cfg = config if config is not None else VehicleConfig()
        self.config = cfg

        # --- Basic Dimensions ---
        self.mass = cfg.mass
        self.wheelbase = cfg.wheelbase
        self.cg_height = cfg.cg_height
        self.weight_dist_f = cfg.weight_dist_f

        # --- Suspension (Tunable Parameters) ---
        # Spring rate AT THE SPRING (N/m). Converted to a wheel rate via the
        # motion ratio (wheel_rate = spring_rate * MR**2).
        self.spring_k_f = cfg.spring_k_f
        self.spring_k_r = cfg.spring_k_r

        # Anti-roll bars (equivalent wheel rate, N/m)
        self.arb_k_f = cfg.arb_k_f
        self.arb_k_r = cfg.arb_k_r

        # Suspension geometry / installation
        self.motion_ratio_f = cfg.motion_ratio_f
        self.motion_ratio_r = cfg.motion_ratio_r
        self.track_f = cfg.track_f
        self.track_r = cfg.track_r
        self.roll_center_height_f = cfg.roll_center_height_f
        self.roll_center_height_r = cfg.roll_center_height_r

        # Unsprung mass (per axle, kg) and its CG height (~ wheel centre)
        self.unsprung_mass_f = cfg.unsprung_mass_f
        self.unsprung_mass_r = cfg.unsprung_mass_r
        self.unsprung_cg_height = cfg.unsprung_cg_height

        # --- Aerodynamics ---
        # Downforce convention: cl_a is (downforce coefficient * area), so a
        # POSITIVE value pushes the car down. aero_balance_f is the fraction
        # of total downforce carried by the front axle.
        self.cl_a = cfg.cl_a
        self.aero_balance_f = cfg.aero_balance_f
        self.cd_a = cfg.cd_a
        self.rho = cfg.rho

        # --- Powertrain ---
        self.power_max = cfg.power_max
        self.efficiency = cfg.efficiency

        # --- Tire ---
        self.tire = cfg.tire
        self.tyre_radius = cfg.tyre_radius
        self.Cr = cfg.Cr

        # --- Engine / Drivetrain (NC2 2.0L MZR, 6-speed) ---
        self.torque_curve = cfg.torque_curve
        self.gear_ratios = cfg.gear_ratios
        self.final_drive = cfg.final_drive
        self.primary_ratio = cfg.primary_ratio
        self.primary_eff = cfg.primary_eff
        self.gear_eff = cfg.gear_eff
        self.final_eff = cfg.final_eff
        self.shift_time = cfg.shift_time

        # Tractive-force-vs-speed lookup (built from torque curve + gearing),
        # so the lap solver does not redo the gear optimisation every step.
        self._build_engine_curve()

        # Speed-dependent lateral-grip lookup (downforce raises grip with
        # speed). Built lazily / by the simulator so it always reflects the
        # current setup and aero parameters.
        self._v_grip = None
        self._ay_grip = None

    def get_static_loads(self):
        """Static vertical load per wheel [FL, FR, RL, RR] in Newtons.

        weight_dist_f is the fraction of weight on the FRONT axle (0.52 = 52%
        front): the front axle carries mass*g*weight_dist_f split evenly across
        its two wheels, the rear axle carries the remainder."""
        g = 9.81
        f_load = self.mass * g * self.weight_dist_f
        r_load = self.mass * g * (1.0 - self.weight_dist_f)
        return np.array([f_load / 2, f_load / 2, r_load / 2, r_load / 2])

    def tire_model(self, vertical_load):
        """
        Peak friction coefficient (mu) at a given vertical load.
        Delegates to the Pacejka Magic Formula tire, which captures real,
        nonlinear load sensitivity (peak mu falls as the tire is overloaded).
        """
        return float(self.tire.peak_mu(vertical_load))

    def roll_stiffness(self):
        """Roll stiffness (K_phi) per axle in N*m/rad.

        Springs act at the wheel through the motion ratio
        (wheel_rate = spring_rate * MR**2); two wheels resisting body roll give
        K_phi = 0.5 * wheel_rate * track**2 (small angle). The ARB equivalent
        wheel rate adds in the same way.
        """
        wr_f = self.spring_k_f * self.motion_ratio_f ** 2 + self.arb_k_f
        wr_r = self.spring_k_r * self.motion_ratio_r ** 2 + self.arb_k_r
        k_phi_f = 0.5 * wr_f * self.track_f ** 2
        k_phi_r = 0.5 * wr_r * self.track_r ** 2
        return k_phi_f, k_phi_r

    def get_max_lat_accel(self, velocity):
        """Maximum lateral acceleration (m/s^2) at a given velocity.

        Standard lateral-load-transfer-distribution model. The per-axle vertical
        load couple (per wheel) is the sum of three moment/track contributions:
          * elastic   - sprung mass rolling about the roll axis, shared between
                        axles by roll-stiffness fraction,
          * geometric - each axle's sprung lateral force reacting at its own
                        roll centre (jacking),
          * unsprung  - unsprung mass reacting at the wheel-centre height.
        The grip-limited axle (min of front/rear) sets the limit; iterated to
        convergence because peak tyre mu is load-sensitive.
        """
        g = 9.81

        k_phi_f, k_phi_r = self.roll_stiffness()
        k_phi_tot = k_phi_f + k_phi_r

        # Sprung / unsprung split (per axle, from the static weight distribution)
        m_uf, m_ur = self.unsprung_mass_f, self.unsprung_mass_r
        m_sf = self.mass * self.weight_dist_f - m_uf
        m_sr = self.mass * (1.0 - self.weight_dist_f) - m_ur
        m_s = m_sf + m_sr

        # Roll-axis height under the sprung CG, then sprung CG lever arm above it
        z_ra = (self.roll_center_height_f * self.weight_dist_f +
                self.roll_center_height_r * (1.0 - self.weight_dist_f))
        h_s = max(self.cg_height - z_ra, 0.0)

        static = self.get_static_loads()  # [FL, FR, RL, RR] (N)

        # Aerodynamic downforce is vertical and symmetric: it raises each
        # corner's base load (split front/rear by aero balance) but does not
        # itself create lateral load transfer. The transfer terms below act on
        # this downforce-augmented base.
        df = self.downforce(velocity)
        df_f = df * self.aero_balance_f
        df_r = df - df_f
        base = static + np.array([df_f / 2.0, df_f / 2.0,
                                  df_r / 2.0, df_r / 2.0])

        ay = 1.0  # start at 1g
        for _ in range(10):
            # Elastic: sprung roll moment distributed by roll-stiffness fraction
            m_roll = m_s * ay * h_s
            if k_phi_tot > 0.0:
                dfz_el_f = (m_roll * k_phi_f / k_phi_tot) / self.track_f
                dfz_el_r = (m_roll * k_phi_r / k_phi_tot) / self.track_r
            else:
                dfz_el_f = dfz_el_r = 0.0

            # Geometric: each axle's sprung lateral force at its roll centre
            dfz_geo_f = m_sf * ay * self.roll_center_height_f / self.track_f
            dfz_geo_r = m_sr * ay * self.roll_center_height_r / self.track_r

            # Unsprung: reacts at the wheel-centre height
            dfz_us_f = m_uf * ay * self.unsprung_cg_height / self.track_f
            dfz_us_r = m_ur * ay * self.unsprung_cg_height / self.track_r

            dfz_f = dfz_el_f + dfz_geo_f + dfz_us_f
            dfz_r = dfz_el_r + dfz_geo_r + dfz_us_r

            loads = np.array([
                max(base[0] + dfz_f, 0.0),  # outside front
                max(base[1] - dfz_f, 0.0),  # inside front
                max(base[2] + dfz_r, 0.0),  # outside rear
                max(base[3] - dfz_r, 0.0),  # inside rear
            ])

            denom_f = loads[0] + loads[1]
            denom_r = loads[2] + loads[3]
            if denom_f <= 0.0 or denom_r <= 0.0:
                break

            mu_f = (self.tire_model(loads[0]) * loads[0] +
                    self.tire_model(loads[1]) * loads[1]) / denom_f
            mu_r = (self.tire_model(loads[2]) * loads[2] +
                    self.tire_model(loads[3]) * loads[3]) / denom_r

            # Limited by the weaker axle (understeer/oversteer not resolved
            # here). Force-based: the achievable lateral force is mu * total
            # normal load, so ay = mu * (m*g + downforce) / m. Downforce raises
            # the normal load above the car's weight, increasing grip (less the
            # tyre load-sensitivity penalty already captured in mu). With no
            # downforce this is exactly mu * g.
            ay_new = min(mu_f, mu_r) * (g + df / self.mass)
            if abs(ay_new - ay) < 0.01:
                ay = ay_new
                break
            ay = ay_new

        return ay

    def build_grip_curve(self, v_top=95.0, n=60):
        """Tabulate max lateral acceleration vs speed.

        Because downforce grows with v^2, peak lateral grip is speed-dependent;
        this caches (v_grid, ay_grid) so the lap solver can look it up cheaply
        instead of re-solving the load-transfer fixed point at every point.
        Rebuild after changing setup, aero, or mass parameters. With cl_a = 0
        the curve is flat and the result is identical to the static solve.
        """
        v_grid = np.linspace(0.0, v_top, n)
        ay_grid = np.array([self.get_max_lat_accel(v) for v in v_grid])
        self._v_grip = v_grid
        self._ay_grip = ay_grid

    def max_lat_accel(self, v):
        """Max lateral acceleration (m/s^2) at speed v from the cached grip
        curve (built on demand). Accepts a scalar or a numpy array."""
        if self._v_grip is None:
            self.build_grip_curve()
        return np.interp(np.abs(v), self._v_grip, self._ay_grip)

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

    def downforce(self, v):
        """Aerodynamic downforce (N, positive downward) = 0.5*rho*cl_a*v^2."""
        return 0.5 * self.rho * self.cl_a * v * v

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
