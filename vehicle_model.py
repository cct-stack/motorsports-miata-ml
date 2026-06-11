import numpy as np

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
        Simplified tire model with load sensitivity.
        Returns peak friction coefficient (mu).
        mu = mu_max * (1 - sensitivity * load)
        """
        mu_max = 1.3
        sensitivity = 0.00005 # per Newton
        mu = mu_max * (1 - sensitivity * vertical_load)
        return max(0.1, mu)

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
