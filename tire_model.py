"""
Pacejka "Magic Formula" lateral tire model.

This is how real tire data is actually consumed. Raw Calspan/flat-trac sweeps
(lateral force Fy vs. slip angle, repeated across vertical loads Fz) are fit to
the Magic Formula; the resulting coefficients are what every serious sim
(including Assetto Corsa) uses internally.

Two ways to get coefficients:
  1. Representative 200-treadwear track-tire values (PacejkaTire.track_200tw()).
     These are literature-shaped defaults, NOT a claim of a specific measured
     tire. They give physically correct, nonlinear load sensitivity.
  2. PacejkaTire.fit_from_raw(): if you have TTC/Calspan raw data (NDA-locked,
     available to FSAE teams), drop in arrays of (slip_angle, Fz, Fy) and this
     fits the real coefficients for your exact tire.

Pure-slip lateral form (no camber/conicity), per Pacejka MF 5.2:
    dfz = (Fz - Fz0) / Fz0
    mu_y = pDy1 + pDy2 * dfz          # peak friction coefficient (load sensitive)
    D    = mu_y * Fz                  # peak lateral force
    C    = pCy1                       # shape factor
    K    = pKy1 * Fz0 * sin(2*atan(Fz/(pKy2*Fz0)))   # cornering stiffness
    B    = K / (C * D)                # stiffness factor
    E    = pEy1 + pEy2 * dfz          # curvature factor
    Fy   = D * sin(C * atan(B*a - E*(B*a - atan(B*a))))
"""

import numpy as np


class PacejkaTire:
    def __init__(self, coeffs: dict, Fz0: float = 2200.0, name: str = "tire"):
        # Fz0 = nominal/reference vertical load (N) the coefficients are scaled to.
        self.Fz0 = Fz0
        self.name = name
        self.c = dict(coeffs)

    # ------------------------------------------------------------------ #
    @classmethod
    def track_200tw(cls):
        """Representative coefficients for a grippy 200-treadwear track tire
        (e.g. RE-71R / RT660 class) at a Miata corner-load range (~2200 N)."""
        coeffs = {
            "pCy1": 1.50,    # shape factor
            "pDy1": 1.45,    # peak mu at nominal load
            "pDy2": -0.10,   # peak mu falloff per unit dfz (load sensitivity)
            "pKy1": 18.0,    # cornering stiffness magnitude
            "pKy2": 2.0,     # load at which stiffness peaks (xFz0)
            "pEy1": -0.50,   # curvature at nominal load
            "pEy2": 0.0,     # curvature load dependence
        }
        return cls(coeffs, Fz0=2200.0, name="200TW_track_representative")

    @classmethod
    def re71rs(cls, Fz0: float = 2500.0):
        """Bridgestone Potenza RE-71RS, shaped to the tire's documented
        characteristics (NOT measured Calspan coefficients):
          - top-of-class peak grip for a 200TW tire (pDy1 high)
          - stiff carcass / high cornering stiffness, so it reaches peak at a
            relatively LOW slip angle (pKy1 high)
          - sharp, peaky breakaway rather than a broad plateau (pCy1 high)
          - meaningful load sensitivity (grip drops as the tire is overloaded)
        Fz0 defaults to ~2500 N to center load sensitivity on an NC Miata
        corner load. Replace with PacejkaTire.fit_from_raw() if you obtain
        real swept data for your exact size/pressure."""
        coeffs = {
            "pCy1": 1.60,    # sharp post-peak falloff (peaky breakaway)
            "pDy1": 1.50,    # benchmark peak mu for the class
            "pDy2": -0.11,   # load sensitivity
            "pKy1": 36.0,    # high cornering stiffness -> peak near ~7 deg slip
            "pKy2": 2.0,
            "pEy1": -0.40,
            "pEy2": 0.0,
        }
        return cls(coeffs, Fz0=Fz0, name="Bridgestone_RE-71RS_representative")

    # ------------------------------------------------------------------ #
    def peak_mu(self, Fz):
        """Effective peak friction coefficient at vertical load Fz (N).

        This matches the old tire_model(load) interface so the rest of the
        vehicle model keeps working unchanged."""
        Fz = np.maximum(np.asarray(Fz, dtype=float), 1.0)
        dfz = (Fz - self.Fz0) / self.Fz0
        mu = self.c["pDy1"] + self.c["pDy2"] * dfz
        return np.clip(mu, 0.1, None)

    def lateral_force(self, alpha, Fz):
        """Full Magic Formula lateral force Fy (N).

        alpha: slip angle (rad). Fz: vertical load (N)."""
        Fz = np.maximum(np.asarray(Fz, dtype=float), 1.0)
        alpha = np.asarray(alpha, dtype=float)
        dfz = (Fz - self.Fz0) / self.Fz0

        mu = self.c["pDy1"] + self.c["pDy2"] * dfz
        D = mu * Fz
        C = self.c["pCy1"]
        K = self.c["pKy1"] * self.Fz0 * np.sin(
            2.0 * np.arctan(Fz / (self.c["pKy2"] * self.Fz0)))
        B = K / (C * D)
        E = self.c["pEy1"] + self.c["pEy2"] * dfz

        Ba = B * alpha
        return D * np.sin(C * np.arctan(Ba - E * (Ba - np.arctan(Ba))))

    # ------------------------------------------------------------------ #
    def to_dict(self):
        return {"coeffs": self.c, "Fz0": self.Fz0, "name": self.name}

    @classmethod
    def from_dict(cls, d: dict):
        return cls(d["coeffs"], Fz0=d.get("Fz0", 2200.0), name=d.get("name", "tire"))

    # ------------------------------------------------------------------ #
    @classmethod
    def fit_from_raw(cls, slip_angle, Fz, Fy, Fz0: float = 2200.0,
                     name: str = "fitted"):
        """Fit Magic Formula coefficients to real raw tire data.

        Pass 1-D arrays of slip_angle (rad), Fz (N), Fy (N) sampled across the
        load sweep (e.g. exported from a TTC/Calspan run). Returns a fitted
        PacejkaTire. Requires scipy.
        """
        from scipy.optimize import curve_fit

        slip_angle = np.asarray(slip_angle, dtype=float)
        Fz = np.asarray(Fz, dtype=float)
        Fy = np.asarray(Fy, dtype=float)

        def model(X, pCy1, pDy1, pDy2, pKy1, pKy2, pEy1, pEy2):
            a, fz = X
            dfz = (fz - Fz0) / Fz0
            D = (pDy1 + pDy2 * dfz) * fz
            K = pKy1 * Fz0 * np.sin(2.0 * np.arctan(fz / (pKy2 * Fz0)))
            B = K / (pCy1 * D)
            E = pEy1 + pEy2 * dfz
            Ba = B * a
            return D * np.sin(pCy1 * np.arctan(Ba - E * (Ba - np.arctan(Ba))))

        p0 = [1.5, 1.45, -0.10, 18.0, 2.0, -0.5, 0.0]
        bounds = ([1.0, 0.5, -1.0, 1.0, 0.5, -3.0, -1.0],
                  [2.0, 3.0, 1.0, 60.0, 6.0, 1.0, 1.0])
        popt, _ = curve_fit(model, (slip_angle, Fz), Fy, p0=p0,
                            bounds=bounds, maxfev=20000)
        keys = ["pCy1", "pDy1", "pDy2", "pKy1", "pKy2", "pEy1", "pEy2"]
        return cls(dict(zip(keys, popt)), Fz0=Fz0, name=name)


if __name__ == "__main__":
    tire = PacejkaTire.re71rs()
    print(f"Tire: {tire.name} (Fz0={tire.Fz0:.0f} N)")
    for fz in [1000, 2500, 3500, 5000]:
        print(f"  Fz={fz:5d} N  ->  peak mu = {float(tire.peak_mu(fz)):.3f}")
    # Lateral force at nominal load, sweeping slip angle (find the peak):
    a = np.radians(np.linspace(0, 14, 15))
    fy = tire.lateral_force(a, tire.Fz0)
    peak_i = int(np.argmax(fy))
    print(f"  Peak Fy @ Fz0: {fy[peak_i]:.0f} N at {np.degrees(a[peak_i]):.1f} deg slip")
    print("  Fy vs slip(deg):",
          {round(np.degrees(ai), 1): round(float(fi), 0)
           for ai, fi in zip(a[::2], fy[::2])})
