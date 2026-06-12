"""Centralised vehicle parameter set.

``VehicleConfig`` is the single source of truth for every NCMiata default. It
lets you describe a car (or a setup variant) as data and hand it to the model,
instead of editing constants inside the class body. ``NCMiata`` copies these
fields onto itself in ``__init__`` so the flat attribute API (``car.spring_k_f``,
``car.cl_a``, ...) that the DoE sampler, optimizer and exporter rely on is
unchanged.

Example
-------
    base = VehicleConfig()
    stiff = replace(base, spring_k_f=90_000.0, arb_k_f=20_000.0)
    car = NCMiata(stiff)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from tire_model import PacejkaTire


def _default_torque_curve() -> List[Tuple[int, int]]:
    # NC2 2.0L MZR (rpm, Nm) at the crank; peak power ~125 kW @ 7000 rpm.
    return [
        (1000, 130), (2000, 160), (3000, 176), (4000, 184),
        (5000, 188), (6000, 182), (7000, 170), (7500, 150), (7750, 120),
    ]


def _default_gear_ratios() -> List[float]:
    return [3.136, 1.888, 1.330, 1.000, 0.814, 0.657]


@dataclass
class VehicleConfig:
    """All vehicle parameters in SI units (kg, m, s, N). Defaults model a
    lightly-prepped NC Miata on Bridgestone RE-71RS."""

    # --- Identity ---
    name: str = "NC Miata"

    # --- Basic dimensions ---
    mass: float = 1130.0            # kg (car + driver)
    wheelbase: float = 2.330        # m
    cg_height: float = 0.450        # m (lowered NC estimate)
    weight_dist_f: float = 0.52     # fraction of weight on the front axle

    # --- Suspension (tunable) ---
    spring_k_f: float = 70_000.0    # front spring rate at the spring (N/m)
    spring_k_r: float = 40_000.0    # rear spring rate (N/m)
    arb_k_f: float = 15_000.0       # front ARB equivalent wheel rate (N/m)
    arb_k_r: float = 5_000.0        # rear ARB equivalent wheel rate (N/m)

    # --- Suspension geometry / installation ---
    motion_ratio_f: float = 0.95    # wheel travel / spring travel, front
    motion_ratio_r: float = 0.90    # rear
    track_f: float = 1.490          # front track (m)
    track_r: float = 1.490          # rear track (m)
    roll_center_height_f: float = 0.050  # front roll-centre height (m)
    roll_center_height_r: float = 0.075  # rear roll-centre height (m)

    # --- Unsprung mass (per axle) ---
    unsprung_mass_f: float = 40.0   # kg
    unsprung_mass_r: float = 40.0   # kg
    unsprung_cg_height: float = 0.280  # m (~ wheel centre)

    # --- Aerodynamics ---
    cl_a: float = 0.0               # downforce coeff * area (0 = stock NC)
    aero_balance_f: float = 0.45    # fraction of downforce on the front axle
    cd_a: float = 0.35 * 1.8        # drag coeff * frontal area (~0.63)
    rho: float = 1.225              # air density (kg/m^3)

    # --- Powertrain (descriptive spec) ---
    power_max: float = 125_000.0    # W (~167 hp)
    efficiency: float = 0.85        # overall drivetrain efficiency

    # --- Tire ---
    tire: PacejkaTire = field(default_factory=PacejkaTire.re71rs)
    tyre_radius: float = 0.306      # m (205/50R16 rolling radius)
    Cr: float = 0.015               # rolling-resistance coefficient

    # --- Engine / drivetrain (NC2 2.0L MZR, 6-speed) ---
    torque_curve: List[Tuple[int, int]] = field(default_factory=_default_torque_curve)
    gear_ratios: List[float] = field(default_factory=_default_gear_ratios)
    final_drive: float = 4.10
    primary_ratio: float = 1.0
    primary_eff: float = 1.0
    gear_eff: float = 0.97
    final_eff: float = 0.97
    shift_time: float = 0.2         # s (kept for export; unused by QSS solver)
