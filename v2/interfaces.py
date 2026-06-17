"""Structural interfaces (PEP 544 Protocols) for the lap-time solver.

The simulator depends on these *behaviours*, not on the concrete ``NCMiata`` /
``Track`` classes. Any object that provides the same attributes/methods (e.g. a
different vehicle, a synthetic track, a test double) can be dropped in without
changing the solver. ``NCMiata`` and ``Track`` satisfy these structurally, so
nothing needs to inherit from them.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VehicleModel(Protocol):
    """Everything the lap solver asks of a vehicle."""

    def build_grip_curve(self, v_top: float = ...) -> None:
        """(Re)build the cached speed-dependent lateral-grip lookup."""
        ...

    def max_lat_accel(self, v):
        """Max lateral acceleration [m/s^2] at speed ``v`` (scalar or array)."""
        ...

    def get_max_long_accel(self, v: float, ay: float, ay_max: float) -> float:
        """Max forward acceleration [m/s^2] at ``v`` given lateral usage ``ay``."""
        ...

    def get_max_decel(self, v: float, ay: float, ay_max: float) -> float:
        """Max braking deceleration [m/s^2, positive] at ``v`` given ``ay``."""
        ...


@runtime_checkable
class TrackModel(Protocol):
    """Track geometry the lap solver integrates over."""

    s: np.ndarray          # distance along the line [m]
    curvature: np.ndarray  # signed curvature [1/m] (+left / -right)
    length: float          # total track length [m]
    config: str            # "Closed" (loop) or "Open" (point-to-point)
