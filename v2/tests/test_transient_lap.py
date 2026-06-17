"""Tests for the QSS<->7-DOF coupling (transient_lap.py).

These lock the behaviour that makes the transient layer worthwhile:
  * a constant-radius skidpad has no transients, so the damper penalty is ~0;
  * a transition-rich slalom incurs a real (>0) penalty;
  * a better-damped car pays a smaller penalty there;
  * the whole pipeline is deterministic.

A short synthetic slalom keeps the time-domain integration cheap enough for a
unit test (Monza-scale laps take tens of seconds and belong in the demo).
"""
import numpy as np
import pytest

from config import VehicleConfig
from track import Track
from vehicle_model import NCMiata
from transient_lap import run_transient_lap


def _slalom(n_pairs=2):
    """A closed left/right slalom: many sharp transitions, little straight."""
    types, lengths, radii = [], [], []
    for _ in range(n_pairs):
        types += ["Left", "Straight", "Right", "Straight"]
        lengths += [22, 16, 22, 16]
        radii += [16.0, 0.0, 16.0, 0.0]
    return Track.from_shape_data(types, lengths, radii, name="Slalom",
                                 config="Closed", mesh_size=3.0)


SOFT = VehicleConfig(damper_bump_f=400, damper_rebound_f=700,
                     damper_bump_r=400, damper_rebound_r=700)
STIFF = VehicleConfig(damper_bump_f=3000, damper_rebound_f=5000,
                      damper_bump_r=3000, damper_rebound_r=5000)


@pytest.fixture(scope="module")
def slalom():
    return _slalom()


def test_result_shapes_and_arrays():
    """grip_scale and the load arrays line up with the track points."""
    tk = Track()
    res = run_transient_lap(NCMiata(), tk, n_laps=2)
    n = len(tk.s)
    assert res.grip_scale.shape == (n,)
    assert res.tire_load_dyn.shape == (n, 4)
    assert res.tire_load_steady.shape == (n, 4)
    assert np.all(res.grip_scale <= 1.0 + 1e-9)
    assert np.all(res.grip_scale >= 0.0)


def test_skidpad_penalty_is_negligible():
    """Constant cornering -> no load-transfer transients -> ~no damper cost."""
    res = run_transient_lap(NCMiata(SOFT), Track(), n_laps=3)
    assert res.delta >= -1e-9
    assert res.delta < 0.05 * res.lap_time_qss * 0.01 + 0.05   # < ~0.05 s
    assert res.grip_scale.min() > 0.98


def test_transient_track_has_positive_penalty(slalom):
    """A transition-rich slalom must cost lap time vs the QSS baseline."""
    res = run_transient_lap(NCMiata(SOFT), slalom, n_laps=2)
    assert res.lap_time_transient > res.lap_time_qss
    assert res.delta > 0.0
    assert res.grip_scale.min() < 1.0


def test_better_damped_setup_pays_smaller_penalty(slalom):
    """Stiffer dampers control the transient load swing -> smaller penalty."""
    soft = run_transient_lap(NCMiata(SOFT), slalom, n_laps=2)
    stiff = run_transient_lap(NCMiata(STIFF), slalom, n_laps=2)
    assert stiff.delta < soft.delta
    assert stiff.grip_scale.min() > soft.grip_scale.min()


def test_penalty_is_deterministic(slalom):
    """Same inputs -> identical lap time and grip scaling (no RNG anywhere)."""
    a = run_transient_lap(NCMiata(SOFT), slalom, n_laps=2)
    b = run_transient_lap(NCMiata(SOFT), slalom, n_laps=2)
    assert a.lap_time_transient == pytest.approx(b.lap_time_transient, rel=0,
                                                 abs=1e-9)
    assert np.array_equal(a.grip_scale, b.grip_scale)


def test_grip_scaling_only_slows_the_lap(slalom):
    """Grip scale <= 1 can never make the QSS lap faster than the baseline."""
    res = run_transient_lap(NCMiata(SOFT), slalom, n_laps=2)
    assert res.lap_time_transient >= res.lap_time_qss - 1e-9
