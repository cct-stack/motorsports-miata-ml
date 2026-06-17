"""Tests for the 7-DOF transient ride model (transient_model.py).

These lock the physical sanity of the time-domain path: static equilibrium,
correct load-transfer direction, vertical-load conservation, and — the whole
reason the model exists — measurable damper sensitivity in the transient.
"""
import numpy as np
import pytest

from dataclasses import replace

from config import VehicleConfig
from transient_model import RideModelParams, SevenDOFRideModel, G


@pytest.fixture(scope="module")
def base_cfg():
    return VehicleConfig()


def _model(cfg, **kw):
    return SevenDOFRideModel(RideModelParams.from_config(cfg, **kw))


def test_static_equilibrium_stays_at_rest(base_cfg):
    """No excitation, no initial perturbation -> the car does not move and each
    tire carries its static load."""
    m = _model(base_cfg)
    r = m.simulate(1.0, ax=0.0, ay=0.0)
    assert np.allclose(r.z_s, 0.0, atol=1e-6)
    assert np.allclose(r.roll, 0.0, atol=1e-6)
    assert np.allclose(r.pitch, 0.0, atol=1e-6)
    assert np.allclose(r.tire_load[-1], m.p.n_static, atol=1.0)


def test_static_loads_sum_to_weight(base_cfg):
    m = _model(base_cfg)
    assert m.p.n_static.sum() == pytest.approx(base_cfg.mass * G, rel=1e-6)


def test_left_turn_loads_right_side(base_cfg):
    """A positive (leftward) lateral accel rolls the body left-side-up and
    transfers load onto the right-hand (outside) wheels."""
    m = _model(base_cfg)
    r = m.simulate(2.0, ax=0.0, ay=1.0 * G)
    fl, fr, rl, rr = r.tire_load[-1]
    assert r.roll[-1] > 0.0                 # left side up
    assert fr > fl and rr > rl              # outside (right) wheels loaded
    assert fl < m.p.n_static[0]             # inside fronts unloaded
    assert rl < m.p.n_static[2]


def test_acceleration_lifts_nose(base_cfg):
    """Forward acceleration transfers load rearward and pitches the nose up."""
    m = _model(base_cfg)
    r = m.simulate(2.0, ax=0.5 * G, ay=0.0)
    front = r.tire_load[-1, 0] + r.tire_load[-1, 1]
    rear = r.tire_load[-1, 2] + r.tire_load[-1, 3]
    front_static = m.p.n_static[0] + m.p.n_static[1]
    assert r.pitch[-1] > 0.0                # nose up
    assert front < front_static             # front unloaded
    assert rear > m.p.n_static[2] + m.p.n_static[3]


def test_vertical_load_conserved_in_pure_cornering(base_cfg):
    """Lateral load transfer is zero-sum: the four tire loads still total the
    car weight (the excitation is horizontal)."""
    m = _model(base_cfg)
    r = m.simulate(2.0, ax=0.0, ay=1.0 * G)
    assert r.tire_load[-1].sum() == pytest.approx(base_cfg.mass * G, rel=1e-3)


def test_stiffer_dampers_reduce_overshoot(base_cfg):
    """Steady-state roll is set by the springs (≈equal), but stiffer dampers
    must cut the transient roll overshoot."""
    soft = replace(base_cfg, damper_bump_f=400, damper_rebound_f=600,
                   damper_bump_r=400, damper_rebound_r=600)
    stiff = replace(base_cfg, damper_bump_f=5000, damper_rebound_f=8000,
                    damper_bump_r=5000, damper_rebound_r=8000)
    r_soft = _model(soft).simulate(4.0, ax=0.0, ay=1.0 * G)
    r_stiff = _model(stiff).simulate(4.0, ax=0.0, ay=1.0 * G)

    # Similar settled roll (springs set the final position): compare the mean
    # over the last 0.5 s so residual ringing of the soft setup averages out.
    tail = slice(-500, None)
    assert (r_soft.roll[tail].mean()
            == pytest.approx(r_stiff.roll[tail].mean(), rel=0.1))
    # But the soft setup overshoots noticeably more on the way there.
    assert r_soft.roll.max() > r_stiff.roll.max() + np.radians(0.5)


def test_asymmetric_damping_is_used(base_cfg):
    """Bump and rebound coefficients are mapped to the right corners/phase."""
    m = _model(base_cfg)
    assert np.allclose(m.p.c_bump[:2], base_cfg.damper_bump_f)
    assert np.allclose(m.p.c_reb[:2], base_cfg.damper_rebound_f)
    assert np.allclose(m.p.c_bump[2:], base_cfg.damper_bump_r)
    assert np.allclose(m.p.c_reb[2:], base_cfg.damper_rebound_r)


def test_tire_loads_nonnegative_reasonable(base_cfg):
    """Under a hard 1.2g corner no tire load goes negative (car stays planted
    on this setup) and loads stay in a physical range."""
    m = _model(base_cfg)
    r = m.simulate(2.0, ax=0.0, ay=1.2 * G)
    assert np.all(r.tire_load > -1e-6)
    assert r.tire_load.max() < base_cfg.mass * G   # no single corner > weight
