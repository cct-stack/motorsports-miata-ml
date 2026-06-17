"""Unit + property tests for the tire model and track geometry.

Characterization values are pinned from the validated baseline so any future
refactor that changes the physics will fail loudly here.
"""
import numpy as np
import pytest

from tire_model import PacejkaTire
from track import Track


@pytest.fixture(scope="module")
def monza_track():
    return Track.monza()


# --------------------------------------------------------------------------- #
# Tire
# --------------------------------------------------------------------------- #
def test_re71rs_peak_mu_ground_truth():
    tire = PacejkaTire.re71rs()
    assert tire.Fz0 == 2500.0
    assert "RE-71RS" in tire.name
    assert abs(float(tire.peak_mu(1000)) - 1.566000) < 1e-3
    assert abs(float(tire.peak_mu(2500)) - 1.500000) < 1e-3
    assert abs(float(tire.peak_mu(3500)) - 1.456000) < 1e-3
    assert abs(float(tire.peak_mu(5000)) - 1.390000) < 1e-3


def test_track_200tw_ground_truth():
    tire = PacejkaTire.track_200tw()
    assert tire.Fz0 == 2200.0
    assert abs(float(tire.peak_mu(2200)) - 1.450000) < 1e-3


def test_peak_mu_load_sensitivity_monotonic():
    tire = PacejkaTire.re71rs()
    fz = [1000, 2000, 3000, 4000, 5000]
    mu = [float(tire.peak_mu(f)) for f in fz]
    for a, b in zip(mu, mu[1:]):
        assert b < a  # peak grip falls as the tyre is overloaded


def test_peak_mu_floor_and_array():
    tire = PacejkaTire.re71rs()
    assert float(tire.peak_mu(50000)) >= 0.1  # never collapses below the floor
    fz = np.array([1000.0, 2000.0, 3000.0])
    mu = tire.peak_mu(fz)
    assert isinstance(mu, np.ndarray)
    assert mu.shape == fz.shape


def test_lateral_force_zero_slip_is_zero():
    tire = PacejkaTire.re71rs()
    assert abs(float(tire.lateral_force(0.0, 2500.0))) < 1e-6


def test_lateral_force_has_single_interior_peak():
    tire = PacejkaTire.re71rs()
    alpha = np.radians(np.linspace(0.0, 14.0, 100))
    fy = tire.lateral_force(alpha, tire.Fz0)
    peak = int(np.argmax(fy))
    assert 0 < peak < len(fy) - 1  # peak is interior, not at an endpoint


def test_tire_dict_round_trip():
    tire = PacejkaTire.re71rs()
    clone = PacejkaTire.from_dict(tire.to_dict())
    assert clone.Fz0 == tire.Fz0
    assert clone.name == tire.name
    assert clone.c == tire.c


# --------------------------------------------------------------------------- #
# Track
# --------------------------------------------------------------------------- #
def test_skidpad_default_geometry():
    trk = Track()
    assert trk.config == "Closed"
    assert len(trk.s) == 200
    assert len(trk.curvature) == 200
    assert abs(trk.length - 2.0 * np.pi * 30.0) < 1e-9
    assert abs(trk.curvature[0] - 1.0 / 30.0) < 1e-6
    assert np.allclose(trk.curvature, 1.0 / 30.0)


def test_skidpad_radius_scales():
    trk = Track(radius=50.0)
    assert abs(trk.length - 2.0 * np.pi * 50.0) < 1e-9
    assert np.allclose(trk.curvature, 1.0 / 50.0)


def test_skidpad_mesh_size():
    trk = Track(n=100)
    assert len(trk.s) == 100
    assert len(trk.curvature) == 100


def test_monza_parse(monza_track):
    assert monza_track.config == "Closed"
    assert len(monza_track.s) == 2878
    assert abs(monza_track.length - 5755.122) < 1.0
    assert np.any(monza_track.curvature > 0)   # left corners
    assert np.any(monza_track.curvature < 0)   # right corners
