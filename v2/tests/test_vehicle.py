"""Unit + property tests for the NCMiata vehicle model.

Pins the validated baseline numbers and checks the physics invariants
(load-transfer symmetry, downforce v^2 scaling, friction circle, backward
compatibility of the aero coupling).
"""
import numpy as np
import pytest

from vehicle_model import NCMiata

G = 9.81
AY0 = 13.362284  # baseline get_max_lat_accel(0.0) for default car


@pytest.fixture
def car():
    return NCMiata()


@pytest.fixture
def aero_car():
    c = NCMiata()
    c.cl_a = 2.0
    c.build_grip_curve()
    return c


# --------------------------------------------------------------------------- #
# Static / steady-state characterization
# --------------------------------------------------------------------------- #
def test_static_loads_ground_truth(car):
    loads = car.get_static_loads()
    np.testing.assert_allclose(
        loads, [2882.178, 2882.178, 2660.472, 2660.472], atol=1e-3)
    assert abs(loads.sum() - car.mass * G) / (car.mass * G) < 1e-9
    front, rear = loads[0] + loads[1], loads[2] + loads[3]
    assert abs(front - car.mass * G * 0.52) / (car.mass * G) < 1e-9
    assert abs(rear - car.mass * G * 0.48) / (car.mass * G) < 1e-9


def test_static_loads_symmetry(car):
    loads = car.get_static_loads()
    assert np.all(loads > 0)
    assert abs(loads[0] - loads[1]) < 1e-6  # FL == FR
    assert abs(loads[2] - loads[3]) < 1e-6  # RL == RR


def test_roll_stiffness_ground_truth(car):
    kf, kr = car.roll_stiffness()
    assert abs(kf - 86778.1588) < 1e-3
    assert abs(kr - 41515.8700) < 1e-3
    assert abs(kf / (kf + kr) - 0.676401) < 1e-4


def test_roll_stiffness_front_stiffening(car):
    kf0, kr0 = car.roll_stiffness()
    car.spring_k_f *= 2.0
    kf1, kr1 = car.roll_stiffness()
    assert kf1 > kf0
    assert kr1 == kr0
    assert kf1 / (kf1 + kr1) > kf0 / (kf0 + kr0)


def test_max_lat_accel_ground_truth(car):
    assert abs(car.get_max_lat_accel(0.0) - AY0) < 1e-3


# --------------------------------------------------------------------------- #
# Aerodynamics
# --------------------------------------------------------------------------- #
def test_resistive_forces_ground_truth(car):
    assert car.downforce(0.0) == 0.0
    assert abs(car.drag_force(30.0) - 347.2875) < 1e-3
    assert abs(car.rolling_resistance() - 166.2795) < 1e-3


def test_tractive_force_ground_truth(car):
    assert abs(car.tractive_force(0.0) - 5139.5525) < 1.0
    assert abs(car.tractive_force(30.0) - 3811.1466) < 1.0


def test_downforce_v_squared_scaling(aero_car):
    assert abs(aero_car.downforce(40.0) - 4.0 * aero_car.downforce(20.0)) < 1e-9
    assert abs(aero_car.downforce(20.0) - 490.0) < 1e-3
    assert abs(aero_car.downforce(40.0) - 1960.0) < 1e-3
    assert abs(aero_car.downforce(60.0) - 4410.0) < 1e-3
    assert abs(aero_car.downforce(80.0) - 7840.0) < 1e-3


def test_drag_v_squared_scaling(car):
    assert abs(car.drag_force(40.0) - 4.0 * car.drag_force(20.0)) < 1e-9


def test_aero_grip_curve_ground_truth(aero_car):
    expected = {0: 13.362284, 20: 13.859472, 40: 15.306651,
                60: 17.595468, 80: 20.561861}
    for v, ay in expected.items():
        assert abs(float(aero_car.max_lat_accel(v)) - ay) < 1e-2


def test_aero_grip_increases_with_speed(aero_car):
    ay = [aero_car.get_max_lat_accel(v) for v in (0, 20, 40, 60, 80)]
    for a, b in zip(ay, ay[1:]):
        assert b > a  # downforce raises grip with speed


def test_aero_zero_speed_matches_no_downforce(aero_car, car):
    assert abs(aero_car.get_max_lat_accel(0.0) - car.get_max_lat_accel(0.0)) < 1e-6


def test_no_downforce_grip_is_flat():
    c = NCMiata()  # cl_a == 0
    c.build_grip_curve()
    base = float(c.max_lat_accel(0.0))
    for v in (0, 20, 40, 80):
        assert abs(float(c.max_lat_accel(v)) - base) < 1e-6


# --------------------------------------------------------------------------- #
# Longitudinal coupling (friction circle)
# --------------------------------------------------------------------------- #
def test_long_accel_ground_truth(car):
    assert abs(car.get_max_long_accel(10.0, 0.0, AY0) - 6.255526) < 1e-2


def test_decel_ground_truth(car):
    assert abs(car.get_max_decel(30.0, 0.0, AY0) - 13.816768) < 1e-2


def test_friction_circle_saturation(car):
    v = 30.0
    ay_max = car.get_max_lat_accel(0.0)
    ax = car.get_max_long_accel(v, ay_max, ay_max)  # no longitudinal grip left
    expected = -(car.drag_force(v) + car.rolling_resistance()) / car.mass
    assert abs(ax - expected) < 1e-3


def test_long_accel_decreases_with_lateral(car):
    v = 20.0
    ay_max = car.get_max_lat_accel(0.0)
    ax = [car.get_max_long_accel(v, ay, ay_max)
          for ay in (0.0, ay_max / 4, ay_max / 2, ay_max)]
    for a, b in zip(ax, ax[1:]):
        assert b <= a + 1e-9


def test_tractive_force_non_negative(car):
    for v in (1.0, 10.0, 30.0, 60.0):
        assert car.tractive_force(v) >= 0.0
