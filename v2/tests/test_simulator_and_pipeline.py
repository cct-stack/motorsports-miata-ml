"""End-to-end tests for the lap simulator and the DoE -> surrogate pipeline.

Locks the validated baseline lap times and checks determinism, the aero
backward-compatibility/benefit, and that the optimization pipeline still
produces sane, finite results through the current simulator interface.
"""
import configparser
import math
import os
import numpy as np
import pytest

from vehicle_model import NCMiata
from simulator import LapSimulator
from track import Track
from doe_sampler import run_lhs_sweep, PARAM_NAMES
from surrogate import LapTimeSurrogate
from config import VehicleConfig
from export_to_ac import export_ac_car


@pytest.fixture(scope="module")
def skidpad():
    return Track()


@pytest.fixture(scope="module")
def monza():
    return Track.monza()


@pytest.fixture(scope="module")
def sweep12():
    return run_lhs_sweep(n_samples=12, seed=1)


def _aero_car(cl_a):
    c = NCMiata()
    c.cl_a = cl_a
    return c


# --------------------------------------------------------------------------- #
# Baseline lap times (the validated reference)
# --------------------------------------------------------------------------- #
def test_skidpad_baseline(skidpad):
    lap, v = LapSimulator(NCMiata(), skidpad).solve()
    assert abs(lap - 9.417740) < 1e-2
    assert len(v) == 200
    assert np.all(v > 0) and np.all(np.isfinite(v))
    assert v.max() - v.min() < 0.1            # near-constant speed on a circle
    assert abs(v.max() - 20.0217) < 1e-2
    assert abs(v.min() - 20.0083) < 1e-2


def test_monza_baseline(monza):
    lap, v = LapSimulator(NCMiata(), monza).solve()
    assert abs(lap - 131.238218) < 5e-2
    assert len(v) == 2878
    assert np.all(v > 0) and np.all(np.isfinite(v))
    assert abs(v.max() - 59.489010) < 1e-1
    assert abs(v.min() - 17.595622) < 1e-1


def test_solver_deterministic(skidpad):
    lap1, _ = LapSimulator(NCMiata(), skidpad).solve()
    lap2, _ = LapSimulator(NCMiata(), skidpad).solve()
    assert abs(lap1 - lap2) < 1e-9


# --------------------------------------------------------------------------- #
# Aero coupling
# --------------------------------------------------------------------------- #
def test_aero_backward_compatible(skidpad):
    lap, _ = LapSimulator(_aero_car(0.0), skidpad).solve()
    assert abs(lap - 9.417740) < 1e-2


def test_aero_never_hurts(skidpad, monza):
    base_sk, _ = LapSimulator(_aero_car(0.0), skidpad).solve()
    aero_sk, _ = LapSimulator(_aero_car(2.0), skidpad).solve()
    assert aero_sk <= base_sk + 1e-6

    base_mz, _ = LapSimulator(_aero_car(0.0), monza).solve()
    aero_mz, _ = LapSimulator(_aero_car(2.0), monza).solve()
    assert aero_mz < base_mz  # downforce helps on a track with fast corners


# --------------------------------------------------------------------------- #
# DoE pipeline
# --------------------------------------------------------------------------- #
def test_doe_sweep_shape_and_range():
    df = run_lhs_sweep(n_samples=6, seed=42)
    assert len(df) == 6
    assert list(df.columns) == PARAM_NAMES + ["lap_time"]
    assert np.all(np.isfinite(df["lap_time"]))
    assert np.all((df["lap_time"] >= 8.0) & (df["lap_time"] <= 12.0))


def test_doe_sweep_deterministic():
    a = run_lhs_sweep(n_samples=6, seed=42)
    b = run_lhs_sweep(n_samples=6, seed=42)
    assert np.allclose(a["lap_time"].values, b["lap_time"].values)


# --------------------------------------------------------------------------- #
# Surrogate
# --------------------------------------------------------------------------- #
def test_surrogate_fits_training_data(sweep12):
    X = sweep12[PARAM_NAMES].values
    y = sweep12["lap_time"].values
    s = LapTimeSurrogate()
    s.fit(sweep12)
    pred = s.predict(X)
    assert len(pred) == len(y)
    assert np.all(np.isfinite(pred))
    assert np.all(np.abs(pred - y) < 0.5)


def test_surrogate_returns_std(sweep12):
    X = sweep12[PARAM_NAMES].values
    s = LapTimeSurrogate()
    s.fit(sweep12)
    pred, std = s.predict(X, return_std=True)
    assert len(pred) == len(std)
    assert np.all(std >= 0)
    assert np.all(np.isfinite(pred)) and np.all(np.isfinite(std))


def test_surrogate_save_load_round_trip(tmp_path, sweep12):
    X = sweep12[PARAM_NAMES].values
    s = LapTimeSurrogate()
    s.fit(sweep12)
    path = str(tmp_path / "surrogate.pkl")
    s.save(path)
    loaded = LapTimeSurrogate.load(path)
    assert np.allclose(s.predict(X), loaded.predict(X), atol=1e-9)


# --------------------------------------------------------------------------- #
# AC export
# --------------------------------------------------------------------------- #

EXPECTED_FILES = [
    "car.ini", "engine.ini", "drivetrain.ini", "suspensions.ini",
    "aero.ini", "tyres.ini", "brakes.ini", "electronics.ini",
]


@pytest.fixture(scope="module")
def ac_export(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("ac_data"))
    export_ac_car(VehicleConfig(), output_dir=out)
    return out


def test_export_creates_all_files(ac_export):
    import os
    for fname in EXPECTED_FILES:
        assert os.path.isfile(os.path.join(ac_export, fname)), f"Missing {fname}"


def _ini(ac_export, fname):
    p = configparser.ConfigParser(strict=False, inline_comment_prefixes=(";",))
    p.read(os.path.join(ac_export, fname))
    return p


def test_car_ini_mass(ac_export):
    import os
    c = VehicleConfig()
    p = _ini(ac_export, "car.ini")
    main_mass = float(p["MASS"]["MAIN"])
    fuel_mass = float(p["MASS"]["FUEL_MASS"])
    # Total must reconstruct c.mass within 1 kg
    assert abs(main_mass + fuel_mass - c.mass) < 1.0


def test_car_ini_cg_height(ac_export):
    c = VehicleConfig()
    p = _ini(ac_export, "car.ini")
    assert abs(float(p["COLLIDER"]["CG_HEIGHT"]) - c.cg_height) < 1e-4


def test_car_ini_axle_distances(ac_export):
    c = VehicleConfig()
    p = _ini(ac_export, "car.ini")
    front = float(p["COLLIDER"]["FRONT_AXLE"])
    rear  = float(p["COLLIDER"]["REAR_AXLE"])
    assert abs(front + rear - c.wheelbase) < 1e-4


def test_engine_ini_torque_curve(ac_export):
    c = VehicleConfig()
    p = _ini(ac_export, "engine.ini")
    n = len(c.torque_curve)
    for i, (rpm, tq) in enumerate(c.torque_curve):
        key = f"TORQUE_CURVE_{i}"
        assert key in p["ENGINE_DATA"] or any(key in s for s in p.sections()), \
            f"Missing {key} in engine.ini"
    # Limiter must be above the peak RPM
    limiter = int(p["ENGINE_DATA"]["LIMITER"])
    peak_rpm = max(p[0] for p in c.torque_curve)
    assert limiter > peak_rpm


def test_drivetrain_ini_gear_count(ac_export):
    c = VehicleConfig()
    p = _ini(ac_export, "drivetrain.ini")
    assert int(p["GEARS"]["COUNT"]) == len(c.gear_ratios)


def test_drivetrain_ini_final_drive(ac_export):
    c = VehicleConfig()
    p = _ini(ac_export, "drivetrain.ini")
    assert abs(float(p["FINAL_GEAR"]["RATIO"]) - c.final_drive) < 1e-4


def test_drivetrain_ini_rwd(ac_export):
    p = _ini(ac_export, "drivetrain.ini")
    assert p["TRACTION"]["TYPE"].strip().upper() == "RWD"


def test_suspensions_ini_wheel_rates(ac_export):
    c = VehicleConfig()
    p = _ini(ac_export, "suspensions.ini")
    wr_f_expected = c.spring_k_f * c.motion_ratio_f ** 2
    wr_r_expected = c.spring_k_r * c.motion_ratio_r ** 2
    assert abs(float(p["FRONT"]["KS"]) - wr_f_expected) < 1.0
    assert abs(float(p["REAR"]["KS"])  - wr_r_expected) < 1.0


def test_suspensions_ini_arb(ac_export):
    c = VehicleConfig()
    p = _ini(ac_export, "suspensions.ini")
    assert abs(float(p["FRONT"]["ANTI_ROLL"]) - c.arb_k_f) < 1.0
    assert abs(float(p["REAR"]["ANTI_ROLL"])  - c.arb_k_r) < 1.0


def test_suspensions_ini_roll_centres(ac_export):
    c = VehicleConfig()
    p = _ini(ac_export, "suspensions.ini")
    assert abs(float(p["FRONT"]["ROLL_CENTER_HEIGHT"]) - c.roll_center_height_f) < 1e-4
    assert abs(float(p["REAR"]["ROLL_CENTER_HEIGHT"])  - c.roll_center_height_r) < 1e-4


def test_aero_ini_body_drag(ac_export):
    c = VehicleConfig()
    p = _ini(ac_export, "aero.ini")
    assert abs(float(p["BODY"]["CD"]) - c.cd_a) < 1e-4


def test_aero_ini_wing_cl_sums_to_cla(ac_export):
    c = VehicleConfig()
    p = _ini(ac_export, "aero.ini")
    cl_f = float(p["WING_0"]["CL"])
    cl_r = float(p["WING_1"]["CL"])
    assert abs(cl_f + cl_r - c.cl_a) < 1e-6


def test_aero_ini_wing_balance(ac_export):
    c = VehicleConfig()
    p = _ini(ac_export, "aero.ini")
    cl_f = float(p["WING_0"]["CL"])
    cl_r = float(p["WING_1"]["CL"])
    total = cl_f + cl_r
    if total > 0:
        assert abs(cl_f / total - c.aero_balance_f) < 1e-6


def test_tyres_ini_peak_mu(ac_export):
    c = VehicleConfig()
    p = _ini(ac_export, "tyres.ini")
    assert abs(float(p["COMPOUND_0"]["DY1"]) - c.tire.c["pDy1"]) < 1e-4
    assert abs(float(p["COMPOUND_0"]["DY2"]) - c.tire.c["pDy2"]) < 1e-4


def test_tyres_ini_geometry(ac_export):
    c = VehicleConfig()
    p = _ini(ac_export, "tyres.ini")
    assert abs(float(p["COMPOUND_0"]["RADIUS"]) - c.tyre_radius) < 1e-4


def test_tyres_ini_rolling_resistance(ac_export):
    c = VehicleConfig()
    p = _ini(ac_export, "tyres.ini")
    assert abs(float(p["COMPOUND_0"]["ROLLING_RESISTANCE_0"]) - c.Cr) < 1e-6


def test_brakes_ini_bias_sums_to_one(ac_export):
    p = _ini(ac_export, "brakes.ini")
    bias_f = float(p["FRONT"]["BIAS"])
    bias_r = float(p["REAR"]["BIAS"])
    assert abs(bias_f + bias_r - 1.0) < 1e-6


def test_electronics_ini_no_aids(ac_export):
    p = _ini(ac_export, "electronics.ini")
    assert p["ABS"]["PRESENT"] == "0"
    assert p["TRACTION_CONTROL"]["PRESENT"] == "0"


def test_export_custom_config(tmp_path):
    """Exporting a modified VehicleConfig writes the modified values."""
    from dataclasses import replace
    c2 = replace(VehicleConfig(), spring_k_f=90_000.0, cl_a=1.5,
                 aero_balance_f=0.40)
    out = str(tmp_path / "custom")
    export_ac_car(c2, output_dir=out)
    import os
    p_susp = _ini(out, "suspensions.ini")
    wr_f = c2.spring_k_f * c2.motion_ratio_f ** 2
    assert abs(float(p_susp["FRONT"]["KS"]) - wr_f) < 1.0
    p_aero = _ini(out, "aero.ini")
    cl_f = float(p_aero["WING_0"]["CL"])
    cl_r = float(p_aero["WING_1"]["CL"])
    assert abs(cl_f + cl_r - c2.cl_a) < 1e-6


# --------------------------------------------------------------------------- #
# Comparison analysis / plotting (headless)
# --------------------------------------------------------------------------- #
import matplotlib
matplotlib.use("Agg")  # no display in CI; must precede any pyplot import

from analysis import (  # noqa: E402
    LapResult, time_delta, summary, compare_figure, laptime_bar_figure,
    gforce_figure, corner_speed_figure, find_corners,
)


def _lap_result(label, cfg, track):
    return LapResult.from_sim(label, LapSimulator(NCMiata(cfg), track))


def test_lapresult_cumtime_matches_solver(skidpad):
    r = _lap_result("Baseline", VehicleConfig(), skidpad)
    # The running integral must end exactly on the solver's reported lap time.
    assert abs(r.cum_time[-1] - r.lap_time) < 1e-6
    assert r.cum_time.shape == r.s.shape
    assert np.all(np.diff(r.cum_time) >= 0)  # monotonic non-decreasing


def test_time_delta_consistency(monza):
    from dataclasses import replace
    base = _lap_result("Baseline", VehicleConfig(), monza)
    opt = _lap_result("Stiff", replace(VehicleConfig(), spring_k_f=110_000.0,
                                        arb_k_f=25_000.0), monza)
    d = time_delta(base, opt)
    assert d.shape == base.s.shape
    # Final cumulative delta equals the lap-time difference.
    assert abs(d[-1] - (opt.lap_time - base.lap_time)) < 1e-6


def test_summary_metrics(skidpad):
    r = _lap_result("Baseline", VehicleConfig(), skidpad)
    rows = summary([r])
    assert rows[0]["label"] == "Baseline"
    assert abs(rows[0]["lap_time"] - r.lap_time) < 1e-9
    assert rows[0]["v_max_kph"] >= rows[0]["v_min_kph"] > 0


def test_compare_figure_builds(tmp_path, monza):
    from dataclasses import replace
    base = _lap_result("Baseline", VehicleConfig(), monza)
    opt = _lap_result("Optimized", replace(VehicleConfig(), spring_k_f=90_000.0),
                      monza)
    fig = compare_figure(base, opt)
    assert len(fig.axes) == 2
    out = tmp_path / "cmp.png"
    fig.savefig(out)
    assert out.is_file() and out.stat().st_size > 0


def test_laptime_bar_figure_builds(tmp_path, skidpad):
    base = _lap_result("Baseline", VehicleConfig(), skidpad)
    fig = laptime_bar_figure([base])
    out = tmp_path / "bar.png"
    fig.savefig(out)
    assert out.is_file() and out.stat().st_size > 0


def test_gforce_properties(skidpad):
    # On a constant-radius skidpad the lateral g is v**2*kappa/g everywhere and
    # there is no sustained longitudinal accel once the car is up to speed.
    r = _lap_result("Baseline", VehicleConfig(), skidpad)
    assert r.lat_g.shape == r.s.shape
    assert r.long_g.shape == r.s.shape
    assert np.all(r.lat_g >= 0)
    assert r.lat_g.max() > 0  # the car is cornering


def test_gforce_figure_builds(tmp_path, monza):
    from dataclasses import replace
    base = _lap_result("Baseline", VehicleConfig(), monza)
    opt = _lap_result("Optimized", replace(VehicleConfig(), spring_k_f=90_000.0),
                      monza)
    fig = gforce_figure(base, opt)
    assert len(fig.axes) == 2
    out = tmp_path / "g.png"
    fig.savefig(out)
    assert out.is_file() and out.stat().st_size > 0


def test_find_corners_on_monza(monza):
    base = _lap_result("Baseline", VehicleConfig(), monza)
    corners = find_corners(base.curvature)
    assert len(corners) > 0
    # slices are ordered, non-empty, and within bounds
    for (i, j) in corners:
        assert 0 <= i < j <= len(base.curvature)


def test_corner_speed_figure_builds(tmp_path, monza):
    from dataclasses import replace
    base = _lap_result("Baseline", VehicleConfig(), monza)
    opt = _lap_result("Optimized", replace(VehicleConfig(), spring_k_f=90_000.0),
                      monza)
    fig = corner_speed_figure(base, opt)
    out = tmp_path / "corners.png"
    fig.savefig(out)
    assert out.is_file() and out.stat().st_size > 0


# --------------------------------------------------------------------------- #
# Extended channels, track map, custom chart, KPI sweep (headless)
# --------------------------------------------------------------------------- #
from analysis import (  # noqa: E402
    track_xy, build_lap_channels, numeric_channels,
    track_map_figure, channel_chart_figure,
    run_kpi_sweep, kpi_chart_figure, KPI_PARAMS, KPI_METRICS,
)


def _channels(cfg, track):
    car = NCMiata(cfg)
    return build_lap_channels(_lap_result("R", cfg, track), car)


def test_track_xy_integrates_curvature():
    # A constant curvature traces a circular arc: heading is linear in s and
    # the path length matches the swept angle * radius.
    s = np.linspace(0.0, 100.0, 201)
    kappa = np.full_like(s, 1.0 / 25.0)        # R = 25 m
    x, y, theta = track_xy(s, kappa)
    assert x.shape == y.shape == theta.shape == s.shape
    assert np.allclose(theta, kappa * s, atol=1e-3)   # theta = integral kappa ds
    # points lie ~on the circle of radius 25 centred at (0, 25)
    r = np.hypot(x, y - 25.0)
    assert np.allclose(r, 25.0, atol=0.5)


def test_build_lap_channels_new_columns(monza):
    df = _channels(VehicleConfig(), monza)
    for col in ("max_corner_speed_kph", "engaged_gear_ratio", "throttle_pct",
                "brake_pct", "long_grip_used_pct", "combined_grip_used_pct",
                "yaw_deg", "pos_x_m", "pos_y_m", "sector_index"):
        assert col in df.columns
    # throttle / brake are 0/100 discrete states
    assert set(np.unique(df["throttle_pct"])) <= {0.0, 100.0}
    assert set(np.unique(df["brake_pct"])) <= {0.0, 100.0}
    # grip-usage percentages are bounded to [0, 100]
    for col in ("lat_grip_used_pct", "long_grip_used_pct", "combined_grip_used_pct"):
        assert df[col].min() >= 0.0 and df[col].max() <= 100.0 + 1e-6
    # sectors split the lap into thirds (1..3)
    assert set(np.unique(df["sector_index"])) <= {1, 2, 3}
    # max corner speed never exceeds the achieved top speed
    assert df["max_corner_speed_kph"].max() <= df["speed_kph"].max() + 1e-6
    # the racing line spans a real 2-D extent (not degenerate)
    assert np.ptp(df["pos_x_m"]) > 1.0 and np.ptp(df["pos_y_m"]) > 1.0


def test_numeric_channels_excludes_driver_state(monza):
    df = _channels(VehicleConfig(), monza)
    cols = numeric_channels(df)
    assert "driver_state" not in cols
    assert "speed_kph" in cols and "pos_x_m" in cols


def test_track_map_figure_builds(tmp_path, monza):
    df = _channels(VehicleConfig(), monza)
    fig = track_map_figure(df, "speed_kph", line_width=3.0, label="test")
    # one data axis + one colour-bar axis
    assert len(fig.axes) >= 2
    out = tmp_path / "trackmap.png"
    fig.savefig(out)
    assert out.is_file() and out.stat().st_size > 0


def test_track_map_figure_bad_channel_falls_back(monza):
    df = _channels(VehicleConfig(), monza)
    # a non-existent channel must not raise; it falls back to speed
    fig = track_map_figure(df, "does_not_exist")
    assert len(fig.axes) >= 2


def test_channel_chart_figure_line_and_scatter(tmp_path, skidpad):
    df = _channels(VehicleConfig(), skidpad)
    f1 = channel_chart_figure(df, "distance_m", "speed_kph", "line")
    f2 = channel_chart_figure(df, "lat_accel_g", "long_accel_g", "scatter")
    for k, fig in enumerate((f1, f2)):
        out = tmp_path / f"chart{k}.png"
        fig.savefig(out)
        assert out.is_file() and out.stat().st_size > 0


def test_run_kpi_sweep_mass_monotonic(skidpad):
    cfg = VehicleConfig()
    values = np.linspace(1000.0, 1300.0, 4)
    sweep = run_kpi_sweep(cfg, skidpad, "mass", values)
    assert list(sweep["mass"]) == [pytest.approx(v) for v in values]
    for m in KPI_METRICS:
        assert m in sweep.columns
    # heavier car => slower lap (monotonic non-decreasing lap time)
    assert np.all(np.diff(sweep["lap_time_s"].to_numpy()) >= -1e-6)


def test_run_kpi_sweep_unknown_param_raises(skidpad):
    with pytest.raises(ValueError):
        run_kpi_sweep(VehicleConfig(), skidpad, "not_a_param", [1.0, 2.0])


def test_kpi_chart_figure_builds(tmp_path, skidpad):
    sweep = run_kpi_sweep(VehicleConfig(), skidpad, "mass",
                          np.linspace(1000.0, 1300.0, 4))
    fig = kpi_chart_figure(sweep, "mass", "lap_time_s", label="test")
    out = tmp_path / "kpi.png"
    fig.savefig(out)
    assert out.is_file() and out.stat().st_size > 0
