"""Assetto Corsa physics export from VehicleConfig.

Generates a complete, editable ``data/`` folder of AC ``.ini`` files for the
NC Miata.  Every value is derived from ``VehicleConfig`` (the single source of
truth), so whatever the optimizer found best is what goes into the game.

Usage
-----
    from config import VehicleConfig
    from export_to_ac import export_ac_car

    export_ac_car(VehicleConfig(), output_dir="nc_miata_qss/data")

The output directory is created if it does not exist.  Drop the entire
``nc_miata_qss/`` folder into ``assettocorsa/content/cars/`` alongside the
``.kn5`` model and ``ui/`` folder from the donor mod.

File map
--------
    car.ini          – mass, inertia, fuel, basic dimensions
    engine.ini       – torque / power curve, rev limiter, inertia
    drivetrain.ini   – gear ratios, final drive, differential, efficiency
    suspensions.ini  – spring / ARB rates, geometry hardpoints
    aero.ini         – Cl·A, Cd·A, aero balance as front/rear wings
    tyres.ini        – Pacejka-derived compound, geometry
    brakes.ini       – max torque, bias
    electronics.ini  – ABS / TC defaults (disabled for a stock road car)
"""
from __future__ import annotations

import math
import os
from typing import Optional

from config import VehicleConfig


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _header(link: str = "") -> str:
    return f"[HEADER]\nVERSION=1\nLINK={link}\n\n"


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


# --------------------------------------------------------------------------- #
# Individual file generators
# --------------------------------------------------------------------------- #

def _car_ini(c: VehicleConfig) -> str:
    """car.ini – mass, inertia box, fuel, CG height."""
    # AC inertia box: a rectangular box whose half-extents (x=lateral,
    # y=vertical, z=longitudinal) give Ixx/Iyy/Izz via the parallel-axis
    # theorem.  Approximate NC Miata dimensions.
    ibox_x = 0.90   # half-width  (m)
    ibox_y = 0.30   # half-height (m)
    ibox_z = 1.60   # half-length (m)
    fuel_kg_per_l = 0.742
    fuel_tank_l   = 48.0   # stock NC tank (litres)
    fuel_mass     = fuel_tank_l * fuel_kg_per_l

    # Driver + fluids included in c.mass; AC MASS key is the DRY body mass.
    # We use c.mass directly (it already includes driver by convention).
    return (
        _header()
        + f"[BASIC]\n"
        + f"FUEL={fuel_tank_l:.0f}\n\n"
        + f"[MASS]\n"
        + f"MAIN={c.mass - fuel_mass:.1f}  ; dry mass (kg) — fuel modelled separately\n"
        + f"FUEL_MASS={fuel_mass:.1f}\n\n"
        + f"[INERTIA]\n"
        + f"; Half-extents of the inertia box (x, y, z in metres).\n"
        + f"INERTIA_BOX={ibox_x:.3f} {ibox_y:.3f} {ibox_z:.3f}\n\n"
        + f"[COLLIDER]\n"
        + f"CG_HEIGHT={c.cg_height:.4f}\n"
        + f"FRONT_AXLE={c.wheelbase * c.weight_dist_f:.4f}   ; dist CG→front axle\n"
        + f"REAR_AXLE={c.wheelbase * (1.0 - c.weight_dist_f):.4f}   ; dist CG→rear axle\n"
    )


def _engine_ini(c: VehicleConfig) -> str:
    """engine.ini – torque curve, rev limiter, engine inertia."""
    rpm_pts = [p[0] for p in c.torque_curve]
    tq_pts  = [p[1] for p in c.torque_curve]
    rpm_idle = 800
    rpm_max  = max(rpm_pts)
    rpm_limiter = int(rpm_max * 1.02)   # ~2% above peak

    # Power curve in kW for reference (AC wants Nm in the torque table).
    # Coast torque: ~20% of peak torque at each rpm (engine braking model).
    coast_frac = 0.20

    curve_lines = "\n".join(
        f"TORQUE_CURVE_{i}={rpm},{tq}" for i, (rpm, tq) in enumerate(c.torque_curve)
    )
    coast_lines = "\n".join(
        f"COAST_CURVE_{i}={rpm},{int(tq * coast_frac)}"
        for i, (rpm, tq) in enumerate(c.torque_curve)
    )

    return (
        _header()
        + f"[ENGINE_DATA]\n"
        + f"ALTITUDE_SENSITIVITY=0.1\n"
        + f"INERTIA=0.120   ; flywheel + internals (kg·m²) — NC 2.0L estimate\n"
        + f"LIMITER={rpm_limiter}\n"
        + f"LIMITER_HZ=50\n"
        + f"MINIMUM={rpm_idle}\n"
        + curve_lines + "\n"
        + coast_lines + "\n\n"
        + f"[TURBO_0]\n"
        + f"LAG_DN=0.0\nLAG_UP=0.0\nMAX_BOOST=0.0\nWASTEGATE=0.0\n"
    )


def _drivetrain_ini(c: VehicleConfig) -> str:
    """drivetrain.ini – gears, final drive, differential, efficiencies."""
    n_gears = len(c.gear_ratios)
    gear_lines = "\n".join(
        f"GEAR_{i+1}={ratio:.4f}" for i, ratio in enumerate(c.gear_ratios)
    )
    # NC Miata is RWD with a Torsen-style LSD (typical street tune ≈ 40% lock).
    return (
        _header()
        + f"[TRACTION]\n"
        + f"TYPE=RWD\n\n"
        + f"[GEARS]\n"
        + f"COUNT={n_gears}\n"
        + f"GEAR_R=-{c.gear_ratios[0]:.4f}   ; reverse — mirror of 1st\n"
        + gear_lines + "\n\n"
        + f"[DIFFERENTIAL]\n"
        + f"POWER=0.40      ; coast-side lock factor\n"
        + f"COAST=0.40\n"
        + f"PRELOAD=25.0    ; Nm\n\n"
        + f"[FINAL_GEAR]\n"
        + f"RATIO={c.final_drive:.4f}\n\n"
        + f"[GEARBOX]\n"
        + f"VALID_SHIFT_RPM_WINDOW=600\n"
        + f"CONTROLS_WINDOW_GAIN=0.40\n"
        + f"INERTIA=0.020\n"
        + f"EFFICIENCY={c.gear_eff:.4f}\n"
        + f"CHANGE_UP_TIME={int(c.shift_time * 1000)}\n"
        + f"CHANGE_DN_TIME={int(c.shift_time * 1000)}\n\n"
        + f"[AXLE]\n"
        + f"INERTIA=0.030\n"
        + f"EFFICIENCY={c.final_eff:.4f}\n"
    )


def _suspensions_ini(c: VehicleConfig) -> str:
    """suspensions.ini – double-wishbone front, multilink rear.

    Hardpoint coordinates are estimated from the NC Miata's known geometry
    (wheelbase, track, CG height, roll-centre heights) rather than from a CAD
    scan.  They give AC the correct effective spring/ARB rates and roll centres
    without needing the full suspension kinematics.

    AC coordinate system: x = lateral (right+), y = vertical (up+), z = fwd.
    All distances from the CG reference point.
    """
    # Derived geometry
    half_f = c.track_f / 2.0
    half_r = c.track_r / 2.0
    a = c.wheelbase * c.weight_dist_f          # CG → front axle
    b = c.wheelbase * (1.0 - c.weight_dist_f)  # CG → rear axle

    # Wheel-rate from spring-rate × MR² (N/m at the wheel)
    wr_f = c.spring_k_f * c.motion_ratio_f ** 2
    wr_r = c.spring_k_r * c.motion_ratio_r ** 2

    # AC wants spring rate at the WHEEL (N/m).
    # ARB is stored separately as an equivalent wheel-rate (N/m).
    hub_y = 0.285  # wheel-centre height (m)

    def _wishbone_block(axle: str, half_t: float, ax_z: float,
                        wr: float, arb: float, rc_h: float,
                        kd_bump: float, kd_rebound: float) -> str:
        """Generate a double-wishbone / multilink block for one axle.

        AC's KD key is a single symmetric damping coefficient (N·s/m).
        We export the average of bump and rebound as KD so neither phase is
        significantly over- or under-represented; both raw values are preserved
        in comments for reference.
        """
        kd_avg = (kd_bump + kd_rebound) / 2.0

        # Upper & lower wishbone inner/outer attachment points (estimated).
        # Inner = chassis-side, outer = upright-side.
        uca_in  = f"0.250 {hub_y + 0.130:.3f} {ax_z:.3f}"
        uca_out = f"{half_t:.3f} {hub_y + 0.080:.3f} {ax_z:.3f}"
        lca_in  = f"0.250 {hub_y - 0.095:.3f} {ax_z:.3f}"
        lca_out = f"{half_t:.3f} {hub_y - 0.110:.3f} {ax_z:.3f}"

        return (
            f"[{axle}]\n"
            + f"TYPE=DW\n"
            + f"BASEY={hub_y:.3f}\n"
            + f"TRACK={half_t * 2:.4f}\n"
            + f"HUB_MASS=20.0\n"
            + f"WHEEL_MASS=10.0\n"
            + f"ROD_LENGTH=0.001\n"
            + f"UCA_INNER={uca_in}\n"
            + f"UCA_OUTER={uca_out}\n"
            + f"LCA_INNER={lca_in}\n"
            + f"LCA_OUTER={lca_out}\n"
            + f"KS={wr:.1f}    ; wheel rate N/m\n"
            + f"KD={kd_avg:.1f}   ; avg damping N·s/m  "
              f"(bump={kd_bump:.0f}  rebound={kd_rebound:.0f})\n"
            + f"PROGRESSIVE_SPRING_K=0\n"
            + f"BUMPSTOP_UP=0.05\n"
            + f"BUMPSTOP_DN=0.05\n"
            + f"BUMPSTOP_K=100000\n"
            + f"ANTI_ROLL={arb:.1f}\n"
            + f"STATIC_CAMBER=-1.5\n"
            + f"TOE=0.0\n"
            + f"ROLL_CENTER_HEIGHT={rc_h:.4f}\n"
        )

    front = _wishbone_block("FRONT", half_f, -a, wr_f, c.arb_k_f,
                            c.roll_center_height_f,
                            c.damper_bump_f, c.damper_rebound_f)
    rear  = _wishbone_block("REAR",  half_r,  b, wr_r, c.arb_k_r,
                            c.roll_center_height_r,
                            c.damper_bump_r, c.damper_rebound_r)

    return _header() + front + "\n" + rear


def _aero_ini(c: VehicleConfig) -> str:
    """aero.ini – body drag/lift plus front and rear wings.

    AC models aero as a body + discrete wing elements. We map our lumped
    Cl·A / Cd·A onto the body + two wings (front and rear). The wings carry
    the downforce split (aero_balance_f / 1-aero_balance_f).
    """
    # Body carries all drag and whatever the stock lift/drag the NC has.
    # Wings carry the added downforce (cl_a) split front/rear.
    body_cd   = c.cd_a          # our Cd·A is the body Cd·A
    body_cl   = 0.0             # no net body lift for a stock NC
    wing_cla_f = c.cl_a * c.aero_balance_f
    wing_cla_r = c.cl_a * (1.0 - c.aero_balance_f)

    def _wing(idx: int, name: str, cla: float, offset_z: float) -> str:
        # AC wing CL is referenced to a wing area; we store Cl·A directly
        # by setting WING_AREA=1.0 so CL = Cl·A numerically.
        return (
            f"[WING_{idx}]\n"
            f"NAME={name}\n"
            f"OFFSET=0 0.300 {offset_z:.3f}\n"
            f"ROTATION=0\n"
            f"CHORD=1\n"
            f"SPAN=1\n"
            f"CL={cla:.4f}\n"
            f"CD=0.0\n"
            f"WING_AREA=1.0\n"
        )

    return (
        _header()
        + f"[BODY]\n"
        + f"CD={body_cd:.4f}    ; Cd*frontal_area (m²)\n"
        + f"CL={body_cl:.4f}    ; Cl*ref_area — negative = downforce\n\n"
        + _wing(0, "FRONT_DOWNFORCE", wing_cla_f, -1.20) + "\n"
        + _wing(1, "REAR_DOWNFORCE",  wing_cla_r,  1.20)
    )


def _tyres_ini(c: VehicleConfig) -> str:
    """tyres.ini – Pacejka-mapped compound for AC's internal tyre model.

    AC uses a simplified Pacejka-like parameterisation (DY1/DY2 = mu peak,
    CY = shape factor, BY1/BY2/EY = stiffness/curvature, DX1/DX2 for
    longitudinal). We map our lateral coefficients directly; longitudinal
    is set to typical track-tyre values (slightly higher peak than lateral).
    """
    t = c.tire
    rim_diameter_in = 16.0      # NC stock rim diameter
    rim_radius = (rim_diameter_in * 0.0254) / 2.0   # metres
    tyre_width = 0.205          # 205/50R16

    # Longitudinal (slightly grippier than lateral, typical for asymmetric tyres)
    dx1 = t.c["pDy1"] * 1.05
    dx2 = t.c["pDy2"] * 0.90

    compound_block = (
        f"[HEADER]\nVERSION=1\nLINK=\n\n"
        f"[COMPOUND_0]\n"
        f"NAME={t.name}\n"
        f"SHORT_NAME=S\n"
        f"WEAR_CURVE=0=0|100=0    ; no wear model for track-day use\n"
        f"WIDTH={tyre_width:.4f}\n"
        f"RIM_RADIUS={rim_radius:.5f}\n"
        f"RADIUS={c.tyre_radius:.4f}\n"
        f"RADIUS_CBF=0.0006\n"
        f"RIM_RADIUS_CBF=0.0006\n"
        f"DAMP=400\n"
        f"RATE={int(c.spring_k_f * c.motion_ratio_f ** 2 * 2.5)}\n"
        f"FLEX=0.035\n"
        f"ANGULAR_INERTIA=0.8\n"
        f"FZ0={t.Fz0:.0f}\n"
        f"; --- Lateral (Pacejka MF — mapped from VehicleConfig.tire) ---\n"
        f"DY1={t.c['pDy1']:.4f}\n"
        f"DY2={t.c['pDy2']:.4f}\n"
        f"DY3=0.0\n"
        f"CY={t.c['pCy1']:.4f}\n"
        f"BY1={t.c['pKy1']:.4f}\n"
        f"BY2={t.c['pKy2']:.4f}\n"
        f"BY3=0.0\n"
        f"EY1={t.c['pEy1']:.4f}\n"
        f"EY2={t.c['pEy2']:.4f}\n"
        f"; --- Longitudinal ---\n"
        f"DX1={dx1:.4f}\n"
        f"DX2={dx2:.4f}\n"
        f"CX=1.65\n"
        f"BX1=22.0\n"
        f"BX2=0.0\n"
        f"EX1=-0.50\n"
        f"EX2=0.0\n"
        f"; --- Rolling resistance ---\n"
        f"ROLLING_RESISTANCE_0={c.Cr:.5f}\n"
        f"ROLLING_RESISTANCE_1=0.0\n"
    )
    return compound_block


def _brakes_ini(c: VehicleConfig) -> str:
    """brakes.ini – max brake torque and bias.

    NC Miata stock brakes: ventilated 280 mm front discs, solid 280 mm rear.
    Max torque estimated from known deceleration capability (~1.0 g at 100 km/h).
    Brake balance typical for front-heavy FR: 57% front.
    """
    mu_pad  = 0.42      # aggressive street/track pad
    r_disc_f = 0.140    # effective disc radius, front (m)
    r_disc_r = 0.140    # effective disc radius, rear  (m)
    n_pads   = 2        # pads per corner
    clamp_f  = 25000    # clamp force per caliper (N) → adjustable
    clamp_r  = 18000

    max_tq_f = mu_pad * r_disc_f * n_pads * clamp_f   # per corner (N·m)
    max_tq_r = mu_pad * r_disc_r * n_pads * clamp_r

    bias_f  = max_tq_f / (max_tq_f + max_tq_r)   # fraction on front

    return (
        _header()
        + f"[FRONT]\n"
        + f"MAX_TORQUE={max_tq_f:.0f}    ; per corner (N·m)\n"
        + f"DISC_RADIUS={r_disc_f:.4f}\n"
        + f"PAD_FRICTION={mu_pad:.3f}\n"
        + f"BIAS={bias_f:.4f}   ; fraction applied to front\n"
        + f"HANDBRAKE=0\n\n"
        + f"[REAR]\n"
        + f"MAX_TORQUE={max_tq_r:.0f}\n"
        + f"DISC_RADIUS={r_disc_r:.4f}\n"
        + f"PAD_FRICTION={mu_pad:.3f}\n"
        + f"BIAS={1.0 - bias_f:.4f}\n"
        + f"HANDBRAKE=1\n"
    )


def _electronics_ini(c: VehicleConfig) -> str:
    """electronics.ini – ABS / TC levels (stock NC = none)."""
    return (
        _header()
        + f"[ABS]\n"
        + f"PRESENT=0\n"
        + f"LEVELS=0\n\n"
        + f"[TRACTION_CONTROL]\n"
        + f"PRESENT=0\n"
        + f"LEVELS=0\n\n"
        + f"[STABILITY_CONTROL]\n"
        + f"PRESENT=0\n"
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def export_ac_car(
    config: Optional[VehicleConfig] = None,
    output_dir: str = "nc_miata_qss/data",
) -> None:
    """Write all AC physics ``.ini`` files to *output_dir*.

    Parameters
    ----------
    config:
        Vehicle parameters.  Defaults to ``VehicleConfig()`` (stock NC Miata).
    output_dir:
        Destination folder.  Created (including parents) if it does not exist.
    """
    c = config if config is not None else VehicleConfig()
    os.makedirs(output_dir, exist_ok=True)

    files = {
        "car.ini":          _car_ini(c),
        "engine.ini":       _engine_ini(c),
        "drivetrain.ini":   _drivetrain_ini(c),
        "suspensions.ini":  _suspensions_ini(c),
        "aero.ini":         _aero_ini(c),
        "tyres.ini":        _tyres_ini(c),
        "brakes.ini":       _brakes_ini(c),
        "electronics.ini":  _electronics_ini(c),
    }

    for filename, content in files.items():
        _write(os.path.join(output_dir, filename), content)
        print(f"  wrote  {os.path.join(output_dir, filename)}")

    print(f"\n✅  AC data folder ready: {output_dir!r}")
    print(f"    {len(files)} physics files exported from VehicleConfig.")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from config import VehicleConfig
    export_ac_car(VehicleConfig(), output_dir="nc_miata_qss/data")
