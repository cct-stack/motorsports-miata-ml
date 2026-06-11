"""
export_openlap.py
=================
Exports the NC Miata vehicle and a simple skidpad track to OpenLAP-compatible
xlsx files.  Drop the generated files into the OpenLAP folder and run:
  1. OpenVEHICLE.m  (reads  NC_Miata_OpenVEHICLE.xlsx)
  2. OpenTRACK.m    (reads  Skidpad_30m_OpenTRACK.xlsx)
  3. OpenLAP.m      (uses both outputs)

Parameter mapping notes
-----------------------
mu_y / mu_y_M / sens_y:
  OpenLAP's linear tire model:  mu_eff = mu_y + sens_y*(mu_y_M*g - Fz_per_tire)
  Our Pacejka:  mu = pDy1 + pDy2*(Fz - Fz0)/Fz0
  => mu_y   = pDy1  (peak mu at nominal load Fz0)
  => mu_y_M = Fz0/g  (nominal "mass" so that Ny = Fz0, mu=mu_y exactly there)
  => sens_y = -pDy2/Fz0  (slope match;  pDy2 < 0 so sens_y > 0)

Identical logic applies to mu_x (we use 0.9*mu_y as a typical longitudinal scaling).

Cornering stiffness (CF/CR, N/deg per axle):
  K = pKy1*Fz0*sin(2*atan(Fz0/(pKy2*Fz0)))  [N/rad per tire at Fz0]
  CF = CR = 2 * K / (pi/180)   (two tires, convert rad -> deg)
"""

import math
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from vehicle_model import NCMiata
from tire_model import PacejkaTire

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hdr(ws, row, label):
    """Write a grey section-header row."""
    ws.cell(row=row, column=1, value=label).font = Font(bold=True)
    ws.cell(row=row, column=1).fill = PatternFill("solid", fgColor="CCCCCC")

def _row(ws, row, label, value, unit=""):
    ws.cell(row=row, column=1, value=label)
    ws.cell(row=row, column=2, value=str(value))
    if unit:
        ws.cell(row=row, column=3, value=unit)


# Engine / drivetrain constants are sourced from NCMiata in vehicle_model.py.
# TYRE_RADIUS_MM is derived from vehicle.tyre_radius at export time.

# Brakes – typical Miata (Wilwood or OEM upgraded)
BRAKE_DISC_D_MM  = 270.0   # front disc diameter
BRAKE_PAD_H_MM   =  35.0   # pad height
BRAKE_PAD_MU     =   0.42  # pad friction coefficient
BRAKE_NOP        =   4     # pistons per caliper (front)
BRAKE_PIST_D_MM  =  38.0   # piston diameter
BRAKE_MAST_D_MM  =  22.0   # master cylinder bore
BRAKE_PED_RATIO  =   5.0   # pedal ratio


# ─────────────────────────────────────────────────────────────────────────────
# Compute derived tire parameters
# ─────────────────────────────────────────────────────────────────────────────

def _tire_params(tire: PacejkaTire):
    """Convert Pacejka coefficients to OpenLAP's linear tire scalars."""
    c = tire.c
    Fz0 = tire.Fz0

    mu_y    = c["pDy1"]
    mu_y_M  = Fz0 / 9.81                    # [kg]  Ny = mu_y_M*g = Fz0
    sens_y  = -c["pDy2"] / Fz0              # [1/N]

    # longitudinal: ~90 % of lateral for a 200TW track tyre
    mu_x    = mu_y * 0.90
    mu_x_M  = mu_y_M
    sens_x  = sens_y * 0.90

    # cornering stiffness per axle [N/deg]  (2 tyres at Fz0 each)
    K_rad   = c["pKy1"] * Fz0 * math.sin(
                2.0 * math.atan(Fz0 / (c["pKy2"] * Fz0)))
    CF = 2.0 * K_rad * (180.0 / math.pi)    # front axle [N/deg]
    CR = CF                                  # symmetric for a mid-engined/balanced car

    return mu_y, mu_y_M, sens_y, mu_x, mu_x_M, sens_x, CF, CR


# ─────────────────────────────────────────────────────────────────────────────
# Write OpenVEHICLE xlsx
# ─────────────────────────────────────────────────────────────────────────────

def export_vehicle(vehicle: NCMiata, path: str = "NC_Miata_OpenVEHICLE.xlsx"):
    t = vehicle.tire
    mu_y, mu_y_M, sens_y, mu_x, mu_x_M, sens_x, CF, CR = _tire_params(t)

    wb = openpyxl.Workbook()

    # ── Info sheet ──────────────────────────────────────────────────────────
    ws = wb.active
    ws.title = "Info"
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["C"].width = 12

    ws.cell(row=1, column=1, value="Variable").font = Font(bold=True)
    ws.cell(row=1, column=2, value="Value").font = Font(bold=True)
    ws.cell(row=1, column=3, value="Unit").font = Font(bold=True)

    rows = [
        # (label,                            value,                     unit)
        ("Name",                             "NC Miata RE-71RS",        ""),
        ("Type",                             "Road Car",                ""),
        ("M",                                vehicle.mass,              "kg"),
        ("df",                               vehicle.weight_dist_f*100, "%"),
        ("L",                                vehicle.wheelbase*1000,    "mm"),
        ("rack",                             15.0,                      "-"),
        ("Cl",                               abs(vehicle.cl_a / 1.8),   "-"),
        ("Cd",                               vehicle.cd_a / 1.8,        "-"),
        ("factor_Cl",                        1.0,                       "-"),
        ("factor_Cd",                        1.0,                       "-"),
        ("da",                               45.0,                      "%"),
        ("A",                                1.80,                      "m2"),
        ("rho",                              vehicle.rho,               "kg/m3"),
        ("br_disc_d",                        BRAKE_DISC_D_MM,           "mm"),
        ("br_pad_h",                         BRAKE_PAD_H_MM,            "mm"),
        ("br_pad_mu",                        BRAKE_PAD_MU,              "-"),
        ("br_nop",                           BRAKE_NOP,                 "-"),
        ("br_pist_d",                        BRAKE_PIST_D_MM,           "mm"),
        ("br_mast_d",                        BRAKE_MAST_D_MM,           "mm"),
        ("br_ped_r",                         BRAKE_PED_RATIO,           "-"),
        ("factor_grip",                      1.0,                       "-"),
        ("tyre_radius",                      round(vehicle.tyre_radius * 1000, 1), "mm"),
        ("Cr",                               vehicle.Cr,                "-"),
        ("mu_x",                             round(mu_x, 4),            "-"),
        ("mu_x_M",                           round(mu_x_M, 2),          "1/kg"),
        ("sens_x",                           round(sens_x, 6),          "-"),
        ("mu_y",                             round(mu_y, 4),            "-"),
        ("mu_y_M",                           round(mu_y_M, 2),          "1/kg"),
        ("sens_y",                           round(sens_y, 6),          "-"),
        ("CF",                               round(CF, 1),              "N/deg"),
        ("CR",                               round(CR, 1),              "N/deg"),
        ("factor_power",                     1.0,                       "-"),
        ("n_thermal",                        1.0,                       "-"),
        ("fuel_LHV",                         43.4e6,                    "J/kg"),
        ("drive",                            "RWD",                     "-"),
        ("shift_time",                       vehicle.shift_time,        "s"),
        ("n_primary",                        vehicle.primary_eff,       "-"),
        ("n_final",                          vehicle.final_eff,         "-"),
        ("n_gearbox",                        vehicle.gear_eff,          "-"),
        ("ratio_primary",                    vehicle.primary_ratio,     "-"),
        ("ratio_final",                      vehicle.final_drive,       "-"),
    ]
    for i, (lbl, val, unit) in enumerate(rows, start=2):
        _row(ws, i, lbl, val, unit)

    # Gear ratios – one per row after the fixed rows
    r_start = 2 + len(rows)
    for g, ratio in enumerate(vehicle.gear_ratios, start=1):
        _row(ws, r_start, f"ratio_gearbox_gear{g}", ratio, "-")
        r_start += 1

    # ── Torque Curve sheet ───────────────────────────────────────────────────
    ws2 = wb.create_sheet("Torque Curve")
    ws2.cell(row=1, column=1, value="Engine_Speed_rpm").font = Font(bold=True)
    ws2.cell(row=1, column=2, value="Torque_Nm").font = Font(bold=True)
    for i, (rpm, nm) in enumerate(vehicle.torque_curve, start=2):
        ws2.cell(row=i, column=1, value=rpm)
        ws2.cell(row=i, column=2, value=nm)

    wb.save(path)
    print(f"[OpenVEHICLE] Saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Write OpenTRACK xlsx  (skidpad – single closed circle, 30 m radius)
# ─────────────────────────────────────────────────────────────────────────────

def export_track(radius_m: float = 30.0, path: str = "Skidpad_30m_OpenTRACK.xlsx"):
    """
    Models the skidpad as a single closed circle.  OpenTRACK 'shape data' mode:
      Shape sheet: Type | Length [m] | Radius [m]
      One 'Left' arc whose arc length = 2*pi*r  (full circle).
    """
    arc_length = 2.0 * math.pi * radius_m

    wb = openpyxl.Workbook()

    # ── Info sheet ──────────────────────────────────────────────────────────
    ws = wb.active
    ws.title = "Info"
    info_data = [
        ("Name",        "Skidpad 30m"),
        ("Country",     "USA"),
        ("City",        "Track Day"),
        ("Type",        "Skidpad"),
        ("Config",      "Closed"),
        ("Direction",   "Clockwise"),
        ("Mirror",      "Off"),
    ]
    for i, (lbl, val) in enumerate(info_data, start=1):
        ws.cell(row=i, column=1, value=lbl)
        ws.cell(row=i, column=2, value=val)

    # ── Shape sheet ──────────────────────────────────────────────────────────
    ws2 = wb.create_sheet("Shape")
    ws2.cell(row=1, column=1, value="Type").font = Font(bold=True)
    ws2.cell(row=1, column=2, value="Section Length [m]").font = Font(bold=True)
    ws2.cell(row=1, column=3, value="Corner Radius [m]").font = Font(bold=True)
    # One right-hand circle
    ws2.cell(row=2, column=1, value="Right")
    ws2.cell(row=2, column=2, value=round(arc_length, 3))
    ws2.cell(row=2, column=3, value=radius_m)

    # ── Elevation sheet (flat) ───────────────────────────────────────────────
    ws3 = wb.create_sheet("Elevation")
    ws3.cell(row=1, column=1, value="Point [m]").font = Font(bold=True)
    ws3.cell(row=1, column=2, value="Elevation [m]").font = Font(bold=True)
    for pt in [0.0, arc_length]:
        r = ws3.max_row + 1
        ws3.cell(row=r, column=1, value=round(pt, 3))
        ws3.cell(row=r, column=2, value=0.0)

    # ── Banking sheet (zero banking) ─────────────────────────────────────────
    ws4 = wb.create_sheet("Banking")
    ws4.cell(row=1, column=1, value="Point [m]").font = Font(bold=True)
    ws4.cell(row=1, column=2, value="Banking [deg]").font = Font(bold=True)
    for pt in [0.0, arc_length]:
        r = ws4.max_row + 1
        ws4.cell(row=r, column=1, value=round(pt, 3))
        ws4.cell(row=r, column=2, value=0.0)

    # ── Grip Factors sheet ───────────────────────────────────────────────────
    ws5 = wb.create_sheet("Grip Factors")
    ws5.cell(row=1, column=1, value="Point [m]").font = Font(bold=True)
    ws5.cell(row=1, column=2, value="Grip Factor [-]").font = Font(bold=True)
    for pt in [0.0, arc_length]:
        r = ws5.max_row + 1
        ws5.cell(row=r, column=1, value=round(pt, 3))
        ws5.cell(row=r, column=2, value=1.0)

    # ── Sectors sheet ────────────────────────────────────────────────────────
    ws6 = wb.create_sheet("Sectors")
    ws6.cell(row=1, column=1, value="Point [m]").font = Font(bold=True)
    ws6.cell(row=1, column=2, value="Sector [-]").font = Font(bold=True)
    ws6.cell(row=2, column=1, value=0.0)
    ws6.cell(row=2, column=2, value=1)

    wb.save(path)
    print(f"[OpenTRACK] Saved  → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    car = NCMiata()
    veh_path   = export_vehicle(car)
    track_path = export_track()

    print()
    print("=" * 60)
    print("OpenLAP export complete.")
    print("=" * 60)
    print()
    print("Next steps in MATLAB / Octave:")
    print("  1. Copy both xlsx files into your OpenLAP folder.")
    print("  2. In OpenVEHICLE.m  set:  filename = 'NC_Miata_OpenVEHICLE.xlsx'")
    print("  3. Run OpenVEHICLE.m  → produces  OpenVEHICLE_NC Miata RE-71RS_Road Car.mat")
    print("  4. In OpenTRACK.m    set:  filename = 'Skidpad_30m_OpenTRACK.xlsx'")
    print("                             mode     = 'shape data'")
    print("  5. Run OpenTRACK.m   → produces  OpenTRACK_Skidpad 30m_Closed_Clockwise.mat")
    print("  6. In OpenLAP.m      set vehiclefile and trackfile to those .mat paths.")
    print("  7. Run OpenLAP.m and compare the reported lap time to our Python sim.")
    print()
    print("Our current Python sim result: ~9.65 s  (skidpad, 30 m radius, RE-71RS)")
    print("OpenLAP should return a lap time in the same ballpark.")
    print("Key differences to expect:")
    print("  - OpenLAP uses a simpler linear tire model (mu_y + sens_y*(Ny-Fz))")
    print("  - OpenLAP does NOT model suspension roll-stiffness balance")
    print("  - On a pure skidpad both sims converge to v = sqrt(mu*g*r)")
    print("    At mu~1.48, r=30m:  v = sqrt(1.48*9.81*30) = 20.87 m/s")
    print("    Lap time estimate = 2*pi*30 / 20.87 ≈ 9.03 s")
    print("    (OpenLAP will be close to this; our GP-optimised result of 9.65 s")
    print("     reflects load-transfer penalty from our suspension model.)")

    # ── Monza comparison (full track, braking + acceleration) ────────────────
    print()
    print("=" * 60)
    print("Monza comparison (longitudinal dynamics)")
    print("=" * 60)
    from track import Track
    from simulator import LapSimulator
    monza = Track.monza()
    t_monza, v_monza = LapSimulator(car, monza).solve()
    print(f"Python sim on Monza: {t_monza:6.2f} s "
          f"({int(t_monza // 60)}:{t_monza % 60:05.2f})  "
          f"v_max = {v_monza.max() * 3.6:.0f} km/h")
    print()
    print("To compare in OpenLAP, no track export is needed — our Monza is read")
    print("directly from OpenLAP's own 'Autodromo Nazionale Monza.xlsx', so both")
    print("simulators run identical geometry. In MATLAB:")
    print("  1. Run OpenVEHICLE.m with filename = 'NC_Miata_OpenVEHICLE.xlsx'")
    print("  2. Run OpenTRACK.m   with filename = 'Autodromo Nazionale Monza.xlsx'")
    print("                            mode     = 'shape data'")
    print("  3. Run OpenLAP.m on those two .mat files; compare lap time + trace.")
