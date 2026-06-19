"""One-shot pipeline runner (the executable entry point).

Runs the whole chain end to end and shows the result as graphs:

    DoE (Latin Hypercube)  ->  GP surrogate  ->  Bayesian optimization
        ->  baseline vs optimized lap simulation  ->  comparison graphs
        ->  Assetto Corsa physics export

Run it from a checkout with ``python run_pipeline.py`` or, once frozen with
PyInstaller (see motorsports_pipeline.spec), by double-clicking the ``.exe``.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

from config import VehicleConfig
from track import Track


def _resource_dir() -> Path:
    """Folder that holds bundled data files (the track ``.xlsx``).

    When frozen by PyInstaller the data lives in the temporary ``_MEIPASS``
    extraction dir; otherwise it is just the script's own folder."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def _build_track(name: str) -> Track:
    if name.lower() == "monza":
        return Track.monza()
    return Track()  # default skidpad


def _frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Motorsport setup optimisation pipeline")
    p.add_argument("--car-file", default=None,
                   help="path to a car JSON file (default: built-in NC Miata)")
    p.add_argument("--track", default="monza", choices=["monza", "skidpad"],
                   help="track to optimise/compare on (default: monza)")
    p.add_argument("--samples", type=int, default=200,
                   help="DoE Latin-Hypercube samples (default: 200)")
    p.add_argument("--trials", type=int, default=2000,
                   help="Bayesian-optimisation trials (default: 2000)")
    p.add_argument("--outdir", default="output",
                   help="where comparison PNGs are written (default: output/)")
    p.add_argument("--export-dir", default="nc_miata_qss/data",
                   help="Assetto Corsa data folder to write (default: nc_miata_qss/data)")
    p.add_argument("--no-show", action="store_true",
                   help="save graphs without opening interactive windows")
    p.add_argument("--transient", action="store_true",
                   help="also run the 7-DOF transient damper analysis on the "
                        "optimized setup and save a damper-sensitivity report")
    p.add_argument("--optimize-dampers", action="store_true",
                   help="jointly optimise dampers with springs/ARBs (8 vars) "
                        "using a transient penalty surrogate; slower (builds a "
                        "transient DoE) but the dampers are tuned, not just fixed")
    p.add_argument("--damper-samples", type=int, default=120,
                   help="transient DoE samples for the damper penalty surrogate "
                        "(only used with --optimize-dampers; default: 120)")
    p.add_argument("--export-channels", default=None,
                   metavar="PATH",
                   help="write per-point channel data for the optimized setup to "
                        "PATH (.csv) or PATH (.xlsx); e.g. --export-channels "
                        "channels_monza.csv")
    p.add_argument("--track-map", default=None, metavar="CHANNEL",
                   help="save a bird's-eye track map of the optimized setup "
                        "coloured by CHANNEL; e.g. --track-map speed_kph")
    p.add_argument("--kpi", default=None, metavar="PARAM:MIN:MAX:STEPS:METRIC",
                   help="run a KPI sensitivity sweep and save a chart; e.g. "
                        "--kpi mass:960:1300:7:lap_time_s")
    args = p.parse_args(argv)

    if args.no_show:
        matplotlib.use("Agg")

    # Imports that pull in heavy deps are deferred so --help stays instant.
    from car_io import load_json
    from doe_sampler import (run_lhs_sweep, PARAM_NAMES, TRANSIENT_PARAM_NAMES,
                             LOWER, UPPER, snap_to_buildable, N_PER_KG_MM)
    from surrogate import LapTimeSurrogate
    from optimizer import make_objective
    from simulator import LapSimulator
    from export_to_ac import export_ac_car
    from analysis import (LapResult, compare_figure, laptime_bar_figure,
                          gforce_figure, corner_speed_figure, summary,
                          build_lap_channels, lap_summary,
                          gg_diagram_figure, channels_figure,
                          channels_to_csv, channels_to_excel,
                          track_map_figure)
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    track = _build_track(args.track)

    # Load car definition
    if args.car_file:
        base_cfg = load_json(args.car_file)
        print(f"\n=== {base_cfg.name}  |  track: {track.name} ===")
    else:
        base_cfg = VehicleConfig()
        print(f"\n=== NC Miata (default)  |  track: {track.name} ===")

    print(f"[1/5] DoE: {args.samples} Latin-Hypercube setups...")
    df = run_lhs_sweep(n_samples=args.samples, track=track, base_cfg=base_cfg)

    print("[2/5] Training Gaussian-Process surrogate...")
    surrogate = LapTimeSurrogate()
    surrogate.fit(df)

    if args.optimize_dampers:
        from optimizer import build_penalty_surrogate, run_joint_optimization
        print(f"[3/5] Joint optimisation (springs/ARBs + dampers, 8 vars)...")
        print(f"      building transient penalty surrogate "
              f"({args.damper_samples} transient DoE runs — this is the slow part)...")
        penalty = build_penalty_surrogate(n_doe_samples=args.damper_samples,
                                          track=track, base_cfg=base_cfg,
                                          force_retrain=True)
        _, buildable = run_joint_optimization(surrogate, penalty,
                                              n_trials=args.trials,
                                              base_cfg=base_cfg, track=track)
        setup_keys = TRANSIENT_PARAM_NAMES
    else:
        print(f"[3/5] Bayesian optimisation: {args.trials} surrogate queries...")
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(make_objective(surrogate), n_trials=args.trials,
                       show_progress_bar=not _frozen())
        best = study.best_params
        buildable = snap_to_buildable(best)
        print("  theoretical → buildable (snapped to purchasable increments):")
        for k in PARAM_NAMES:
            print(f"    {k:12s} = {best[k]/N_PER_KG_MM:6.2f} kg/mm  →  "
                  f"{buildable[k]/N_PER_KG_MM:.2f} kg/mm")
        setup_keys = PARAM_NAMES

    print("[4/5] Simulating baseline vs optimized (using buildable setup)...")
    opt_cfg = replace(base_cfg, **{k: buildable[k] for k in setup_keys})
    from vehicle_model import NCMiata
    base = LapResult.from_sim("Baseline", LapSimulator(NCMiata(base_cfg), track))
    opt = LapResult.from_sim("Optimized", LapSimulator(NCMiata(opt_cfg), track))

    print(f"    Baseline : {base.lap_time:7.3f}s   "
          f"(v {base.v_min*3.6:5.1f}-{base.v_max*3.6:5.1f} km/h)")
    print(f"    Optimized: {opt.lap_time:7.3f}s   "
          f"(v {opt.v_min*3.6:5.1f}-{opt.v_max*3.6:5.1f} km/h)")
    print(f"    Gain     : {base.lap_time - opt.lap_time:+.3f}s")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig_cmp = compare_figure(base, opt)
    fig_bar = laptime_bar_figure([base, opt])
    fig_g = gforce_figure(base, opt)
    fig_corner = corner_speed_figure(base, opt)
    cmp_png = outdir / f"comparison_{args.track}.png"
    bar_png = outdir / f"laptime_{args.track}.png"
    g_png = outdir / f"gforce_{args.track}.png"
    corner_png = outdir / f"corner_speed_{args.track}.png"
    fig_cmp.savefig(cmp_png, dpi=120)
    fig_bar.savefig(bar_png, dpi=120)
    fig_g.savefig(g_png, dpi=120)
    fig_corner.savefig(corner_png, dpi=120)
    print(f"    graphs -> {cmp_png}, {bar_png}, {g_png}, {corner_png}")

    # Per-point channel data (built once; shared by export + track map)
    ch_df = None
    if args.export_channels or args.track_map:
        ch_df = build_lap_channels(opt, NCMiata(opt_cfg))

    # Optional per-point channel export
    if args.export_channels:
        ch_path = Path(args.export_channels)
        smry    = lap_summary(ch_df)
        print(f"[+] Channel summary — optimized setup on {track.name}:")
        print(f"    v_max {smry['v_max_kph']:.1f} km/h  |  "
              f"max lat g {smry['max_lat_g']:.2f}  |  "
              f"max power {smry['max_power_kW']:.1f} kW  |  "
              f"gears used {smry['gears_used']}")
        print(f"    throttle {smry['pct_accelerating']:.0f}%  |  "
              f"braking {smry['pct_braking']:.0f}%  |  "
              f"cornering {smry['pct_cornering']:.0f}%")
        if ch_path.suffix.lower() == ".xlsx":
            channels_to_excel(ch_df, str(ch_path))
        else:
            channels_to_csv(ch_df, str(ch_path))
        print(f"    channel data -> {ch_path}")

        # Save GG and channel figures alongside the other PNGs
        fig_gg  = gg_diagram_figure(ch_df, label=f"{opt_cfg.name} — {track.name}")
        fig_ch  = channels_figure(ch_df,   label=f"{opt_cfg.name} — {track.name}")
        gg_png  = outdir / f"gg_diagram_{args.track}.png"
        ch_png  = outdir / f"channels_{args.track}.png"
        fig_gg.savefig(gg_png, dpi=120)
        fig_ch.savefig(ch_png, dpi=120)
        print(f"    figures -> {gg_png}, {ch_png}")

    # Optional track map coloured by a channel
    if args.track_map:
        fig_tm = track_map_figure(ch_df, args.track_map,
                                  label=f"{opt_cfg.name} — {track.name}")
        tm_png = outdir / f"track_map_{args.track}.png"
        fig_tm.savefig(tm_png, dpi=120)
        print(f"[+] Track map ({args.track_map}) -> {tm_png}")

    print(f"[5/5] Exporting Assetto Corsa physics -> {args.export_dir}")
    export_ac_car(opt_cfg, output_dir=args.export_dir)

    if args.transient:
        from transient_lap import run_transient_lap
        from analysis import transient_report_figure
        print("[+] Transient damper analysis (7-DOF) on the optimized setup...")
        tr = run_transient_lap(NCMiata(opt_cfg), track)
        print(f"    damper penalty : {tr.delta:+.3f}s  "
              f"(min retained grip {tr.grip_scale.min():.3f})")
        fig_tr = transient_report_figure(tr, title=f"{track.name} — optimized")
        tr_png = outdir / f"transient_{args.track}.png"
        fig_tr.savefig(tr_png, dpi=120)
        print(f"    report -> {tr_png}")

    # Optional KPI sensitivity sweep
    if args.kpi:
        import numpy as np
        from analysis import (run_kpi_sweep, kpi_chart_figure,
                              KPI_PARAMS, KPI_METRICS)
        try:
            pname, pmin, pmax, psteps, pmetric = args.kpi.split(":")
            values = np.linspace(float(pmin), float(pmax), int(psteps))
        except ValueError:
            print(f"[!] --kpi must be PARAM:MIN:MAX:STEPS:METRIC, got {args.kpi!r}")
            return 2
        if pname not in KPI_PARAMS:
            print(f"[!] unknown KPI parameter {pname!r}; choose from "
                  f"{', '.join(KPI_PARAMS)}")
            return 2
        if pmetric not in KPI_METRICS:
            print(f"[!] unknown KPI metric {pmetric!r}; choose from "
                  f"{', '.join(KPI_METRICS)}")
            return 2
        print(f"[+] KPI sweep: {pname} over {len(values)} values "
              f"[{values[0]:g} … {values[-1]:g}], metric {pmetric}...")
        sweep = run_kpi_sweep(base_cfg, track, pname, values)
        for _, r in sweep.iterrows():
            print(f"    {pname}={r[pname]:.4g}  →  {pmetric}={r[pmetric]:.4g}")
        fig_kpi = kpi_chart_figure(sweep, pname, pmetric, label=base_cfg.name)
        kpi_png = outdir / f"kpi_{pname}_{pmetric}_{args.track}.png"
        fig_kpi.savefig(kpi_png, dpi=120)
        print(f"    chart -> {kpi_png}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()

    if _frozen():
        input("\nDone. Press Enter to close...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
