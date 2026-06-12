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
    args = p.parse_args(argv)

    if args.no_show:
        matplotlib.use("Agg")

    # Imports that pull in heavy deps are deferred so --help stays instant.
    from car_io import load_json
    from doe_sampler import run_lhs_sweep, PARAM_NAMES, LOWER, UPPER
    from surrogate import LapTimeSurrogate
    from optimizer import make_objective
    from simulator import LapSimulator
    from export_to_ac import export_ac_car
    from analysis import LapResult, compare_figure, laptime_bar_figure, summary
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

    print(f"[3/5] Bayesian optimisation: {args.trials} surrogate queries...")
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(make_objective(surrogate), n_trials=args.trials,
                   show_progress_bar=not _frozen())
    best = study.best_params
    print("  optimal setup:")
    for k in PARAM_NAMES:
        print(f"    {k:12s} = {best[k]/10000:6.2f} kg/mm")

    print("[4/5] Simulating baseline vs optimized...")
    opt_cfg = replace(base_cfg, **{k: best[k] for k in PARAM_NAMES})
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
    cmp_png = outdir / f"comparison_{args.track}.png"
    bar_png = outdir / f"laptime_{args.track}.png"
    fig_cmp.savefig(cmp_png, dpi=120)
    fig_bar.savefig(bar_png, dpi=120)
    print(f"    graphs -> {cmp_png}  and  {bar_png}")

    print(f"[5/5] Exporting Assetto Corsa physics -> {args.export_dir}")
    export_ac_car(opt_cfg, output_dir=args.export_dir)

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()

    if _frozen():
        input("\nDone. Press Enter to close...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
