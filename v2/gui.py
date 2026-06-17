"""gui.py — Tkinter GUI for the motorsport setup-optimisation pipeline.

Launched instead of run_pipeline.py when you want a graphical interface.
Double-click the .exe (or ``python gui.py``) to open the window.

Layout
------
  [Car Definition]  — load/save JSON, editable key params
  [Run Settings]    — track, DoE samples, Bayesian-opt trials
  [Run / Export]    — buttons
  [Notebook]        — Log | Speed trace | Lap-time bar
"""
from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import matplotlib
matplotlib.use("Agg")          # must come before pyplot import
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import VehicleConfig
from car_io import load_json, save_json


# ---------------------------------------------------------------------------
# Parameter definitions shown in the "Car Definition" panel
# (field_name, label, unit, min, max)
# ---------------------------------------------------------------------------
_PARAMS = [
    ("name",          "Car name",          "",       None,  None),
    ("mass",          "Mass",              "kg",     400,   3000),
    ("wheelbase",     "Wheelbase",         "m",      1.5,   4.0),
    ("weight_dist_f", "Weight dist front", "frac",   0.30,  0.70),
    ("cg_height",     "CG height",         "m",      0.20,  0.80),
    ("spring_k_f",    "Spring front",      "N/m",    10000, 200000),
    ("spring_k_r",    "Spring rear",       "N/m",    10000, 200000),
    ("arb_k_f",         "ARB front",           "N/m",    0,     50000),
    ("arb_k_r",         "ARB rear",            "N/m",    0,     30000),
    # --- Dampers (data / AC export only; not used by QSS solver) ---
    ("damper_bump_f",    "Damper bump front",   "N·s/m",  500,   8000),
    ("damper_rebound_f", "Damper rebound front","N·s/m",  500,   12000),
    ("damper_bump_r",    "Damper bump rear",    "N·s/m",  500,   8000),
    ("damper_rebound_r", "Damper rebound rear", "N·s/m",  500,   12000),
    ("cl_a",             "Cl·A (downforce)",    "m²",     0,     5.0),
    ("cd_a",          "Cd·A (drag)",       "m²",     0.1,   2.0),
    ("aero_balance_f","Aero balance front","frac",   0.20,  0.80),
    ("power_max",     "Peak power",        "W",      20000, 1000000),
    ("tyre_radius",   "Tyre radius",       "m",      0.25,  0.40),
    ("Cr",            "Rolling resist.",   "",       0.005, 0.05),
]


class PipelineGUI(tk.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("Motorsport Pipeline")
        self.resizable(True, True)
        self.minsize(860, 640)

        self._cfg = VehicleConfig()          # current car config
        self._log_queue: queue.Queue[str] = queue.Queue()
        self._result_figs: list = []         # [cmp_fig, bar_fig] after run
        self._export_cfg: VehicleConfig | None = None  # optimised config

        self._build_ui()
        self._load_cfg_into_fields(self._cfg)
        self._poll_log()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=4)

        self._build_car_frame(top)
        self._build_run_frame(top)

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # Tab 0: Log
        log_frame = ttk.Frame(nb)
        nb.add(log_frame, text="Log")
        self._log_box = scrolledtext.ScrolledText(log_frame, state="disabled",
                                                  wrap=tk.WORD, font=("Courier", 9))
        self._log_box.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Speed trace
        self._tab_speed = ttk.Frame(nb)
        nb.add(self._tab_speed, text="Speed trace")

        # Tab 2: Lap-time bar
        self._tab_bar = ttk.Frame(nb)
        nb.add(self._tab_bar, text="Lap time")

        # Tab 3: G-force traces
        self._tab_gforce = ttk.Frame(nb)
        nb.add(self._tab_gforce, text="G-forces")

        # Tab 4: Corner min-speeds
        self._tab_corner = ttk.Frame(nb)
        nb.add(self._tab_corner, text="Corner speeds")

        # Tab 5: Damper analysis (7-DOF transient)
        self._tab_transient = ttk.Frame(nb)
        nb.add(self._tab_transient, text="Damper analysis")

        self._notebook = nb

    def _build_car_frame(self, parent):
        frm = ttk.LabelFrame(parent, text="Car Definition", padding=6)
        frm.pack(fill=tk.X, pady=(0, 4))

        btn_row = ttk.Frame(frm)
        btn_row.grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 4))
        ttk.Button(btn_row, text="Load JSON…",  command=self._on_load).pack(side=tk.LEFT, padx=(0,4))
        ttk.Button(btn_row, text="Save JSON…",  command=self._on_save).pack(side=tk.LEFT, padx=(0,4))
        ttk.Button(btn_row, text="Reset to NC Miata", command=self._on_reset).pack(side=tk.LEFT)

        self._fields: dict[str, tk.StringVar] = {}
        for idx, (fname, label, unit, lo, hi) in enumerate(_PARAMS):
            row = (idx // 2) + 1
            col = (idx % 2) * 3
            ttk.Label(frm, text=label + ":").grid(row=row, column=col, sticky=tk.E, padx=(8,2))
            var = tk.StringVar()
            self._fields[fname] = var
            entry = ttk.Entry(frm, textvariable=var, width=14)
            entry.grid(row=row, column=col+1, sticky=tk.W, padx=(0,2))
            if unit:
                ttk.Label(frm, text=unit, foreground="#666").grid(row=row, column=col+2, sticky=tk.W)

    def _build_run_frame(self, parent):
        frm = ttk.LabelFrame(parent, text="Run Settings", padding=6)
        frm.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(frm, text="Track:").grid(row=0, column=0, sticky=tk.E, padx=(0,4))
        self._track_var = tk.StringVar(value="monza")
        ttk.Combobox(frm, textvariable=self._track_var, values=["monza","skidpad"],
                     state="readonly", width=10).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(frm, text="DoE samples:").grid(row=0, column=2, sticky=tk.E, padx=(16,4))
        self._samples_var = tk.StringVar(value="200")
        ttk.Spinbox(frm, textvariable=self._samples_var, from_=10, to=2000, increment=10,
                    width=7).grid(row=0, column=3, sticky=tk.W)

        ttk.Label(frm, text="Opt trials:").grid(row=0, column=4, sticky=tk.E, padx=(16,4))
        self._trials_var = tk.StringVar(value="2000")
        ttk.Spinbox(frm, textvariable=self._trials_var, from_=100, to=20000, increment=100,
                    width=8).grid(row=0, column=5, sticky=tk.W)

        self._run_btn = ttk.Button(frm, text="▶  Run Pipeline", command=self._on_run)
        self._run_btn.grid(row=0, column=6, padx=(24,4))

        self._export_btn = ttk.Button(frm, text="Export to Assetto Corsa…",
                                      command=self._on_export, state="disabled")
        self._export_btn.grid(row=0, column=7, padx=(4,0))

        self._transient_btn = ttk.Button(frm, text="Damper analysis (transient)",
                                         command=self._on_transient)
        self._transient_btn.grid(row=1, column=6, columnspan=2, pady=(6, 0))

    # ------------------------------------------------------------------ #
    # Field helpers
    # ------------------------------------------------------------------ #

    def _load_cfg_into_fields(self, cfg: VehicleConfig):
        for fname, var in self._fields.items():
            val = getattr(cfg, fname, "")
            var.set(str(val) if not isinstance(val, float) else f"{val:g}")

    def _read_fields_into_cfg(self) -> VehicleConfig:
        """Parse the editable fields and return an updated VehicleConfig."""
        kwargs = {}
        for fname, var in self._fields.items():
            raw = var.get().strip()
            # type-cast to match the dataclass field type
            orig = getattr(self._cfg, fname)
            if isinstance(orig, str):
                kwargs[fname] = raw
            elif isinstance(orig, float):
                kwargs[fname] = float(raw)
            elif isinstance(orig, int):
                kwargs[fname] = int(raw)
        return replace(self._cfg, **kwargs)

    # ------------------------------------------------------------------ #
    # Button callbacks
    # ------------------------------------------------------------------ #

    def _on_load(self):
        path = filedialog.askopenfilename(
            title="Load car JSON",
            filetypes=[("Car JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            self._cfg = load_json(path)
            self._load_cfg_into_fields(self._cfg)
            self._log(f"Loaded car: {self._cfg.name}  ({path})\n")
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def _on_save(self):
        try:
            self._cfg = self._read_fields_into_cfg()
        except Exception as exc:
            messagebox.showerror("Parse error", str(exc))
            return
        path = filedialog.asksaveasfilename(
            title="Save car JSON",
            defaultextension=".json",
            filetypes=[("Car JSON", "*.json")])
        if not path:
            return
        try:
            save_json(self._cfg, path)
            self._log(f"Saved car to {path}\n")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))

    def _on_reset(self):
        self._cfg = VehicleConfig()
        self._load_cfg_into_fields(self._cfg)
        self._log("Reset to NC Miata defaults.\n")

    def _on_run(self):
        try:
            self._cfg = self._read_fields_into_cfg()
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return
        self._run_btn.config(state="disabled")
        self._export_btn.config(state="disabled")
        self._log("─" * 60 + "\n")
        thread = threading.Thread(target=self._run_pipeline, daemon=True)
        thread.start()

    def _on_export(self):
        if self._export_cfg is None:
            return
        folder = filedialog.askdirectory(title="Assetto Corsa data folder")
        if not folder:
            return
        from export_to_ac import export_ac_car
        try:
            export_ac_car(self._export_cfg, output_dir=folder)
            self._log(f"AC physics exported to {folder}\n")
            messagebox.showinfo("Export done", f"Physics written to:\n{folder}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))

    def _on_transient(self):
        """Run the 7-DOF transient damper analysis on the current car config.

        Works standalone (no full optimisation needed): it scores how much the
        *current* damper settings cost in lap time through load-transfer
        transients, so you can edit the damper fields and re-run to compare."""
        try:
            self._cfg = self._read_fields_into_cfg()
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return
        self._transient_btn.config(state="disabled")
        self._log("─" * 60 + "\n")
        thread = threading.Thread(target=self._run_transient, daemon=True)
        thread.start()

    def _run_transient(self):
        import sys as _sys
        orig_stdout = _sys.stdout
        _sys.stdout = _QueueWriter(self._log_queue)
        try:
            self._transient_body()
        except Exception as exc:
            self._log_queue.put(f"\n[ERROR] {exc}\n")
        finally:
            _sys.stdout = orig_stdout
            self.after(0, lambda: self._transient_btn.config(state="normal"))

    def _transient_body(self):
        from track import Track
        from vehicle_model import NCMiata
        from transient_lap import run_transient_lap
        from analysis import transient_report_figure

        track_name = self._track_var.get()
        track = Track.monza() if track_name == "monza" else Track()
        cfg = self._cfg
        print(f"Damper analysis (7-DOF transient)  |  {cfg.name}  |  {track.name}")
        if track_name == "monza":
            print("  Monza is ~3000 points over several laps — expect ~30-60s …")
        tr = run_transient_lap(NCMiata(cfg), track)
        print(f"  damper penalty   : {tr.delta:+.3f}s "
              f"(QSS {tr.lap_time_qss:.3f}s → transient {tr.lap_time_transient:.3f}s)")
        print(f"  min retained grip: {tr.grip_scale.min():.3f}")
        fig = transient_report_figure(tr, title=f"{track.name} — {cfg.name}")
        self.after(0, lambda: self._show_transient_figure(fig))
        print("Done!\n")

    def _show_transient_figure(self, fig):
        tab = self._tab_transient
        for widget in tab.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._notebook.select(tab)

    # ------------------------------------------------------------------ #
    # Pipeline (runs in background thread)
    # ------------------------------------------------------------------ #

    def _run_pipeline(self):
        import sys as _sys
        # Redirect stdout to the log queue
        orig_stdout = _sys.stdout
        _sys.stdout = _QueueWriter(self._log_queue)
        try:
            self._pipeline_body()
        except Exception as exc:
            self._log_queue.put(f"\n[ERROR] {exc}\n")
        finally:
            _sys.stdout = orig_stdout
            self.after(0, lambda: self._run_btn.config(state="normal"))

    def _pipeline_body(self):
        import optuna
        from dataclasses import replace as _replace
        from doe_sampler import (run_lhs_sweep, PARAM_NAMES,
                                  snap_to_buildable, N_PER_KG_MM)
        from surrogate import LapTimeSurrogate
        from optimizer import make_objective
        from simulator import LapSimulator
        from track import Track
        from vehicle_model import NCMiata
        from analysis import (LapResult, compare_figure, laptime_bar_figure,
                              gforce_figure, corner_speed_figure)

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        track_name = self._track_var.get()
        n_samples  = int(self._samples_var.get())
        n_trials   = int(self._trials_var.get())
        base_cfg   = self._cfg

        track = Track.monza() if track_name == "monza" else Track()

        print(f"Car: {base_cfg.name}  |  track: {track.name}")
        print(f"[1/4] DoE — {n_samples} LHS setups …")
        df = run_lhs_sweep(n_samples=n_samples, track=track, base_cfg=base_cfg)

        print("[2/4] Training GP surrogate …")
        surrogate = LapTimeSurrogate()
        surrogate.fit(df)

        print(f"[3/4] Bayesian optimisation — {n_trials} trials …")
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(make_objective(surrogate), n_trials=n_trials,
                       show_progress_bar=False)
        best = study.best_params
        buildable = snap_to_buildable(best)
        print("  theoretical → buildable (snapped to purchasable increments):")
        for k in PARAM_NAMES:
            unit = "kg/mm" if "spring" in k else "kg/mm equiv"
            print(f"    {k:12s} = {best[k]/N_PER_KG_MM:.2f}  →  "
                  f"{buildable[k]/N_PER_KG_MM:.2f} {unit}")

        print("[4/4] Simulating baseline vs optimized (buildable setup) …")
        opt_cfg = _replace(base_cfg, **{k: buildable[k] for k in PARAM_NAMES})
        base_res = LapResult.from_sim("Baseline",  LapSimulator(NCMiata(base_cfg), track))
        opt_res  = LapResult.from_sim("Optimized", LapSimulator(NCMiata(opt_cfg),  track))
        print(f"  Baseline : {base_res.lap_time:.3f}s")
        print(f"  Optimized: {opt_res.lap_time:.3f}s")
        print(f"  Gain     : {base_res.lap_time - opt_res.lap_time:+.3f}s")

        self._export_cfg = opt_cfg

        fig_cmp = compare_figure(base_res, opt_res)
        fig_bar = laptime_bar_figure([base_res, opt_res])
        fig_g = gforce_figure(base_res, opt_res)
        fig_corner = corner_speed_figure(base_res, opt_res)
        self.after(0, lambda: self._show_figures(fig_cmp, fig_bar, fig_g, fig_corner))
        self.after(0, lambda: self._export_btn.config(state="normal"))
        print("Done!\n")

    # ------------------------------------------------------------------ #
    # Figure embedding
    # ------------------------------------------------------------------ #

    def _show_figures(self, fig_cmp, fig_bar, fig_g, fig_corner):
        for tab, fig in ((self._tab_speed, fig_cmp),
                         (self._tab_bar, fig_bar),
                         (self._tab_gforce, fig_g),
                         (self._tab_corner, fig_corner)):
            for widget in tab.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Switch to the speed-trace tab
        self._notebook.select(self._tab_speed)

    # ------------------------------------------------------------------ #
    # Log helpers
    # ------------------------------------------------------------------ #

    def _log(self, msg: str):
        self._log_box.config(state="normal")
        self._log_box.insert(tk.END, msg)
        self._log_box.see(tk.END)
        self._log_box.config(state="disabled")

    def _poll_log(self):
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self._log(msg)
        except queue.Empty:
            pass
        self.after(100, self._poll_log)


# ---------------------------------------------------------------------------
# stdout redirector for the background thread
# ---------------------------------------------------------------------------

class _QueueWriter:
    def __init__(self, q: queue.Queue):
        self._q = q

    def write(self, text: str):
        if text:
            self._q.put(text)

    def flush(self):
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = PipelineGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

