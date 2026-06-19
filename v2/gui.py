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

import platform
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
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

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
        self._apply_screen_geometry()

        self._cfg = VehicleConfig()          # current car config
        self._log_queue: queue.Queue[str] = queue.Queue()
        self._result_figs: list = []         # [cmp_fig, bar_fig] after run
        self._export_cfg: VehicleConfig | None = None  # optimised config

        self._build_ui()
        self._load_cfg_into_fields(self._cfg)
        self._poll_log()

    # ------------------------------------------------------------------ #
    # Window sizing
    # ------------------------------------------------------------------ #

    def _apply_screen_geometry(self):
        """Size the window to the host screen so the EXE opens at a sensible
        size on any monitor instead of a fixed geometry."""
        self.minsize(860, 640)
        try:
            self.update_idletasks()
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            # Scale Tk fonts/widgets to the monitor DPI (crisp on hi-DPI).
            try:
                self.tk.call("tk", "scaling", self.winfo_fpixels("1i") / 72.0)
            except Exception:
                pass
            w, h = max(860, int(sw * 0.85)), max(640, int(sh * 0.85))
            x, y = (sw - w) // 2, max(0, (sh - h) // 3)
            self.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass
        # Best-effort true maximize where the window manager supports it.
        try:
            self.state("zoomed")
        except Exception:
            try:
                self.attributes("-zoomed", True)
            except Exception:
                pass

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

        # Tab 6: G-G diagram
        self._tab_gg = ttk.Frame(nb)
        nb.add(self._tab_gg, text="G-G diagram")

        # Tab 7: Channel data
        self._tab_channels = ttk.Frame(nb)
        nb.add(self._tab_channels, text="Channels")

        # Tab 8: Track map (colour-by-any-channel)
        self._tab_trackmap = ttk.Frame(nb)
        nb.add(self._tab_trackmap, text="Track map")
        self._build_trackmap_tab(self._tab_trackmap)

        # Tab 9: Custom chart (any X vs any Y)
        self._tab_customchart = ttk.Frame(nb)
        nb.add(self._tab_customchart, text="Custom chart")
        self._build_customchart_tab(self._tab_customchart)

        # Tab 10: KPI sweep (parameter sensitivity)
        self._tab_kpi = ttk.Frame(nb)
        nb.add(self._tab_kpi, text="KPI sweep")
        self._build_kpi_tab(self._tab_kpi)

        self._notebook = nb
        self._channels_df = None   # last computed channel DataFrame
        self._kpi_df = None        # last computed KPI sweep DataFrame

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

        self._export_data_btn = ttk.Button(frm, text="Export data (CSV/Excel)…",
                                           command=self._on_export_data, state="disabled")
        self._export_data_btn.grid(row=0, column=8, padx=(4,0))

        ttk.Label(frm, text="Damper DoE:").grid(row=1, column=0, sticky=tk.E, padx=(0,4))
        self._damper_samples_var = tk.StringVar(value="60")
        ttk.Spinbox(frm, textvariable=self._damper_samples_var, from_=10, to=500,
                    increment=10, width=7).grid(row=1, column=1, sticky=tk.W)

        self._transient_btn = ttk.Button(frm, text="Damper analysis (transient)",
                                         command=self._on_transient)
        self._transient_btn.grid(row=1, column=6, columnspan=2, pady=(6, 0))

        self._optimize_btn = ttk.Button(frm, text="\u2699  Optimize dampers (slow)",
                                        command=self._on_optimize_dampers)
        self._optimize_btn.grid(row=2, column=6, columnspan=2, pady=(6, 0))

    # ------------------------------------------------------------------ #
    # Data-explorer tabs (track map / custom chart / KPI sweep)
    # ------------------------------------------------------------------ #

    def _build_trackmap_tab(self, tab):
        ctl = ttk.Frame(tab)
        ctl.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(ctl, text="Colour by:").pack(side=tk.LEFT, padx=(0, 4))
        self._tm_channel_var = tk.StringVar(value="speed_kph")
        self._tm_channel_combo = ttk.Combobox(
            ctl, textvariable=self._tm_channel_var, state="readonly", width=24)
        self._tm_channel_combo.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(ctl, text="Line width:").pack(side=tk.LEFT, padx=(0, 4))
        self._tm_width_var = tk.StringVar(value="4")
        ttk.Spinbox(ctl, textvariable=self._tm_width_var, from_=1, to=12,
                    increment=1, width=5).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(ctl, text="Draw", command=self._on_draw_trackmap).pack(side=tk.LEFT)
        self._trackmap_plot = ttk.Frame(tab)
        self._trackmap_plot.pack(fill=tk.BOTH, expand=True)

    def _build_customchart_tab(self, tab):
        ctl = ttk.Frame(tab)
        ctl.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(ctl, text="X:").pack(side=tk.LEFT, padx=(0, 4))
        self._cc_x_var = tk.StringVar(value="distance_m")
        self._cc_x_combo = ttk.Combobox(
            ctl, textvariable=self._cc_x_var, state="readonly", width=22)
        self._cc_x_combo.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(ctl, text="Y:").pack(side=tk.LEFT, padx=(0, 4))
        self._cc_y_var = tk.StringVar(value="speed_kph")
        self._cc_y_combo = ttk.Combobox(
            ctl, textvariable=self._cc_y_var, state="readonly", width=22)
        self._cc_y_combo.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(ctl, text="Type:").pack(side=tk.LEFT, padx=(0, 4))
        self._cc_kind_var = tk.StringVar(value="line")
        ttk.Combobox(ctl, textvariable=self._cc_kind_var, state="readonly",
                     values=["line", "scatter"], width=8).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(ctl, text="Plot", command=self._on_draw_customchart).pack(side=tk.LEFT)
        self._customchart_plot = ttk.Frame(tab)
        self._customchart_plot.pack(fill=tk.BOTH, expand=True)

    def _build_kpi_tab(self, tab):
        from analysis import KPI_PARAMS, KPI_METRICS
        ctl = ttk.Frame(tab)
        ctl.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(ctl, text="Parameter:").pack(side=tk.LEFT, padx=(0, 4))
        self._kpi_param_var = tk.StringVar(value="mass")
        param_combo = ttk.Combobox(ctl, textvariable=self._kpi_param_var,
                                   state="readonly", width=16,
                                   values=list(KPI_PARAMS.keys()))
        param_combo.pack(side=tk.LEFT, padx=(0, 8))
        param_combo.bind("<<ComboboxSelected>>", self._on_kpi_param_changed)
        ttk.Label(ctl, text="Min:").pack(side=tk.LEFT, padx=(0, 2))
        self._kpi_min_var = tk.StringVar(value="960")
        ttk.Entry(ctl, textvariable=self._kpi_min_var, width=9).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(ctl, text="Max:").pack(side=tk.LEFT, padx=(0, 2))
        self._kpi_max_var = tk.StringVar(value="1300")
        ttk.Entry(ctl, textvariable=self._kpi_max_var, width=9).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(ctl, text="Steps:").pack(side=tk.LEFT, padx=(0, 2))
        self._kpi_steps_var = tk.StringVar(value="7")
        ttk.Spinbox(ctl, textvariable=self._kpi_steps_var, from_=3, to=30,
                    increment=1, width=5).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(ctl, text="Metric:").pack(side=tk.LEFT, padx=(0, 2))
        self._kpi_metric_var = tk.StringVar(value="lap_time_s")
        ttk.Combobox(ctl, textvariable=self._kpi_metric_var, state="readonly",
                     width=16, values=list(KPI_METRICS.keys())).pack(side=tk.LEFT, padx=(0, 12))
        self._kpi_btn = ttk.Button(ctl, text="Run sweep", command=self._on_run_kpi)
        self._kpi_btn.pack(side=tk.LEFT)
        self._kpi_plot = ttk.Frame(tab)
        self._kpi_plot.pack(fill=tk.BOTH, expand=True)

    def _on_kpi_param_changed(self, _event=None):
        """Auto-fill the sweep min/max around the current config value."""
        param = self._kpi_param_var.get()
        try:
            base = float(getattr(self._read_fields_into_cfg(), param))
        except Exception:
            try:
                base = float(getattr(self._cfg, param))
            except Exception:
                return
        lo, hi = (0.85 * base, 1.15 * base) if base else (0.0, 1.0)
        self._kpi_min_var.set(f"{lo:g}")
        self._kpi_max_var.set(f"{hi:g}")

    def _populate_channel_selectors(self):
        """Refresh the track-map / custom-chart channel lists from the latest
        channel DataFrame, keeping any still-valid current selection."""
        if self._channels_df is None:
            return
        from analysis import numeric_channels
        cols = numeric_channels(self._channels_df)
        if not cols:
            return
        self._tm_channel_combo["values"] = cols
        self._cc_x_combo["values"] = cols
        self._cc_y_combo["values"] = cols
        if self._tm_channel_var.get() not in cols:
            self._tm_channel_var.set("speed_kph" if "speed_kph" in cols else cols[0])
        if self._cc_x_var.get() not in cols:
            self._cc_x_var.set("distance_m" if "distance_m" in cols else cols[0])
        if self._cc_y_var.get() not in cols:
            self._cc_y_var.set("speed_kph" if "speed_kph" in cols else cols[-1])

    def _on_draw_trackmap(self):
        if self._channels_df is None:
            messagebox.showinfo("No data", "Run the pipeline first to generate "
                                "channel data, then draw the track map.")
            return
        from analysis import track_map_figure
        try:
            width = float(self._tm_width_var.get())
        except ValueError:
            width = 4.0
        fig = track_map_figure(self._channels_df, self._tm_channel_var.get(),
                               line_width=width, label=self._cfg.name)
        self._embed_figure(self._trackmap_plot, fig)

    def _on_draw_customchart(self):
        if self._channels_df is None:
            messagebox.showinfo("No data", "Run the pipeline first to generate "
                                "channel data, then plot a custom chart.")
            return
        from analysis import channel_chart_figure
        fig = channel_chart_figure(self._channels_df, self._cc_x_var.get(),
                                   self._cc_y_var.get(), self._cc_kind_var.get(),
                                   label=self._cfg.name)
        self._embed_figure(self._customchart_plot, fig)

    def _on_run_kpi(self):
        try:
            self._cfg = self._read_fields_into_cfg()
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return
        try:
            mn = float(self._kpi_min_var.get())
            mx = float(self._kpi_max_var.get())
            steps = int(self._kpi_steps_var.get())
            assert steps >= 2 and mx > mn
        except (ValueError, AssertionError):
            messagebox.showerror("Invalid sweep",
                                 "Need Min < Max and Steps ≥ 2.")
            return
        self._kpi_btn.config(state="disabled")
        self._log("─" * 60 + "\n")
        thread = threading.Thread(
            target=self._run_kpi, args=(mn, mx, steps), daemon=True)
        thread.start()

    def _run_kpi(self, mn, mx, steps):
        import sys as _sys
        orig_stdout = _sys.stdout
        _sys.stdout = _QueueWriter(self._log_queue)
        try:
            self._kpi_body(mn, mx, steps)
        except Exception as exc:
            self._log_queue.put(f"\n[ERROR] {exc}\n")
        finally:
            _sys.stdout = orig_stdout
            self.after(0, lambda: self._kpi_btn.config(state="normal"))

    def _kpi_body(self, mn, mx, steps):
        import numpy as np
        from track import Track
        from analysis import run_kpi_sweep, kpi_chart_figure

        param = self._kpi_param_var.get()
        metric = self._kpi_metric_var.get()
        track = Track.monza() if self._track_var.get() == "monza" else Track()
        values = np.linspace(mn, mx, steps)
        print(f"KPI sweep  |  {self._cfg.name}  |  {track.name}")
        print(f"  sweeping {param} over {steps} values "
              f"[{mn:g} … {mx:g}], metric = {metric}")
        sweep = run_kpi_sweep(self._cfg, track, param, values)
        self._kpi_df = sweep
        for _, r in sweep.iterrows():
            print(f"    {param}={r[param]:.4g}  →  {metric}={r[metric]:.4g}")
        fig = kpi_chart_figure(sweep, param, metric, label=self._cfg.name)
        self.after(0, lambda: self._show_kpi_figure(fig))
        print("Done!\n")

    def _show_kpi_figure(self, fig):
        self._embed_figure(self._kpi_plot, fig)
        self._notebook.select(self._tab_kpi)

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
        self._optimize_btn.config(state="disabled")
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

    def _on_export_data(self):
        """Save the last-computed channel DataFrame to CSV or Excel."""
        if self._channels_df is None:
            return
        path = filedialog.asksaveasfilename(
            title="Export channel data",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx"), ("All files", "*.*")])
        if not path:
            return
        from analysis import channels_to_csv, channels_to_excel
        try:
            if path.lower().endswith(".xlsx"):
                channels_to_excel(self._channels_df, path)
            else:
                channels_to_csv(self._channels_df, path)
            self._log(f"Channel data exported to {path}\n")
            messagebox.showinfo("Export done", f"Data written to:\n{path}")
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

    def _embed_figure(self, tab, fig):
        """Draw *fig* into *tab* with an interactive pan/zoom/save toolbar."""
        for widget in tab.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, tab, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._enable_scroll_zoom(canvas)

    @staticmethod
    def _enable_scroll_zoom(canvas, base_scale: float = 1.2):
        """Zoom the axes under the cursor in/out on mouse-wheel scroll,
        keeping the point under the cursor fixed (scroll up = zoom in).

        This supplements the toolbar's rectangle-zoom: just point at a
        region and roll the wheel. ``home`` on the toolbar resets the view.
        """
        def _on_scroll(event):
            ax = event.inaxes
            if ax is None or event.xdata is None or event.ydata is None:
                return
            if event.button == "up":
                scale = 1.0 / base_scale
            elif event.button == "down":
                scale = base_scale
            else:
                return
            xdata, ydata = event.xdata, event.ydata
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            ax.set_xlim(xdata - (xdata - x0) * scale,
                        xdata + (x1 - xdata) * scale)
            ax.set_ylim(ydata - (ydata - y0) * scale,
                        ydata + (y1 - ydata) * scale)
            canvas.draw_idle()

        canvas.mpl_connect("scroll_event", _on_scroll)

    def _show_transient_figure(self, fig):
        self._embed_figure(self._tab_transient, fig)
        self._notebook.select(self._tab_transient)

    # ------------------------------------------------------------------ #
    # Optimize dampers (joint 8-variable optimisation, background thread)
    # ------------------------------------------------------------------ #

    def _on_optimize_dampers(self):
        try:
            self._cfg = self._read_fields_into_cfg()
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return
        n_damper = int(self._damper_samples_var.get())
        track_name = self._track_var.get()
        per = 45 if track_name == "monza" else 1   # ~seconds per transient sample
        est_min = max(1, round(n_damper * per / 60))
        proceed = messagebox.askokcancel(
            "Optimize dampers — heads up",
            f"Joint damper optimisation builds a transient DoE of {n_damper} "
            f"7-DOF runs, then a Bayesian search over 8 variables "
            f"(springs, ARBs, and all four dampers).\n\n"
            f"On '{track_name}' the transient DoE alone can take roughly "
            f"{est_min} minute(s); a smaller 'Damper DoE' value is faster but "
            f"coarser.\n\n"
            f"The window stays responsive and progress is shown in the Log tab. "
            f"Start now?")
        if not proceed:
            return
        self._run_btn.config(state="disabled")
        self._export_btn.config(state="disabled")
        self._transient_btn.config(state="disabled")
        self._optimize_btn.config(state="disabled")
        self._log("─" * 60 + "\n")
        thread = threading.Thread(target=self._run_optimize_dampers, daemon=True)
        thread.start()

    def _run_optimize_dampers(self):
        import sys as _sys
        orig_stdout = _sys.stdout
        _sys.stdout = _QueueWriter(self._log_queue)
        try:
            self._optimize_dampers_body()
        except Exception as exc:
            self._log_queue.put(f"\n[ERROR] {exc}\n")
        finally:
            _sys.stdout = orig_stdout
            self.after(0, lambda: self._run_btn.config(state="normal"))
            self.after(0, lambda: self._transient_btn.config(state="normal"))
            self.after(0, lambda: self._optimize_btn.config(state="normal"))

    def _optimize_dampers_body(self):
        import optuna
        from dataclasses import replace as _replace
        from doe_sampler import run_lhs_sweep, run_transient_doe
        from surrogate import LapTimeSurrogate
        from optimizer import run_joint_optimization
        from track import Track
        from vehicle_model import NCMiata
        from transient_lap import run_transient_lap
        from analysis import transient_report_figure

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        track_name = self._track_var.get()
        n_samples  = int(self._samples_var.get())
        n_trials   = int(self._trials_var.get())
        n_damper   = int(self._damper_samples_var.get())
        base_cfg   = self._cfg
        track = Track.monza() if track_name == "monza" else Track()

        print(f"Optimize dampers (joint 8-variable)  |  {base_cfg.name}  |  {track.name}")
        print(f"[1/4] QSS DoE — {n_samples} LHS setups …")
        df = run_lhs_sweep(n_samples=n_samples, track=track, base_cfg=base_cfg)

        print("[2/4] Training QSS GP surrogate …")
        qss = LapTimeSurrogate()
        qss.fit(df)

        print(f"[3/4] Transient DoE — {n_damper} 7-DOF runs (the slow part) …")
        tdf = run_transient_doe(n_samples=n_damper, track=track, base_cfg=base_cfg)
        penalty = LapTimeSurrogate()
        penalty.fit(tdf, target_col="delta")

        print(f"[4/4] Joint Bayesian optimisation — {n_trials} trials, 8 variables …")
        _, buildable = run_joint_optimization(qss, penalty, n_trials=n_trials,
                                              base_cfg=base_cfg, track=track,
                                              show_progress_bar=False)

        opt_cfg = _replace(base_cfg, **buildable)
        self._export_cfg = opt_cfg
        self.after(0, lambda: self._load_cfg_into_fields(opt_cfg))

        tr = run_transient_lap(NCMiata(opt_cfg), track)
        fig = transient_report_figure(
            tr, title=f"{track.name} — optimised dampers ({base_cfg.name})")
        self.after(0, lambda: self._show_transient_figure(fig))
        self.after(0, lambda: self._export_btn.config(state="normal"))
        print("Done!  Optimised damper values are loaded into the form; "
              "use 'Export to Assetto Corsa…' to save them.\n")

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
            self.after(0, lambda: self._optimize_btn.config(state="normal"))

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
                              gforce_figure, corner_speed_figure,
                              build_lap_channels, lap_summary,
                              gg_diagram_figure, channels_figure)

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

        # Channel data for the optimised setup
        opt_car = NCMiata(opt_cfg)
        ch_df = build_lap_channels(opt_res, opt_car)
        smry  = lap_summary(ch_df)
        print(f"  v_max {smry['v_max_kph']:.1f} km/h  "
              f"| max lat g {smry['max_lat_g']:.2f}  "
              f"| max power {smry['max_power_kW']:.1f} kW  "
              f"| gears used {smry['gears_used']}")

        fig_cmp    = compare_figure(base_res, opt_res)
        fig_bar    = laptime_bar_figure([base_res, opt_res])
        fig_g      = gforce_figure(base_res, opt_res)
        fig_corner = corner_speed_figure(base_res, opt_res)
        fig_gg     = gg_diagram_figure(ch_df, label=f"{opt_cfg.name} — {track.name}")
        fig_ch     = channels_figure(ch_df, label=f"{opt_cfg.name} — {track.name}")

        self._channels_df = ch_df
        self.after(0, lambda: self._show_figures(
            fig_cmp, fig_bar, fig_g, fig_corner, fig_gg, fig_ch))
        self.after(0, lambda: self._export_btn.config(state="normal"))
        self.after(0, lambda: self._export_data_btn.config(state="normal"))
        print("Done!\n")

    # ------------------------------------------------------------------ #
    # Figure embedding
    # ------------------------------------------------------------------ #

    def _show_figures(self, fig_cmp, fig_bar, fig_g, fig_corner,
                      fig_gg=None, fig_ch=None):
        for tab, fig in ((self._tab_speed, fig_cmp),
                         (self._tab_bar, fig_bar),
                         (self._tab_gforce, fig_g),
                         (self._tab_corner, fig_corner)):
            self._embed_figure(tab, fig)
        if fig_gg is not None:
            self._embed_figure(self._tab_gg, fig_gg)
        if fig_ch is not None:
            self._embed_figure(self._tab_channels, fig_ch)

        # Refresh the data-explorer selectors and draw their default views
        self._populate_channel_selectors()
        try:
            self._on_draw_trackmap()
            self._on_draw_customchart()
        except Exception:
            pass

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

def _enable_high_dpi():
    """On Windows, make the process DPI-aware so the GUI is crisp and sized
    correctly on high-resolution / scaled displays. No-op elsewhere."""
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor v2
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()       # older Windows
    except Exception:
        pass


def main():
    _enable_high_dpi()
    app = PipelineGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

