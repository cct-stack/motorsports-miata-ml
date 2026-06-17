# PyInstaller build spec — Motorsport Pipeline v2 (damper / transient edition).
#
# Same GUI app as v1 (entry point gui.py) plus the 7-DOF transient damper
# analysis: the "Damper analysis" tab scores how much the current damper
# settings cost in lap time through load-transfer transients.
#
# Build (on the OS you want the binary for — PyInstaller does NOT cross-compile,
# so build the Windows .exe on Windows):
#
#     pip install pyinstaller
#     pyinstaller motorsports_pipeline_v2.spec
#
# Result lands in dist/motorsports_pipeline_v2.exe  (Windows) or
# dist/motorsports_pipeline_v2  (Linux/macOS).

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# --- bundle data files (track + default car template) ---------------------- #
datas = [
    ("Autodromo Nazionale Monza.xlsx", "."),
    ("NC_Miata_default.json", "."),
]
binaries = []
hiddenimports = []

# Scientific / ML packages — pull everything so no hidden data files are missed
for pkg in ("sklearn", "scipy", "optuna", "pandas"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas     += pkg_datas
    binaries  += pkg_binaries
    hiddenimports += pkg_hidden

hiddenimports += collect_submodules("matplotlib")
hiddenimports += [
    "matplotlib.backends.backend_tkagg",
    "tkinter",
    "tkinter.ttk",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "tkinter.scrolledtext",
    # v2: transient damper model (imported lazily by gui.py's Damper analysis)
    "transient_lap",
    "transient_model",
]


a = Analysis(
    ["gui.py"],          # <-- GUI entry point (same as v1)
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter.test", "pytest"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="motorsports_pipeline_v2",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,         # windowless — all output goes to the GUI log box
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
