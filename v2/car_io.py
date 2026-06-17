"""car_io.py — save / load a VehicleConfig as a JSON file.

The JSON format is a direct serialisation of VehicleConfig fields plus the
nested PacejkaTire (which already has to_dict / from_dict).  Every field has a
round-trip guarantee: load_json(save_json(cfg)) == cfg.

Usage
-----
    from car_io import load_json, save_json
    cfg = load_json("my_car.json")   # -> VehicleConfig
    save_json(cfg, "my_car.json")    # writes / overwrites
"""
from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Union

from config import VehicleConfig
from tire_model import PacejkaTire


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_json(cfg: VehicleConfig, path: Union[str, Path]) -> None:
    """Serialise *cfg* to a UTF-8 JSON file at *path*."""
    d = _cfg_to_dict(cfg)
    Path(path).write_text(json.dumps(d, indent=2), encoding="utf-8")


def load_json(path: Union[str, Path]) -> VehicleConfig:
    """Deserialise a JSON file written by :func:`save_json` into a
    :class:`~config.VehicleConfig`.  Unknown keys are silently ignored so
    future fields won't break older files."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return _dict_to_cfg(raw)


def default_nc_miata_json(path: Union[str, Path]) -> None:
    """Write the built-in NC Miata defaults to *path* (useful as a template)."""
    save_json(VehicleConfig(), path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cfg_to_dict(cfg: VehicleConfig) -> dict:
    """Convert a VehicleConfig to a plain JSON-serialisable dict."""
    d: dict = {}
    for f in fields(cfg):
        val = getattr(cfg, f.name)
        if isinstance(val, PacejkaTire):
            d[f.name] = val.to_dict()
        elif isinstance(val, list) and val and isinstance(val[0], (list, tuple)):
            # torque_curve: list of (rpm, Nm) tuples -> list of [rpm, Nm]
            d[f.name] = [list(item) for item in val]
        else:
            d[f.name] = val
    return d


def _dict_to_cfg(raw: dict) -> VehicleConfig:
    """Convert a plain dict (from JSON) back to a VehicleConfig."""
    known = {f.name: f for f in fields(VehicleConfig)}
    kwargs: dict = {}

    for key, val in raw.items():
        if key not in known:
            continue  # forward-compatibility: skip unknown keys
        f = known[key]
        if f.name == "tire":
            kwargs["tire"] = PacejkaTire.from_dict(val)
        elif f.name == "torque_curve":
            # Accept both [[rpm, nm], ...] and [[rpm, nm], ...]
            kwargs["torque_curve"] = [tuple(item) for item in val]
        elif f.name == "gear_ratios":
            kwargs["gear_ratios"] = list(val)
        else:
            kwargs[key] = val

    return VehicleConfig(**kwargs)
