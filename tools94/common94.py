"""Exclusive artifacts, seeds and source identities for the cold-start experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/cold_start94"
PLAN = ROOT / "docs/experiments94/cold_start.json"
SEEDS = (94011, 94023, 94037)
CALIBRATION = ((948101, 948102, 948103), (948201, 948202, 948203))
EVALUATION = (949001, 949002, 949003)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False, default=json_value)
        f.write("\n")


def arrays(path, values):
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **values)


def fit_reset(seed, r, j):
    return 944000 + 1000 * SEEDS.index(seed) + 100 * r + j


def tag(k):
    return f"{'m' if k < 0 else 'p'}{abs(k):02d}"


def seed_dir(seed):
    return BASE / f"seed{seed}_01"


def clocks():
    return dict(native_decisions=0, servo_updates=0, physics_substeps=0,
                world_updates=0, policy_updates=0, ppo_updates=0)


def source_files():
    paths = [PLAN, *sorted((ROOT / "tools94").glob("*.py"))]
    paths.extend(ROOT / path for path in (
        "training/configs/pilot03d.json", "training/d1_rl/env.py", "training/d1_rl/interface.py",
        "training/d1_rl/delay_env18.py", "tools87/terrain87.py", "tools89/course89.py",
        "tools91/model91.py", "tools52/adapter52.py", "tools52/dynamics52.py"))
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}
