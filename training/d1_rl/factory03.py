"""Explicit task selection; older configurations retain their original environment."""
from __future__ import annotations

import json
from pathlib import Path

from .env import D1Env

ROOT = Path(__file__).resolve().parents[2]


def environment_class(config):
    values = config if isinstance(config, dict) else json.loads(Path(config).read_text())
    version = values["version"]
    if version == "d1-pure-ppo-pilot-03c":
        from .task03c import D1Task03cEnv
        return D1Task03cEnv
    if version == "d1-pure-ppo-pilot-03d":
        from .task03d import D1Task03dEnv
        return D1Task03dEnv
    if version in ("d1-pure-ppo-pilot-03", "d1-pure-ppo-pilot-03b"):
        return D1Env
    raise ValueError(f"Unsupported D1 task configuration: {version}")


def environment_sources(config):
    """Every authored source needed to reproduce the selected environment."""
    paths = [Path(__file__).resolve(), ROOT / "training/d1_rl/__init__.py", ROOT / "training/d1_rl/env.py",
             ROOT / "training/d1_rl/interface.py"]
    if environment_class(config) is not D1Env:
        paths += [ROOT / "training/d1_rl/fast_checks.py",
                  ROOT / "training/d1_rl/task03c.py", ROOT / "training/configs/pilot03b.json"]
    if environment_class(config).__name__ == "D1Task03dEnv":
        paths += [ROOT / "training/d1_rl/task03d.py", ROOT / "training/configs/pilot03c.json"]
    return paths
