"""Task selection for contract 04 configurations; 03 versions defer to factory03.

Written as a separate module so the frozen 03 factory, trainer, exporter and
evaluator stay byte-identical. tools04/train_d1_ppo04.py injects this module in
place of factory03 before running the frozen trainer.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import factory03

ROOT = Path(__file__).resolve().parents[2]

# Captured at import time. tools04/train_d1_ppo04.py rebinds the factory03
# attributes to the functions below, so looking them up lazily would make the
# 03 fallback call back into this module forever.
_FACTORY03_ENVIRONMENT_CLASS = factory03.environment_class
_FACTORY03_ENVIRONMENT_SOURCES = factory03.environment_sources


def _values(config):
    return config if isinstance(config, dict) else json.loads(Path(config).read_text())


def environment_class(config):
    if _values(config)["version"] == "d1-pure-ppo-pilot-04a":
        from .task04a import D1Task04aEnv
        return D1Task04aEnv
    return _FACTORY03_ENVIRONMENT_CLASS(config)


def environment_sources(config):
    """Every authored source needed to reproduce the selected environment."""
    if _values(config)["version"] != "d1-pure-ppo-pilot-04a":
        return _FACTORY03_ENVIRONMENT_SOURCES(config)
    # The 04a chain is the full 03d chain plus this module and the 04a task.
    base = _FACTORY03_ENVIRONMENT_SOURCES(json.loads(
        (ROOT / "training/configs/pilot03d.json").read_text()))
    return [*base, Path(__file__).resolve(), ROOT / "training/d1_rl/task04a.py",
            ROOT / "training/configs/pilot03d.json"]
