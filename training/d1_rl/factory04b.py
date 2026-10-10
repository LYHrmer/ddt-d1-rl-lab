"""Task selection for contract 04b; earlier versions defer down the chain.

Each contract adds one immutable link rather than editing the previous factory,
because factory04.py's SHA-256 is recorded in
evidence/career_contract_04/pilot04a_release_02.json and in pilot04a_02's
protocol. Editing it would invalidate 04a's recorded identity.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import factory04

ROOT = Path(__file__).resolve().parents[2]

# Captured at import time; the training entry rebinds factory03's attributes.
_UPSTREAM_ENVIRONMENT_CLASS = factory04.environment_class
_UPSTREAM_ENVIRONMENT_SOURCES = factory04.environment_sources


def _values(config):
    return config if isinstance(config, dict) else json.loads(Path(config).read_text())


def environment_class(config):
    if _values(config)["version"] == "d1-pure-ppo-pilot-04b":
        from .task04b import D1Task04bEnv
        return D1Task04bEnv
    return _UPSTREAM_ENVIRONMENT_CLASS(config)


def environment_sources(config):
    """Every authored source needed to reproduce the selected environment."""
    if _values(config)["version"] != "d1-pure-ppo-pilot-04b":
        return _UPSTREAM_ENVIRONMENT_SOURCES(config)
    # 04b is the 03d chain plus this module and the 04b task; it does not use 04a.
    base = _UPSTREAM_ENVIRONMENT_SOURCES(json.loads(
        (ROOT / "training/configs/pilot03d.json").read_text()))
    return [*base, Path(__file__).resolve(), ROOT / "training/d1_rl/task04b.py",
            ROOT / "training/configs/pilot03d.json"]
