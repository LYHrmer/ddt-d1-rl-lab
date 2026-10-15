"""Static half of the zero-model, zero-physics guard.

tests/conftest.py replaces the MuJoCo entry points at runtime, but a runtime
patch has a measured blind spot: a name bound earlier with
`from mujoco import mj_step` keeps pointing at the original function. This
module closes that with an AST scan that does not depend on import order.

It imports only the list of names (tests/_forbidden.py), never conftest, so it
still runs, and still means something, if the runtime layer is ever removed.
test_runtime_guard.py exercises the runtime layer separately.

Scope of the scan: tests/ and the modules those tests import for logic
(training/d1_rl/interface.py and the two reward modules). The full static checks
under tools04/ build environments on purpose and are not scanned; they run on
the workstation, not here.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from _forbidden import MODEL_CONSTRUCTORS, PHYSICS_FUNCTIONS

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = set(PHYSICS_FUNCTIONS) | set(MODEL_CONSTRUCTORS)
# The scan is for tests and the logic modules they import; this file and the
# guard define the forbidden names, so naming them here is not a violation.
SCANNED = sorted(
    p for p in [*(ROOT/'tests').glob('*.py'),
                ROOT/'training/d1_rl/interface.py',
                ROOT/'training/d1_rl/task04a.py',
                ROOT/'training/d1_rl/task04b.py',
                ROOT/'tools05/release_probe05.py']
    if p.name not in {'conftest.py', '_forbidden.py', 'test_static_guard.py',
                      'test_runtime_guard.py'})


def violations(source: str) -> list[str]:
    """Names from the forbidden set that the code imports or calls."""
    tree = ast.parse(source)
    mujoco_aliases = set()
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'mujoco' or alias.name.startswith('mujoco.'):
                    mujoco_aliases.add((alias.asname or alias.name).split('.')[0])
        elif isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] == 'mujoco':
            for alias in node.names:
                if alias.name in FORBIDDEN or alias.name == 'MjModel':
                    found.append(f'from {node.module} import {alias.name} (line {node.lineno})')
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN:
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            chain_is_mujoco = isinstance(root, ast.Name) and root.id in mujoco_aliases
            # MjModel.from_xml_path(...) reached through a name bound from mujoco
            via_class = (isinstance(node.value, ast.Name) and node.value.id == 'MjModel')
            if chain_is_mujoco or via_class:
                found.append(f'{node.attr} (line {node.lineno})')
    return found


# --- the scan itself ---------------------------------------------------------

@pytest.mark.parametrize('path', SCANNED, ids=lambda p: p.name)
def test_no_scanned_file_builds_a_model_or_advances_physics(path):
    assert violations(path.read_text()) == [], path.name


def test_the_scan_covers_the_files_it_claims_to():
    names = {p.name for p in SCANNED}
    assert {'test_interface.py', 'test_contract04_terms.py', 'test_handoff_integrity.py',
            'test_cleanup_receipts.py', 'interface.py', 'task04a.py', 'task04b.py',
            'release_probe05.py'} <= names


# --- the scanner must catch each evasion, or its silence means nothing --------

@pytest.mark.parametrize('source', [
    'import mujoco\nmujoco.mj_step(m, d)\n',
    'import mujoco as mj\nmj.mj_forward(m, d)\n',
    'from mujoco import mj_step\nmj_step(m, d)\n',
    'from mujoco import mj_resetData as reset\n',
    'import mujoco\nmujoco.MjModel.from_xml_path("a.xml")\n',
    'import mujoco as mj\nmj.MjModel.from_xml_string("<mujoco/>")\n',
    'from mujoco import MjModel\nMjModel.from_binary_path("a.mjb")\n',
    'import mujoco\nstep = mujoco.mj_step\n',
])
def test_scanner_catches_each_way_of_reaching_the_simulator(source):
    assert violations(source), source


@pytest.mark.parametrize('source', [
    '# mj_step is mentioned only in a comment\nx = 1\n',
    'text = "mujoco.mj_step(m, d)"\n',
    'import mujoco\nprint(mujoco.mj_versionString())\n',
    'import numpy as np\nnp.mj_step = 1\n',
    'def mj_step():\n    pass\nmj_step()\n',
])
def test_scanner_does_not_flag_harmless_mentions(source):
    assert violations(source) == [], source
