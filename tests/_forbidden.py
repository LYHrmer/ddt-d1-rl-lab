"""The list of MuJoCo entry points forbidden under tests/, with no side effects.

Kept apart from conftest.py on purpose. conftest.py installs the runtime patch
as an import side effect; the static scan only needs the names. If the scan
imported conftest it could not run, or would mean nothing, whenever the runtime
layer was removed, and the two layers are meant to cover for each other: the
runtime patch misses names bound before it was installed, and the scan misses
nothing that is reached through a name it can see.
"""
from __future__ import annotations

# Entry points that either build a model or move simulated time. Anything that
# reads model state without advancing it (mj_name2id and so on) is left alone.
MODEL_CONSTRUCTORS = ('from_xml_path', 'from_xml_string', 'from_binary_path')
PHYSICS_FUNCTIONS = ('mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_resetData',
                     'mj_forwardSkip', 'mj_inverse', 'mj_rnePostConstraint')
