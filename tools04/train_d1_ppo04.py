#!/usr/bin/env python3
"""Contract 04 training entry: the frozen 03 trainer with factory04 injected.

tools/train_d1_ppo.py is a frozen input of the 03d release, so it is not edited
and not copied. It is executed here, unmodified, as __main__ via runpy after
d1_rl.factory03's two lookup functions are replaced by the factory04 versions.
Every hyperparameter, budget cap, counter, source snapshot and receipt therefore
comes from the frozen trainer.

Why injection instead of a new trainer: SubprocVecEnv uses the forkserver start
method, and cloudpickle serialises the env factory's module-level references by
name. Running the frozen file as __main__ keeps `make_env` a __main__ closure
(serialised by value) while `environment_class` resolves to d1_rl.factory04,
which the workers import by reference - so the children build the 04a task, not
the 03d one. Verify this with tools04/check_task04a_static.py, which exercises
the same round-trip without integrating any physics.
"""
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
FROZEN_TRAINER = ROOT/'tools/train_d1_ppo.py'


def main(argv):
    if not FROZEN_TRAINER.is_file():
        raise SystemExit(f'Frozen trainer missing: {FROZEN_TRAINER}')
    release = json.loads((ROOT/'evidence/career_contract_03/pilot03d_release_01.json').read_text())
    expected = release['inputs_sha256']['tools/train_d1_ppo.py']
    actual = hashlib.sha256(FROZEN_TRAINER.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Frozen trainer SHA256 changed: {actual} != {expected}')

    sys.path.insert(0, str(ROOT/'training'))
    import d1_rl.factory03 as factory03
    import d1_rl.factory04 as factory04
    factory03.environment_class = factory04.environment_class
    factory03.environment_sources = factory04.environment_sources

    sys.argv = [str(FROZEN_TRAINER), *argv]
    runpy.run_path(str(FROZEN_TRAINER), run_name='__main__')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
