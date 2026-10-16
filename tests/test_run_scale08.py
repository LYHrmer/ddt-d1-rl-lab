"""Tests for tools08/run_scale08.py, the contract 08 launcher and equivalence watch.

The launcher decides whether the scale run is the same experiment as the frozen 03d
run for its first million steps, and interrupts it when it is not. So the comparison
rules are pinned exactly, and the interrupt path is exercised end to end with a fake
trainer that diverges on purpose. Nothing here builds an environment or runs physics.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools08'))

import run_scale08 as launcher  # noqa: E402


def row(k, **overrides):
    base = {'rollout': k, 'transitions': 4096*k, 'elapsed_training_s': 1.5*k,
            'actual_counts': {'native_steps': 40960*k, 'resets': 90*k},
            'reward_contribution_mean_per_tick': {'vx_tracking': .014*k, 'alive': .005}}
    base.update(overrides)
    return base


# --- comparison rules -------------------------------------------------------------
def test_identical_rollouts_have_no_differences():
    assert launcher.compare_rollout(row(3), row(3)) == []


def test_timing_and_memory_fields_are_ignored():
    assert launcher.compare_rollout(row(3, elapsed_training_s=99., process_tree_rss_bytes=1), row(3)) == []


def test_a_reward_term_off_by_more_than_the_tolerance_is_a_difference():
    bad = row(3); bad['reward_contribution_mean_per_tick']['vx_tracking'] *= 1+1e-8
    assert any('vx_tracking' in p for p in launcher.compare_rollout(bad, row(3)))


def test_a_reward_term_within_the_tolerance_is_not():
    ok = row(3); ok['reward_contribution_mean_per_tick']['vx_tracking'] *= 1+1e-10
    assert launcher.compare_rollout(ok, row(3)) == []


@pytest.mark.parametrize('key, value', [('rollout', 4), ('transitions', 4097)])
def test_index_and_transition_count_must_match_exactly(key, value):
    assert launcher.compare_rollout(row(3, **{key: value}), row(3))


def test_counters_must_match_exactly():
    assert launcher.compare_rollout(row(3, actual_counts={'native_steps': 1, 'resets': 270}), row(3))


def test_a_missing_or_extra_reward_term_is_a_difference():
    extra = row(3); extra['reward_contribution_mean_per_tick']['new'] = 0.
    assert any('reward terms differ' in p for p in launcher.compare_rollout(extra, row(3)))


def test_close_handles_zero_and_non_finite_values():
    assert launcher.close(0., 0.) and not launcher.close(0., 1e-300)
    assert not launcher.close(float('nan'), float('nan'))
    assert launcher.close(1., 1.+5e-10) and not launcher.close(1., 1.+2e-9)


# --- reading the ledgers ------------------------------------------------------------
def test_a_half_written_last_line_is_not_read():
    assert launcher.complete_lines('{"a": 1}\n{"a": 2}\n{"a"') == ['{"a": 1}', '{"a": 2}']
    assert launcher.complete_lines('') == []


def test_reading_the_frozen_ledger_requires_enough_rollouts(tmp_path):
    path = tmp_path/'ledger.jsonl'; path.write_text(''.join(json.dumps(row(k))+'\n' for k in (1, 2, 3)))
    assert [r['rollout'] for r in launcher.read_frozen(path, 2)] == [1, 2]
    with pytest.raises(SystemExit, match='fewer than'):
        launcher.read_frozen(path, 4)


# --- what gets launched -------------------------------------------------------------
def test_the_command_is_fixed_by_the_kind_of_run():
    cmd = launcher.command('scale', Path('/out'), 'python3')
    assert cmd[cmd.index('--steps')+1] == '8388608' and cmd[cmd.index('--kind')+1] == 'scale'
    assert cmd[cmd.index('--seed')+1] == '101' and cmd[cmd.index('--envs')+1] == '8'
    assert cmd[cmd.index('--max-wall')+1] == '9000' and cmd[2].endswith('tools08/train_d1_ppo08.py')
    pre = launcher.command('preflight', Path('/out'), 'python3')
    assert pre[pre.index('--steps')+1] == '32768' and pre[pre.index('--max-wall')+1] == '600'


def test_the_plan_matches_the_contract():
    assert launcher.PLANS['scale']['steps'] == 8388608 and launcher.PLANS['scale']['compare_rollouts'] == 256
    assert launcher.PLANS['preflight']['steps'] == 32768 and launcher.PLANS['preflight']['compare_rollouts'] == 8
    assert launcher.PLANS['scale']['steps'] // 4096 >= launcher.PLANS['scale']['compare_rollouts']


def test_the_trainer_environment_pins_threads_and_import_path():
    env = launcher.trainer_env({'PATH': '/bin', 'OMP_NUM_THREADS': '8'})
    assert env['OMP_NUM_THREADS'] == '1' and env['OPENBLAS_NUM_THREADS'] == '1' and env['PATH'] == '/bin'
    assert env['PYTHONPATH'].split(':') == [str(ROOT/'.python-deps'), str(ROOT/'training')]


# --- order gates -------------------------------------------------------------------
def test_the_scale_run_needs_a_passed_preflight(tmp_path):
    path = tmp_path/'r.json'
    for bad in ({'kind': 'preflight', 'equivalence': 'FAIL'}, {'kind': 'scale', 'equivalence': 'PASS'}, {}):
        path.write_text(json.dumps(bad))
        with pytest.raises(SystemExit):
            launcher.require_preflight_passed(path)
    path.write_text(json.dumps({'kind': 'preflight', 'equivalence': 'PASS'}))
    launcher.require_preflight_passed(path)


def test_the_release_must_match_inputs_and_this_launcher(tmp_path):
    (tmp_path/'f').write_bytes(b'x')
    receipt = tmp_path/'release.json'
    receipt.write_text(json.dumps({'frozen_inputs': {'f': launcher.sha(tmp_path/'f')},
                                   'launcher_sha256': launcher.sha(launcher.__file__)}))
    launcher.verify_release(receipt, root=tmp_path)
    (tmp_path/'f').write_bytes(b'changed')
    with pytest.raises(SystemExit, match='changed'):
        launcher.verify_release(receipt, root=tmp_path)
    (tmp_path/'f').write_bytes(b'x')
    receipt.write_text(json.dumps({'frozen_inputs': {}, 'launcher_sha256': '0'*64}))
    with pytest.raises(SystemExit, match='launcher differs'):
        launcher.verify_release(receipt, root=tmp_path)


# --- the watch, end to end with a fake trainer --------------------------------------
FAKE = '''
import json, signal, sys, time
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--output")+1]); out.mkdir(parents=True)
rows = json.loads(Path(sys.argv[-1]).read_text())
def stop(*_):
    (out/"result.json").write_text(json.dumps({"status": "ERROR", "interrupted": True})); sys.exit(1)
signal.signal(signal.SIGINT, stop)
with (out/"ledger.jsonl").open("a") as f:
    for r in rows:
        f.write(json.dumps(r)+"\\n"); f.flush(); time.sleep(0.05)
if "--hang" in sys.argv:
    time.sleep(30)
(out/"result.json").write_text(json.dumps({"status": "COMPLETE"}))
'''


def fake_setup(tmp_path, monkeypatch, written, frozen, hang=False):
    fake = tmp_path/'fake.py'; fake.write_text(FAKE)
    rows_file = tmp_path/'rows.json'; rows_file.write_text(json.dumps(written))
    ledger = tmp_path/'frozen.jsonl'; ledger.write_text(''.join(json.dumps(r)+'\n' for r in frozen))
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    monkeypatch.setattr(launcher, 'FROZEN_LEDGER', 'frozen.jsonl')
    monkeypatch.setattr(launcher, 'PLANS', {'preflight': {'steps': 32768, 'max_wall': 600, 'compare_rollouts': len(frozen),
                                                          'output': 'out/run'}})
    monkeypatch.setattr(launcher, 'verify_release', lambda *a, **k: {})
    extra = ['--hang'] if hang else []
    monkeypatch.setattr(launcher, 'command',
                        lambda kind, output, python=sys.executable: [sys.executable, str(fake), '--output', str(output), *extra, str(rows_file)])
    (tmp_path/'release.json').write_text('{}')
    return tmp_path/'out/run_launcher/report.json'


def test_a_run_that_matches_the_frozen_ledger_passes(tmp_path, monkeypatch):
    frozen = [row(k) for k in (1, 2, 3)]
    report_path = fake_setup(tmp_path, monkeypatch, frozen, frozen)
    assert launcher.run('preflight', tmp_path/'release.json', poll_s=.02) == 0
    report = json.loads(report_path.read_text())
    assert report['equivalence'] == 'PASS' and report['rollouts_compared'] == 3
    assert report['interrupted_by_launcher'] is False


def test_a_run_that_diverges_is_interrupted_and_recorded_as_failed(tmp_path, monkeypatch):
    frozen = [row(k) for k in (1, 2, 3, 4)]
    written = [row(1), row(2), row(3, transitions=1), row(4)]
    report_path = fake_setup(tmp_path, monkeypatch, written, frozen, hang=True)
    assert launcher.run('preflight', tmp_path/'release.json', poll_s=.02) == 1
    report = json.loads(report_path.read_text())
    assert report['equivalence'] == 'FAIL' and report['first_mismatch']['rollout'] == 3
    assert report['interrupted_by_launcher'] is True and report['rollouts_compared'] == 2
    assert report['trainer_result']['interrupted'] is True and report['wall_s'] < 20


def test_a_run_that_ends_before_enough_rollouts_is_incomplete_not_passed(tmp_path, monkeypatch):
    frozen = [row(k) for k in (1, 2, 3)]
    report_path = fake_setup(tmp_path, monkeypatch, frozen[:2], frozen)
    assert launcher.run('preflight', tmp_path/'release.json', poll_s=.02) == 1
    assert json.loads(report_path.read_text())['equivalence'] == 'INCOMPLETE'


def test_rollouts_written_after_the_last_poll_are_still_compared(tmp_path, monkeypatch):
    """The real preflight lasts about 25 s and its last rollouts land after the final poll."""
    frozen = [row(k) for k in (1, 2, 3)]
    report_path = fake_setup(tmp_path, monkeypatch, frozen, frozen)
    assert launcher.run('preflight', tmp_path/'release.json', poll_s=3.) == 0
    report = json.loads(report_path.read_text())
    assert report['equivalence'] == 'PASS' and report['rollouts_compared'] == 3


def test_a_late_mismatch_after_the_last_poll_is_still_caught(tmp_path, monkeypatch):
    frozen = [row(k) for k in (1, 2, 3)]
    report_path = fake_setup(tmp_path, monkeypatch, [row(1), row(2), row(3, transitions=7)], frozen)
    assert launcher.run('preflight', tmp_path/'release.json', poll_s=3.) == 1
    assert json.loads(report_path.read_text())['first_mismatch']['rollout'] == 3


def test_the_scale_run_refuses_to_start_without_a_passed_preflight(tmp_path, monkeypatch):
    fake_setup(tmp_path, monkeypatch, [row(1)], [row(1)])
    monkeypatch.setattr(launcher, 'PLANS', {'scale': {'steps': 8388608, 'max_wall': 9000,
                                                      'compare_rollouts': 1, 'output': 'out/scale'}})
    bad = tmp_path/'pre.json'; bad.write_text(json.dumps({'kind': 'preflight', 'equivalence': 'FAIL'}))
    with pytest.raises(SystemExit, match='preflight'):
        launcher.run('scale', tmp_path/'release.json', preflight_receipt=bad, poll_s=.02)
    assert not (tmp_path/'out/scale').exists() and not (tmp_path/'out/scale_launcher').exists()


def test_an_existing_output_is_never_reused(tmp_path, monkeypatch):
    fake_setup(tmp_path, monkeypatch, [row(1)], [row(1)])
    (tmp_path/'out/run').mkdir(parents=True)
    with pytest.raises(SystemExit, match='already exists'):
        launcher.run('preflight', tmp_path/'release.json', poll_s=.02)


# --- the new entry differs from the frozen trainer in exactly three lines -----------
def test_the_new_entry_is_the_frozen_trainer_with_three_lines_changed():
    sys.path.insert(0, str(ROOT/'tools08'))
    import check_scale08_static as static
    changes = static.line_changes((ROOT/'tools/train_d1_ppo.py').read_text(), (ROOT/launcher.ENTRY).read_text())
    assert changes == static.EXPECTED_CHANGES


def test_line_changes_rejects_any_change_in_length():
    import check_scale08_static as static
    assert static.line_changes('a\nb', 'a\nb\nc') is None
    assert static.line_changes('a\nb', 'a\nc') == {1: ('b', 'c')}
