"""Tests for the pure logic of tools05/probe_stop_factorisation05.py.

The probe replays a frozen course with the wheel actions forced to zero after the
cut. The parts that decide whether its result can be trusted are all pure: what
is overridden and from which tick, whether the replay matches the frozen trace,
where the verdict thresholds fall, and which receipts must exist before a later
run may start. Nothing here loads a model or runs physics.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from _workstation import needs_workstation_evidence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools05'))

import probe_stop_factorisation05 as probe  # noqa: E402

WHEELS = list(probe.WHEEL_ACTION_INDEX)
LEGS = [i for i in range(16) if i not in WHEELS]


# --- the override -------------------------------------------------------------
def test_wheels_are_untouched_before_the_cut_tick():
    action = np.linspace(-.8, .8, 16, dtype=np.float32)
    np.testing.assert_array_equal(probe.force_wheels_zero(action, 399, 400, True), action)


def test_wheels_are_zero_from_the_cut_tick_on_and_legs_are_not():
    action = np.linspace(-.8, .8, 16, dtype=np.float32)
    for tick in (400, 401, 599):
        out = probe.force_wheels_zero(action, tick, 400, True)
        assert np.all(out[WHEELS] == 0.)
        np.testing.assert_array_equal(out[LEGS], action[LEGS])


def test_only_the_four_wheel_indices_are_zeroed():
    out = probe.force_wheels_zero(np.ones(16, dtype=np.float32), 500, 400, True)
    assert np.flatnonzero(out == 0.).tolist() == [3, 7, 11, 15]


def test_disabled_override_leaves_the_policy_arm_alone():
    action = np.linspace(-.8, .8, 16, dtype=np.float32)
    np.testing.assert_array_equal(probe.force_wheels_zero(action, 500, 400, False), action)


def test_override_returns_a_copy_and_keeps_the_dtype():
    action = np.ones(16, dtype=np.float32)
    out = probe.force_wheels_zero(action, 500, 400, True)
    assert out is not action and np.all(action == 1.)
    assert out.dtype == np.float32


# --- the command schedule -----------------------------------------------------
def test_command_switches_exactly_at_the_segment_start():
    sequence = probe.build_sequence(.2)
    assert probe.command_at_tick(sequence, 99) == (0., 0.)
    assert probe.command_at_tick(sequence, 100) == (.2, 0.)
    assert probe.command_at_tick(sequence, 399) == (.2, 0.)
    assert probe.command_at_tick(sequence, 400) == (0., 0.)


def test_cut_tick_is_derived_from_the_sequence():
    assert probe.cut_tick_of(probe.build_sequence(.2), 600) == 400
    assert probe.cut_tick_of(probe.build_sequence(.3), 600) == 400
    assert probe.cut_tick_of(probe.build_sequence(0.), 600) is None
    assert probe.cut_tick_of(probe.build_sequence(.2), 400) is None   # horizon ends before the cut


@needs_workstation_evidence
def test_cut_tick_matches_the_command_in_the_frozen_evaluation_trace():
    with np.load(ROOT/probe.FROZEN_COMPARISON) as frozen:
        moving = np.any(frozen['command_during_interval'] != 0., axis=1)
    first_zero = int(np.flatnonzero(moving[:-1] & ~moving[1:])[-1])+1
    assert first_zero == probe.CUT_TICK
    for tick in range(len(moving)):
        assert (probe.command_at_tick(probe.RUNS['A']['sequence'], tick)[0] != 0.) == moving[tick]


def test_declared_cut_ticks_agree_with_the_sequences():
    for name, spec in probe.RUNS.items():
        assert probe.cut_tick_of(spec['sequence'], spec['ticks']) == spec['cut_tick'], name


def test_sequences_stay_inside_the_frozen_command_limit():
    for spec in probe.RUNS.values():
        assert all(abs(seg['vx']) <= .3 and abs(seg['wz']) <= .3 for seg in spec['sequence'])
    probe.build_sequence(.3)
    with pytest.raises(ValueError):
        probe.build_sequence(.3000001)
    with pytest.raises(ValueError):
        probe.build_sequence(-.31)


# --- the budget ---------------------------------------------------------------
def test_planned_native_steps_and_the_pool_cap():
    assert {k: probe.planned_native(k) for k in probe.RUNS} == {
        'smoke': 20, 'A': 6000, 'B': 6000, 'C': 6000}
    assert sum(probe.planned_native(k) for k in probe.RUNS) == probe.POOL_CAP_NATIVE == 18020


def test_only_run_a_and_b_force_the_wheels_and_c_is_the_matched_control():
    assert [probe.RUNS[k]['zero_wheels'] for k in ('A', 'B', 'C')] == [True, True, False]
    assert probe.RUNS['B']['sequence'] == probe.RUNS['C']['sequence']


# --- fidelity (P1) ------------------------------------------------------------
def trace(n=600, offset=0.):
    t = np.arange(n, dtype=float)
    return {'body_vx_m_s': .1*np.sin(t/50.)+offset, 'base_x_m': .01*t+offset,
            'base_y_m': -.002*t+offset}


def test_identical_replay_passes_the_fidelity_gate():
    out = probe.fidelity(trace(), trace())
    assert out['passed'] and out['ticks_compared'] == 400


@pytest.mark.parametrize('key', ['body_vx_m_s', 'base_x_m', 'base_y_m'])
def test_a_difference_in_any_compared_channel_fails_the_gate(key):
    changed = trace(); changed[key][399] += 2e-6
    assert not probe.fidelity(changed, trace())['passed']


def test_a_difference_at_the_tolerance_passes_and_just_above_fails():
    ok, bad = trace(), trace()
    ok['base_x_m'][10] += probe.P1_TOLERANCE*.5
    bad['base_x_m'][10] += probe.P1_TOLERANCE*2.
    assert probe.fidelity(ok, trace())['passed']
    assert not probe.fidelity(bad, trace())['passed']


def test_a_difference_exactly_at_the_tolerance_passes_and_just_above_fails():
    """0.5 and 0.25 are exact in binary, so the boundary is hit without rounding noise."""
    base = {k: np.zeros(6) for k in ('body_vx_m_s', 'base_x_m', 'base_y_m')}
    at = {**base, 'base_x_m': np.array([0., 0., .5, 0., 0., 0.])}
    above = {**base, 'base_x_m': np.array([0., 0., .5000001, 0., 0., 0.])}
    assert probe.fidelity(at, base, cut_tick=4, tolerance=.5)['passed']
    assert not probe.fidelity(above, base, cut_tick=4, tolerance=.5)['passed']


def test_differences_after_the_cut_do_not_count():
    changed = trace(); changed['body_vx_m_s'][400:] += 1.
    assert probe.fidelity(changed, trace())['passed']


def test_the_last_pre_cut_tick_is_compared_and_the_cut_tick_is_not():
    last, first_after = trace(), trace()
    last['body_vx_m_s'][399] += 1.
    first_after['body_vx_m_s'][400] += 1.
    assert not probe.fidelity(last, trace())['passed']
    assert probe.fidelity(first_after, trace())['passed']


def test_a_trace_shorter_than_the_cut_fails_instead_of_passing_vacuously():
    assert not probe.fidelity(trace(300), trace())['passed']
    assert not probe.fidelity(trace(), trace(300))['passed']


# --- the thresholds (P2) ------------------------------------------------------
@pytest.mark.parametrize('tau, expected', [
    (.113, 'SUPPORTED'), (.29, 'SUPPORTED'), (.2900001, 'UNCERTAIN'), (.35, 'UNCERTAIN'),
    (.4099999, 'UNCERTAIN'), (.41, 'REFUTED'), (.58, 'REFUTED'), (None, 'UNMEASURABLE')])
def test_p2_classification_at_the_frozen_boundaries(tau, expected):
    assert probe.classify_p2(tau) == expected


def test_the_policy_arm_tau_is_in_the_refuted_region_by_construction():
    assert probe.classify_p2(probe.TAU_POLICY_ARM_S) == 'REFUTED'


@needs_workstation_evidence
def test_baseline_tau_equals_the_analysis_receipt():
    report = json.loads((ROOT/'evidence/career_contract_04/cut_to_zero_analysis_04/report.json').read_text())
    recorded = report['versions']['04b']['forward_stop']['decay_time_constant_s']['ticks_10_to_50']
    assert probe.TAU_POLICY_ARM_S == pytest.approx(recorded, abs=1e-12)


# --- judge --------------------------------------------------------------------
def analysis(tau=.2, euclid=.03, ratio=1., available=200):
    return {'decay_time_constant_s': {'ticks_10_to_50': tau}, 'ticks_after_cut_available': available,
            'final_euclidean_displacement_m': euclid,
            'first_order_stop': {'predicted_over_path': ratio}}


def test_failed_fidelity_makes_everything_else_not_evaluable():
    out = probe.judge(analysis(), analysis(), analysis(), fidelity_passed=False)
    assert out == {'P1': 'NOT_MET', 'P2': 'NOT_EVALUABLE', 'P3': 'NOT_EVALUABLE',
                   'P4': 'NOT_EVALUABLE', 'P5': 'NOT_EVALUABLE'}


def test_p3_threshold_is_strict_below():
    assert probe.judge(analysis(euclid=.0499))['P3'] == 'MET'
    assert probe.judge(analysis(euclid=.05))['P3'] == 'NOT_MET'


def test_p3_is_not_met_when_the_run_ended_before_200_ticks_after_the_cut():
    assert probe.judge(analysis(euclid=.01, available=120))['P3'] == 'NOT_MET'


def test_p4_ratio_boundaries():
    for ratio, expected in ((.75, 'MET'), (1.25, 'MET'), (.7499, 'NOT_MET'), (1.2501, 'NOT_MET')):
        out = probe.judge(analysis(tau=.2), analysis(tau=.2*ratio))
        assert out['P4'] == expected, ratio


def test_p5_requires_both_runs_inside_the_range():
    assert probe.judge(analysis(ratio=.8), analysis(ratio=1.2))['P5'] == 'MET'
    assert probe.judge(analysis(ratio=.79), analysis(ratio=1.))['P5'] == 'NOT_MET'
    assert probe.judge(analysis(ratio=1.), analysis(ratio=1.21))['P5'] == 'NOT_MET'


def test_missing_runs_are_not_evaluable_rather_than_passing():
    out = probe.judge(analysis())
    assert out['P4'] == 'NOT_EVALUABLE' and out['P5'] == 'NOT_EVALUABLE'
    assert probe.judge(None)['P2'] == 'UNMEASURABLE' and probe.judge(None)['P3'] == 'NOT_EVALUABLE'


def test_interpretation_follows_the_pre_registered_table():
    table = {('SUPPORTED', 'MET'): 'enough to pass',
             ('SUPPORTED', 'NOT_MET'): 'not sufficient',
             ('REFUTED', 'MET'): 'not the main lever',
             ('REFUTED', 'NOT_MET'): 'not the main lever',
             ('UNCERTAIN', 'MET'): 'inconclusive',
             ('UNMEASURABLE', 'NOT_EVALUABLE'): 'inconclusive',
             ('SUPPORTED', 'NOT_EVALUABLE'): 'inconclusive'}
    for (p2, p3), fragment in table.items():
        assert fragment in probe.interpretation(p2, p3), (p2, p3)
    assert 'no basis for a training contract' in probe.interpretation('UNCERTAIN', 'MET')


# --- the output must be readable by the analyser unchanged --------------------
def synthetic_run(zero_wheels, ticks=600, cut=400):
    """A coast whose speed decays with a known time constant, laid out as the probe records it."""
    tau, v0, tick_s = .25, .15, .02
    t = np.arange(ticks)
    speed = np.where(t < cut, v0, v0*np.exp(-(t-cut)*tick_s/tau))
    x = np.concatenate([[0.], np.cumsum(speed*tick_s)])
    rng = np.random.default_rng(0)
    raw = rng.uniform(-.5, .5, (ticks, 16)).astype(np.float32)
    executed = np.array([probe.force_wheels_zero(raw[k], k, cut, zero_wheels) for k in range(ticks)])
    command = np.zeros((ticks, 3)); command[100:cut, 0] = .2
    trace = {'reward': np.zeros(ticks), 'command_during_interval': command,
             'body_vx_m_s': speed, 'body_vy_m_s': np.zeros(ticks), 'body_wz_rad_s': np.zeros(ticks),
             'base_x_m': x[1:], 'base_y_m': np.zeros(ticks), 'base_height_m': np.full(ticks, .4),
             'base_roll_rad': np.zeros(ticks), 'base_pitch_rad': np.zeros(ticks),
             'base_yaw_rad': np.zeros(ticks), 'tilt_rad': np.zeros(ticks),
             'fallen': np.zeros(ticks, bool), 'illegal_body_contact': np.zeros(ticks, bool),
             'tau_saturation_count': np.zeros(ticks), 'issued_action': executed,
             'applied_action': executed}
    return probe.assemble_trace(trace, raw, np.stack([x, np.zeros_like(x)], axis=1)), raw, tau, v0


def test_assembled_trace_has_every_key_the_analyser_reads():
    arrays, *_ = synthetic_run(True)
    assert set(arrays) >= {'command_during_interval', 'boundary_xy_m', 'issued_action',
                           'policy_interval_start_s', 'body_vx_m_s', 'body_vy_m_s',
                           'body_wz_rad_s', 'base_yaw_rad'}
    assert arrays['boundary_xy_m'].shape[0] == arrays['issued_action'].shape[0]+1


def test_issued_action_is_the_executed_action_not_the_policy_output():
    arrays, raw, *_ = synthetic_run(True)
    assert np.all(arrays['issued_action'][400:, probe.WHEEL_ACTION_INDEX] == 0.)
    assert np.any(arrays['policy_action_raw'][400:, probe.WHEEL_ACTION_INDEX] != 0.)
    np.testing.assert_array_equal(arrays['policy_action_raw'], raw)


def test_the_real_analyser_reads_the_assembled_trace_and_recovers_the_known_decay(tmp_path):
    import analyze_cut_to_zero04 as analyser
    arrays, _, tau, v0 = synthetic_run(True)
    path = tmp_path/'trace.npz'
    np.savez_compressed(path, **arrays)
    result = analyser.analyse_trace(path, None)
    assert result['cut_tick'] == probe.CUT_TICK
    assert result['translation_task'] is True
    assert result['entry_speed_m_s'] == pytest.approx(v0)
    assert result['decay_time_constant_s']['ticks_10_to_50'] == pytest.approx(tau, rel=1e-6)
    assert result['ticks_after_cut_available'] == 200
    assert result['wheel_action_mean_square_per_tick']['0-5'] == 0.


def test_the_policy_arm_keeps_its_wheel_actions_in_the_analysed_trace(tmp_path):
    import analyze_cut_to_zero04 as analyser
    arrays, *_ = synthetic_run(False)
    path = tmp_path/'trace.npz'
    np.savez_compressed(path, **arrays)
    assert analyser.analyse_trace(path, None)['wheel_action_mean_square_per_tick']['0-5'] > 0.


# --- order gates and frozen inputs -------------------------------------------
def write_report(tmp_path, **fields):
    path = tmp_path/'report.json'
    path.write_text(json.dumps(fields))
    return path


def test_a_needs_a_completed_smoke_receipt(tmp_path):
    with pytest.raises(SystemExit):
        probe.require_passed(tmp_path/'absent.json', 'A')
    with pytest.raises(SystemExit):
        probe.require_passed(write_report(tmp_path, run='smoke', status='CRASHED'), 'A')
    with pytest.raises(SystemExit):
        probe.require_passed(write_report(tmp_path, run='B', status='COMPLETE'), 'A')
    probe.require_passed(write_report(tmp_path, run='smoke', status='COMPLETE'), 'A')


@pytest.mark.parametrize('name', ['B', 'C'])
def test_b_and_c_need_a_run_a_receipt_whose_fidelity_passed(tmp_path, name):
    with pytest.raises(SystemExit):
        probe.require_passed(write_report(tmp_path, run='A', status='COMPLETE',
                                          fidelity={'passed': False}), name)
    with pytest.raises(SystemExit):
        probe.require_passed(write_report(tmp_path, run='A', status='COMPLETE'), name)
    with pytest.raises(SystemExit):
        probe.require_passed(write_report(tmp_path, run='smoke', status='COMPLETE',
                                          fidelity={'passed': True}), name)
    with pytest.raises(SystemExit):
        probe.require_passed(write_report(tmp_path, run='A', status='CRASHED',
                                          fidelity={'passed': True}), name)
    probe.require_passed(write_report(tmp_path, run='A', status='COMPLETE',
                                      fidelity={'passed': True}), name)


def test_frozen_input_checks(tmp_path):
    good = tmp_path/'input.bin'; good.write_bytes(b'abc')
    digest = probe.sha(good)
    probe.check_frozen_inputs({str(good): digest})
    with pytest.raises(ValueError, match='changed'):
        probe.check_frozen_inputs({str(good): '0'*64})
    with pytest.raises(ValueError, match='missing'):
        probe.check_frozen_inputs({str(tmp_path/'absent'): digest})


def release_fixture(tmp_path, tool_sha=None, drop=None):
    """A small table of real temporary files, so each test breaks exactly one thing."""
    files = {'one.bin': b'one', 'two.bin': b'two'}
    for name, data in files.items():
        (tmp_path/name).write_bytes(data)
    table = {name: probe.sha(tmp_path/name) for name in files}
    frozen = {k: v for k, v in table.items() if k != drop}
    receipt = tmp_path/'release.json'
    receipt.write_text(json.dumps({'frozen_inputs': frozen,
                                   'tool_sha256': tool_sha or probe.sha(probe.__file__)}))
    return receipt, table


def test_a_complete_matching_release_receipt_is_accepted(tmp_path):
    receipt, table = release_fixture(tmp_path)
    assert probe.verify_release(receipt, table, tmp_path)['tool_sha256'] == probe.sha(probe.__file__)


def test_release_receipt_from_a_different_tool_is_refused(tmp_path):
    receipt, table = release_fixture(tmp_path, tool_sha='0'*64)
    with pytest.raises(SystemExit, match='differs'):
        probe.verify_release(receipt, table, tmp_path)


def test_release_receipt_that_omits_a_frozen_input_is_refused(tmp_path):
    receipt, table = release_fixture(tmp_path, drop='two.bin')
    with pytest.raises(SystemExit, match='does not freeze'):
        probe.verify_release(receipt, table, tmp_path)


def test_release_receipt_is_refused_when_an_input_changed_after_freezing(tmp_path):
    receipt, table = release_fixture(tmp_path)
    (tmp_path/'two.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        probe.verify_release(receipt, table, tmp_path)


def test_release_receipt_is_refused_when_the_expected_digest_differs(tmp_path):
    receipt, table = release_fixture(tmp_path)
    with pytest.raises(SystemExit, match='does not freeze'):
        probe.verify_release(receipt, {**table, 'one.bin': '0'*64}, tmp_path)


@needs_workstation_evidence
def test_the_real_frozen_inputs_still_match_their_recorded_digests():
    probe.check_frozen_inputs(probe.FROZEN_INPUTS)


def test_the_output_directory_is_created_before_the_environment_is_built():
    source = Path(probe.__file__).read_text()
    body = source[source.index('def execute_run'):]
    assert body.index('output.mkdir(parents=True, exist_ok=False)') < body.index('environment_class(config)(')


def test_a_plan_over_budget_is_refused_before_anything_is_created(tmp_path):
    target = tmp_path/'out'
    with pytest.raises(SystemExit, match='exceeds'):
        probe.execute_run('A', tmp_path/'p.onnx', tmp_path/'c.json', 'd1_rl.factory04b',
                          target, native_budget=5999)
    assert not target.exists()


def test_an_existing_output_directory_is_never_reused():
    source = Path(probe.__file__).read_text()
    assert source.count('exist_ok=False') >= 2     # run output and verdict output
    assert 'exist_ok=True' not in source


# --- the override must be seen to reach the robot -----------------------------
def effect_arrays(applied_wheels_after_cut, wanted=.3, n=8, cut=4):
    applied = np.zeros((n, 16)); applied[cut:, probe.WHEEL_ACTION_INDEX] = applied_wheels_after_cut
    raw = np.zeros((n, 16)); raw[cut:, probe.WHEEL_ACTION_INDEX] = wanted
    return {'applied_action': applied, 'policy_action_raw': raw}


def test_effective_override_is_zero_applied_while_the_policy_wanted_otherwise():
    out = probe.override_effect(effect_arrays(0.), 4, True)
    assert out['effective'] and out['checked_ticks'] == 4


def test_override_is_not_effective_when_the_robot_still_got_a_wheel_command():
    assert not probe.override_effect(effect_arrays(.05), 4, True)['effective']


def test_override_is_not_effective_when_the_policy_wanted_nothing_anyway():
    """A forced zero on top of an already-zero action proves nothing, so it must not pass."""
    assert not probe.override_effect(effect_arrays(0., wanted=0.), 4, True)['effective']


def test_override_check_needs_a_tick_at_or_after_the_cut():
    assert not probe.override_effect(effect_arrays(0.), 8, True)['effective']


def test_policy_arm_is_effective_only_when_applied_matches_what_the_policy_wanted():
    assert probe.override_effect(effect_arrays(.3), 4, False)['effective']
    assert not probe.override_effect(effect_arrays(0.), 4, False)['effective']


def test_the_probe_records_issued_action_as_what_it_handed_to_the_environment():
    source = Path(probe.__file__).read_text()
    assert "item = {'reward': reward, 'issued_action': executed, **info}" in source
    assert "'issued_action', 'applied_action')" in source


# --- the frozen input table ---------------------------------------------------
def test_every_frozen_sha_appears_in_the_contract_text():
    text = (ROOT/'docs/career_execution_contract_05m.md').read_text()
    for path, digest in probe.FROZEN_INPUTS.items():
        assert digest in text, path


def test_contract_text_and_tool_agree_on_thresholds_and_budget():
    text = (ROOT/'docs/career_execution_contract_05m.md').read_text()
    for fragment in ('0.29 s', '0.41 s', '18,020', '0.05 m', '[0.75, 1.25]', '[0.8, 1.2]', '1e-6'):
        assert fragment in text, fragment
    assert (probe.P2_SUPPORTED_AT_OR_BELOW_S, probe.P2_REFUTED_AT_OR_ABOVE_S) == (.29, .41)
    assert (probe.P4_RATIO_RANGE, probe.P5_RATIO_RANGE) == ((.75, 1.25), (.8, 1.2))
    assert probe.P1_TOLERANCE == 1e-6 and probe.P3_DISPLACEMENT_BELOW_M == .05


def test_the_probe_is_not_in_the_static_scan_because_it_does_step_physics():
    """execute_run advances physics through env.step, which the name-based scan cannot see.

    Listing the probe there would suggest it has none. What protects the tests is the runtime
    guard, which replaces mj_step itself: a test that reached env.step would be refused.
    """
    from test_static_guard import SCANNED
    assert 'probe_stop_factorisation05.py' not in {p.name for p in SCANNED}
    assert 'env.step(' in Path(probe.__file__).read_text()
