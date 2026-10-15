"""Tests for tools05/compare_arms05.py, the receipt behind the 05M run A reading.

The comparison claims that the decay is not exponential and that the wheel action
explains a share of the displacement. Both claims rest on helpers that can be given
cases with known answers. Nothing here loads a model or runs physics.
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

import compare_arms05 as arms  # noqa: E402

RECEIPT = ROOT/'evidence/career_contract_05/probe05_arms_03/report.json'


def decay(tau, v0=.15, n=600, cut=400):
    t = np.arange(n)
    return np.where(t < cut, v0, v0*np.exp(-(t-cut)*.02/tau))


# --- speeds and time constants ---------------------------------------------------
def test_speed_is_the_horizontal_norm():
    s = arms.speed_series({'body_vx_m_s': [3., 0.], 'body_vy_m_s': [4., -2.]})
    np.testing.assert_allclose(s, [5., 2.])


def test_speeds_are_read_at_ticks_after_the_cut_and_skip_ticks_past_the_end():
    speed = np.arange(450, dtype=float)
    out = arms.speeds_at(speed, 400, ticks=(0, 10, 49, 50))
    assert out == {'0': 400., '10': 410., '49': 449.}


def test_every_window_recovers_the_constant_of_a_true_exponential():
    out = arms.window_taus(decay(.25), 400)
    for window, tau in out.items():
        assert tau == pytest.approx(.25, rel=1e-6), window


def test_a_decay_with_a_slow_tail_gives_a_different_constant_in_each_window():
    t = np.arange(600); k = np.maximum(t-400, 0)*.02
    speed = np.where(t < 400, .13, .11*np.exp(-k/.15) + .03*np.exp(-k/1.5))
    out = arms.window_taus(speed, 400, windows=((0, 10), (10, 50)))
    assert out['0-10'] < out['10-50']
    assert out['10-50'] > 2.*out['0-10']


def test_the_fit_floor_drops_samples_that_are_only_noise():
    speed = decay(.25); speed[460:] = .001
    assert arms.window_taus(speed, 400, windows=((0, 100),), floor=.005)['0-100'] is not None


# --- displacement and share ------------------------------------------------------
def line(n, step):
    return np.stack([np.arange(n)*step, np.zeros(n)], axis=1)


def test_euclidean_and_path_agree_on_a_straight_line():
    out = arms.displacement(line(700, .001), 400)
    assert out['euclidean_m'] == pytest.approx(.2) and out['path_m'] == pytest.approx(.2)


def test_path_counts_motion_that_euclidean_cancels():
    xy = np.zeros((700, 2)); xy[401:, 0] = .01; xy[450:, 0] = 0.
    out = arms.displacement(xy, 400)
    assert out['euclidean_m'] == pytest.approx(0.) and out['path_m'] == pytest.approx(.02)


def test_displacement_stops_at_the_horizon_and_at_the_end_of_the_trace():
    assert arms.displacement(line(700, .001), 400, horizon=50)['euclidean_m'] == pytest.approx(.05)
    assert arms.displacement(line(430, .001), 400)['euclidean_m'] == pytest.approx(.029)


def test_the_share_explained_is_the_fraction_that_disappears():
    assert arms.share_explained(.0753, .0532) == pytest.approx(1.-.0532/.0753)
    assert arms.share_explained(.1, .1) == pytest.approx(0.)
    assert arms.share_explained(.1, 0.) == pytest.approx(1.)
    assert arms.share_explained(0., .05) is None


def test_the_share_is_negative_when_forcing_zero_made_it_worse():
    assert arms.share_explained(.05, .08) < 0.


# --- posture ----------------------------------------------------------------------
def test_posture_reads_the_median_of_the_named_window_after_the_cut():
    pitch = np.zeros(700); pitch[420:450] = -.15; pitch[550:600] = -.11
    trace = {'base_pitch_rad': pitch, 'base_height_m': np.full(700, .44)}
    assert arms.posture_after(trace, 400, 20, 50)['pitch_rad'] == pytest.approx(-.15)
    assert arms.posture_after(trace, 400, 150, 200)['pitch_rad'] == pytest.approx(-.11)


# --- the whole description ---------------------------------------------------------
def trace_with(speed, n=600):
    x = np.concatenate([[0.], np.cumsum(speed*.02)])
    return {'body_vx_m_s': speed, 'body_vy_m_s': np.zeros(n), 'boundary_xy_m': np.stack([x, np.zeros_like(x)], axis=1),
            'base_pitch_rad': np.full(n, -.1), 'base_height_m': np.full(n, .44)}


def test_describe_reports_entry_speed_and_the_stop_displacement_of_a_known_decay():
    out = arms.describe(trace_with(decay(.25, v0=.15)))
    assert out['entry_speed_m_s'] == pytest.approx(.15)
    assert out['displacement_200_ticks']['path_m'] == pytest.approx(.15*.25*(1.-np.exp(-4./.25)), rel=.05)
    assert out['tau_s_by_fit_window_ticks']['10-50'] == pytest.approx(.25, rel=1e-6)


def test_a_faster_decay_stops_sooner_and_travels_less():
    fast, slow = arms.describe(trace_with(decay(.1))), arms.describe(trace_with(decay(.4)))
    assert fast['displacement_200_ticks']['path_m'] < slow['displacement_200_ticks']['path_m']
    assert fast['ticks_until_speed_below_0p02'] < slow['ticks_until_speed_below_0p02']


def test_posture_median_ignores_a_single_spike_in_the_window():
    pitch = np.full(700, -.15); pitch[430] = 1.
    trace = {'base_pitch_rad': pitch, 'base_height_m': np.full(700, .44)}
    assert arms.posture_after(trace, 400, 20, 50)['pitch_rad'] == pytest.approx(-.15)


def test_entry_speed_is_the_mean_of_the_five_ticks_before_the_cut():
    speed = decay(.25, v0=.15); speed[390:395] = .30; speed[395:400] = .10   # last five average .10, last ten .20
    assert arms.describe(trace_with(speed))['entry_speed_m_s'] == pytest.approx(.10)


def test_describe_reads_the_early_and_late_posture_windows_it_names():
    pitch = np.zeros(700); pitch[420:450] = -.15; pitch[550:600] = -.11
    t = trace_with(decay(.25)); t['base_pitch_rad'] = pitch[:600]
    out = arms.describe(t)
    assert out['posture_ticks_20_to_50']['pitch_rad'] == pytest.approx(-.15)
    assert out['posture_ticks_150_to_200']['pitch_rad'] == pytest.approx(-.11)


def test_the_default_fit_windows_include_the_four_the_results_document_quotes():
    out = arms.describe(trace_with(decay(.25)))
    assert list(out['tau_s_by_fit_window_ticks']) == ['0-10', '5-25', '10-50', '25-100']


def test_main_writes_a_receipt_with_both_arms_the_share_and_the_post_hoc_label(tmp_path, monkeypatch):
    for name, tau in (('policy', .5), ('forced', .2)):
        np.savez_compressed(tmp_path/f'{name}.npz', **trace_with(decay(tau)))
    out = tmp_path/'out'
    monkeypatch.setattr(sys, 'argv', ['compare', '--policy-arm', str(tmp_path/'policy.npz'),
                                      '--forced-zero-arm', str(tmp_path/'forced.npz'), '--output', str(out)])
    assert arms.main() == 0
    report = json.loads((out/'report.json').read_text())
    assert report['physics_steps_executed'] == 0 and report['cut_tick'] == 400
    assert 'post-hoc' in report['note']
    assert report['inputs']['policy_arm_sha256'] == arms.sha(tmp_path/'policy.npz')
    assert report['share_of_euclidean_displacement_explained_by_policy_wheel_action'] > 0.
    assert report['policy_wheel_arm']['tau_s_by_fit_window_ticks']['10-50'] > report['forced_zero_wheel_arm']['tau_s_by_fit_window_ticks']['10-50']
    with pytest.raises(FileExistsError):
        arms.main()


def test_pitch_on_the_drive_plateau_is_the_median_of_the_last_fifty_ticks_before_the_cut():
    tr = trace_with(decay(.25)); pitch = np.full(600, -.30); pitch[350:400] = -.15; pitch[375] = 1.
    tr['base_pitch_rad'] = pitch
    assert arms.describe(tr)['pitch_on_drive_plateau_rad'] == pytest.approx(-.15)


def test_peak_pitch_is_a_magnitude_and_is_split_at_the_cut():
    tr = trace_with(decay(.25)); pitch = np.zeros(600); pitch[100] = -.2; pitch[450] = .3
    tr['base_pitch_rad'] = pitch
    peak = arms.describe(tr)['peak_abs_pitch_rad']
    assert peak['before_cut'] == pytest.approx(.2) and peak['after_cut'] == pytest.approx(.3)


def test_a_negative_peak_after_the_cut_counts_by_its_magnitude():
    """Forward lean is negative pitch, so the larger lean has the more negative value."""
    tr = trace_with(decay(.25)); pitch = np.zeros(600); pitch[100] = .1; pitch[450] = -.3
    tr['base_pitch_rad'] = pitch
    peak = arms.describe(tr)['peak_abs_pitch_rad']
    assert peak['after_cut'] == pytest.approx(.3) and peak['before_cut'] == pytest.approx(.1)


def test_a_peak_just_after_the_cut_belongs_to_after_and_just_before_to_before():
    tr = trace_with(decay(.25)); pitch = np.zeros(600); pitch[399] = -.2; pitch[400] = -.3
    tr['base_pitch_rad'] = pitch
    peak = arms.describe(tr)['peak_abs_pitch_rad']
    assert peak['before_cut'] == pytest.approx(.2) and peak['after_cut'] == pytest.approx(.3)


def test_the_cut_tick_itself_is_in_after_and_a_larger_value_just_before_is_not():
    """The tick before the cut holds the biggest value overall, so including it in `after` would show."""
    tr = trace_with(decay(.25)); pitch = np.zeros(600); pitch[399] = -.4; pitch[400] = -.3
    tr['base_pitch_rad'] = pitch
    peak = arms.describe(tr)['peak_abs_pitch_rad']
    assert peak['before_cut'] == pytest.approx(.4) and peak['after_cut'] == pytest.approx(.3)


def test_pitch_per_unit_speed_is_the_plateau_pitch_over_the_entry_speed():
    tr = trace_with(decay(.25, v0=.2)); tr['base_pitch_rad'] = np.full(600, -.22)
    assert arms.describe(tr)['pitch_per_unit_entry_speed_rad_per_m_s'] == pytest.approx(-1.1)


def test_the_tool_imports_no_simulator():
    source = Path(arms.__file__).read_text()
    assert 'import mujoco' not in source and 'onnxruntime' not in source


# --- numbers in the results document come from the receipt ---------------------------
@needs_workstation_evidence
def test_numbers_in_the_results_document_agree_with_the_receipt():
    if not RECEIPT.is_file():
        pytest.skip('arm comparison receipt not on this machine')
    r = json.loads(RECEIPT.read_text())
    text = (ROOT/'docs/career_execution_results_04.md').read_text()
    forced, policy = r['forced_zero_wheel_arm'], r['policy_wheel_arm']
    assert f"{forced['displacement_200_ticks']['euclidean_m']:.4f}" in text
    assert f"{policy['displacement_200_ticks']['euclidean_m']:.4f}" in text
    assert f"{100*r['share_of_euclidean_displacement_explained_by_policy_wheel_action']:.0f}%" in text
    for window in ('0-10', '5-25', '10-50'):
        assert f"{forced['tau_s_by_fit_window_ticks'][window]:.2f}" in text


@needs_workstation_evidence
def test_the_full_verdict_in_the_results_document_agrees_with_its_receipt():
    verdict = ROOT/'evidence/career_contract_05/probe05_verdict_01/verdict.json'
    if not verdict.is_file():
        pytest.skip('verdict receipt not on this machine')
    v = json.loads(verdict.read_text())
    text = (ROOT/'docs/career_execution_results_04.md').read_text()
    assert v['predictions'] == {'P1': 'MET', 'P2': 'REFUTED', 'P3': 'NOT_MET', 'P4': 'MET', 'P5': 'NOT_MET'}
    for key in ('A_forced_zero_cmd0p2', 'B_forced_zero_cmd0p3'):
        assert f"{v['tau_s'][key]:.3f}" in text
    assert f"{v['p4_precondition']['entry_speed_ratio_B_over_A']:.2f}" in text
    for name in 'ABC':
        r = json.loads((ROOT/f'evidence/career_contract_05/probe05_run{name}_01/report.json').read_text())['analysis']
        assert f"{r['entry_speed_m_s']:.4f}" in text
        assert f"{r['final_euclidean_displacement_m']:.4f}" in text
    assert sum(json.loads((ROOT/f'evidence/career_contract_05/probe05_run{n}_01/report.json').read_text())['actual_counts']['native_steps']
               for n in ('A', 'B', 'C')) == 18000
