"""Tests for tools05/analyze_lean05.py.

The tool exists because a ratio that silently included the resting lean was read as a
constant of proportionality. So the tests pin exactly that distinction, plus the
two-point line and the check of whether it passes through rest. No model, no physics.
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

import analyze_lean05 as lean  # noqa: E402

RECEIPT = ROOT/'evidence/career_contract_05/lean_analysis_01/report.json'


def trace(rest, plateau, settled, speed, n=600):
    pitch = np.zeros(n); pitch[:100] = rest; pitch[100:400] = plateau; pitch[400:] = settled
    vx = np.zeros(n); vx[100:400] = speed
    return {'base_pitch_rad': pitch, 'body_vx_m_s': vx, 'body_vy_m_s': np.zeros(n),
            'tilt_rad': np.abs(pitch), 'base_height_m': np.full(n, .44)}


def test_the_ratio_including_rest_and_the_ratio_above_rest_differ_by_the_resting_lean():
    p = lean.lean_point(trace(-.0665, -.1467, -.11, .1322))
    assert p['ratio_including_rest_rad_per_m_s'] == pytest.approx(-.1467/.1322)
    assert p['ratio_above_rest_rad_per_m_s'] == pytest.approx((-.1467+.0665)/.1322)
    assert p['lean_above_rest_rad'] == pytest.approx(-.0802)


def test_with_no_resting_lean_the_two_ratios_agree():
    p = lean.lean_point(trace(0., -.15, 0., .15))
    assert p['ratio_including_rest_rad_per_m_s'] == pytest.approx(p['ratio_above_rest_rad_per_m_s'])


def test_each_window_is_read_where_it_says():
    p = lean.lean_point(trace(-.06, -.15, -.11, .13))
    assert (p['rest_pitch_rad'], p['plateau_pitch_rad'], p['settled_pitch_rad']) == pytest.approx((-.06, -.15, -.11))
    assert p['speed_m_s'] == pytest.approx(.13)


def test_the_windows_are_the_ones_the_documents_quote():
    assert lean.REST == (25, 100) and lean.PLATEAU == (350, 400) and lean.SETTLED == (550, 600) and lean.ENTRY == (395, 400)


def test_entry_speed_is_horizontal_and_uses_only_the_last_five_ticks():
    t = trace(0., 0., 0., .1); t['body_vy_m_s'][395:400] = .1; t['body_vx_m_s'][390:395] = 9.
    assert lean.entry_speed(t) == pytest.approx(np.hypot(.1, .1))


def test_zero_speed_gives_no_ratio_instead_of_dividing_by_zero():
    p = lean.lean_point(trace(-.06, -.06, -.06, 0.))
    assert p['ratio_including_rest_rad_per_m_s'] is None and p['ratio_above_rest_rad_per_m_s'] is None


def test_two_point_line_on_a_true_line_recovers_slope_and_intercept():
    a = {'speed_m_s': .1, 'plateau_pitch_rad': -.05-.1}
    b = {'speed_m_s': .2, 'plateau_pitch_rad': -.05-.2}
    line = lean.two_point_line(a, b)
    assert line['slope_rad_per_m_s'] == pytest.approx(-1.) and line['intercept_at_zero_speed_rad'] == pytest.approx(-.05)


def test_two_point_line_with_equal_speeds_is_none():
    p = {'speed_m_s': .1, 'plateau_pitch_rad': -.1}
    assert lean.two_point_line(p, p) is None


def test_through_rest_is_true_only_when_the_intercept_matches_rest():
    line = {'slope_rad_per_m_s': -1., 'intercept_at_zero_speed_rad': -.05}
    assert lean.through_rest(line, -.05) is True
    assert lean.through_rest(line, -.05+.019) is True
    assert lean.through_rest(line, -.05+.021) is False
    assert lean.through_rest(None, -.05) is None


def test_the_04b_shape_is_monotonic_but_not_a_line_through_rest():
    """The three measured points: lean grows with speed, but no single line joins them."""
    low = lean.lean_point(trace(-.0665, -.1467, -.11, .1322))
    high = lean.lean_point(trace(-.0665, -.2083, -.14, .1758))
    line = lean.two_point_line(low, high)
    assert line['slope_rad_per_m_s'] == pytest.approx(-1.413, abs=.01)
    assert lean.through_rest(line, low['rest_pitch_rad']) is False


def test_release_states_read_the_last_drive_row_and_the_tick_before_the_cut():
    rows = [{'tick': 98, 'vx': 1., 'vy': 0., 'tilt': .5, 'height': .1},
            {'tick': 99, 'vx': .06, 'vy': .08, 'tilt': .08, 'height': .44}]
    out = lean.release_state_from_probe(rows)
    assert out == {'tick': 99, 'speed_m_s': pytest.approx(.1), 'tilt_rad': .08, 'height_m': .44}
    t = trace(0., -.15, 0., .13)
    assert lean.release_state_from_trace(t, 400)['tick'] == 399
    assert lean.release_state_from_trace(t, 400)['tilt_rad'] == pytest.approx(.15)


def test_main_records_the_passive_braking_caveat_and_refuses_to_overwrite(tmp_path, monkeypatch):
    for name, (p, v) in {'low': (-.1467, .1322), 'high': (-.2083, .1758), 'a': (-.1467, .1322)}.items():
        np.savez_compressed(tmp_path/f'{name}.npz', **trace(-.0665, p, -.11, v))
    (tmp_path/'drive.jsonl').write_text(json.dumps({'tick': 99, 'vx': .08, 'vy': 0., 'tilt': .08, 'height': .44})+'\n')
    out = tmp_path/'out'
    monkeypatch.setattr(sys, 'argv', ['lean', '--low', str(tmp_path/'low.npz'), '--high', str(tmp_path/'high.npz'),
                                      '--probe-drive', str(tmp_path/'drive.jsonl'), '--forced-zero', str(tmp_path/'a.npz'),
                                      '--output', str(out)])
    assert lean.main() == 0
    r = json.loads((out/'report.json').read_text())
    assert 'passive damping' in r['note'] and 'active reverse braking was never measured' in r['note']
    assert r['physics_steps_executed'] == 0 and r['line_passes_through_rest_within_0p02_rad'] is False
    with pytest.raises(FileExistsError):
        lean.main()


def test_the_tool_imports_no_simulator():
    source = Path(lean.__file__).read_text()
    assert 'import mujoco' not in source and 'onnxruntime' not in source


@needs_workstation_evidence
def test_numbers_quoted_in_the_documents_agree_with_the_receipt():
    if not RECEIPT.is_file():
        pytest.skip('lean receipt not on this machine')
    r = json.loads(RECEIPT.read_text())
    readme = (ROOT/'README.md').read_text()
    results = (ROOT/'docs/career_execution_results_04.md').read_text()
    rest = r['low_command']['rest_pitch_rad']
    for text in (readme, results):
        assert f'{rest:+.3f}' in text
    assert f"{r['line_through_plateaus']['slope_rad_per_m_s']:+.2f}" in results
    assert f"{r['line_through_plateaus']['intercept_at_zero_speed_rad']:+.3f}" in results
    for point in ('low_command', 'high_command'):
        assert f"{r[point]['ratio_above_rest_rad_per_m_s']:+.2f}" in results
    probe, run_a = r['wheels_forced_to_zero_release_state']['appendix_a_probe'], r['wheels_forced_to_zero_release_state']['contract_05m_run_a']
    assert f"{probe['tilt_rad']:.3f}" in results and f"{run_a['tilt_rad']:.3f}" in results
    summary = readme.split('## 当前结果')[0]
    lean_item = next(line for line in summary.splitlines() if '驱动相稳定在偏低的平台上' in line)
    assert '形状未知' in lean_item
    for sentence in lean_item.split('。'):                 # the old ratio may appear only as the thing corrected
        if '-1.11 到 -1.18' in sentence or '−1.11 到 −1.18' in sentence:
            assert '已更正' in sentence or '读过头' in sentence, sentence
