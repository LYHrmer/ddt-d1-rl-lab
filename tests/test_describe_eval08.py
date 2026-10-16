"""Tests for tools08/describe_eval08.py. No model, no physics."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools08'))

import describe_eval08 as describe  # noqa: E402


def xy_from_x(xs):
    return np.stack([np.asarray(xs, dtype=float), np.zeros(len(xs))], axis=1)


def test_a_straight_stop_has_path_equal_to_displacement_and_no_reversal():
    xs = np.concatenate([np.zeros(401), np.linspace(0, .04, 200)])
    out = describe.after_cut_motion(xy_from_x(xs))
    assert out['euclidean_m'] == pytest.approx(.04) and out['path_m'] == pytest.approx(.04)
    assert out['path_over_euclidean'] == pytest.approx(1.) and out['x_direction_reversals'] == 0


def test_back_and_forth_motion_is_seen_by_path_and_reversals_not_by_displacement():
    xs = np.zeros(601); xs[401:] = np.tile([.01, 0.], 100)
    out = describe.after_cut_motion(xy_from_x(xs))
    assert out['euclidean_m'] == pytest.approx(0.) and out['path_m'] == pytest.approx(2.)
    assert out['path_over_euclidean'] is None and out['x_direction_reversals'] == 199


def test_standing_still_ticks_do_not_count_as_reversals():
    xs = np.zeros(601); xs[401:450] = np.linspace(0, .01, 49); xs[450:] = .01
    assert describe.after_cut_motion(xy_from_x(xs))['x_direction_reversals'] == 0


def test_a_trace_that_ends_before_the_cut_has_no_after_cut_motion():
    assert describe.after_cut_motion(xy_from_x(np.zeros(300))) is None


def trace(n=600, speed=.2, pitch_drive=-.17, pitch_end=-.4):
    vx = np.zeros(n); vx[100:400] = speed
    pitch = np.zeros(n); pitch[350:400] = pitch_drive; pitch[400:] = np.linspace(pitch_drive, pitch_end, n-400)
    return {'body_vx_m_s': vx, 'base_pitch_rad': pitch, 'boundary_xy_m': xy_from_x(np.linspace(0, .5, n+1))}


def test_describe_reads_drive_speed_along_the_command_and_the_lean():
    out = describe.describe(trace(), 1.)
    assert out['drive_last_second_speed_along_command_m_s'] == pytest.approx(.2)
    assert out['drive_last_second_pitch_rad'] == pytest.approx(-.17) and out['pitch_at_end_rad'] == pytest.approx(-.4)
    assert out['peak_abs_pitch_after_cut_rad'] == pytest.approx(.4) and out['ended_early'] is False
    rev = trace(speed=-.2)
    assert describe.describe(rev, -1.)['drive_last_second_speed_along_command_m_s'] == pytest.approx(.2)


def test_an_early_end_is_flagged_and_turn_has_no_drive_speed():
    out = describe.describe(trace(n=485), 0.)
    assert out['ended_early'] is True and out['ticks'] == 485 and out['drive_last_second_speed_along_command_m_s'] is None


def test_the_tool_imports_no_simulator():
    source = Path(describe.__file__).read_text()
    assert 'import mujoco' not in source and 'onnxruntime' not in source
