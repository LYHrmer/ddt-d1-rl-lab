import hashlib
import json
import math
from pathlib import Path

import pytest

from tools84.terrain_gradient84 import (
    FD_STEP_M,
    MeasuredProfile,
    audit_file,
    check_frame,
)


def measured_frame(query=0.65):
    # The measured tracks have different slopes and opposite beam index order.
    return {
        'x_m': [[0.0, 0.5, 0.8, 1.1], [1.0, 0.9, 0.5, 0.1]],
        'z_m': [[0.0, 0.16, 0.19, 0.8], [-0.5, -0.3, -0.1, 0.0]],
        'query_x_m': query,
    }


@pytest.mark.parametrize(('xs', 'zs', 'query'), [
    ([0.0, 0.5, 0.8, 1.1], [0, None, 2, 3], 0.65),
    ([0.0, 0.5, math.nan, 1.1], [0, 1, 2, 3], 0.65),
    ([0.0, 0.5, 0.8, 1.1], [0, 1, 2, 3], 1.11),
    ([0.0, 0.5, 0.8, 1.1], [0, 1, 2, 3], 0.5),
    ([0.0, 0.5, 0.8, 1.1], [0, 1, 2, 3], 0.5 + FD_STEP_M / 2),
    ([0.0, 0.5, 0.5, 1.1], [0, 1, 2, 3], 0.5),
    ([0.0, 0.8, 0.4, 1.1], [0, 1, 2, 3], 0.65),
])
def test_one_unsupported_track_invalidates_the_stencil(xs, zs, query):
    frame = measured_frame(query)
    frame['x_m'][0] = xs
    frame['z_m'][0] = zs
    profile = MeasuredProfile(frame['x_m'], frame['z_m'])
    assert profile.segments(query, FD_STEP_M) is None


def test_missing_beams_outside_the_selected_pair_are_allowed():
    frame = measured_frame()
    frame['x_m'][0][3] = None
    frame['z_m'][1][0] = math.nan
    profile = MeasuredProfile(frame['x_m'], frame['z_m'])
    assert len(profile.segments(frame['query_x_m'], FD_STEP_M)) == 2


@pytest.mark.parametrize('bad', [[], [[1, 2], [3, 4]], [[0, 1, 2, 3], [1, 2, 3, math.inf]]])
def test_bad_shapes_and_infinities_fail(bad):
    with pytest.raises(ValueError):
        MeasuredProfile(bad, measured_frame()['z_m'])


def test_track_derivatives_are_independent_and_stop_only_blocks_the_query_path():
    torch = pytest.importorskip('torch')
    frame = measured_frame()
    record = check_frame(frame)
    assert record['status'] == 'PASS'
    assert record['height_m'] == pytest.approx([0.175, -0.175])
    for field in ('analytic_dz_dx', 'finite_difference_dz_dx', 'autograd_dz_dx'):
        assert record[field] == pytest.approx([0.1, -0.5])
    assert record['forward_equal_when_stopped']
    assert record['stopped_dz_dx'] == [0.0, 0.0]
    profile = MeasuredProfile(frame['x_m'], frame['z_m'])
    q = torch.tensor(0.65, dtype=torch.float64, requires_grad=True)
    bypass = 2 * q + profile.heights(q, stop_coordinate_gradient=True).sum()
    assert torch.autograd.grad(bypass, q)[0].item() == 2.0


def test_file_audit_counts_coverage_and_refuses_overwrite(tmp_path):
    pytest.importorskip('torch')
    frames, output = tmp_path / 'frames.jsonl', tmp_path / 'report.json'
    flat = measured_frame()
    flat['z_m'] = [[0] * 4, [0] * 4]
    frames.write_text('\n'.join(json.dumps(r) for r in (flat, measured_frame(), measured_frame(5))) + '\n')
    report = audit_file(frames, output)
    assert report == json.loads(output.read_text())
    assert report['status'] == 'PASS'
    assert [report[key] for key in ('frames', 'checked', 'passed', 'failed', 'uncovered')] == [3, 2, 2, 0, 1]
    assert report['nonzero_slope_frames'] == 1
    assert report['input_sha256'] == hashlib.sha256(frames.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        audit_file(frames, output)
    frames.write_text(json.dumps(measured_frame(5)))
    unobserved = audit_file(frames, tmp_path / 'unobserved.json')
    assert unobserved['status'] == 'NO_COVERAGE'
    assert (unobserved['passed'], unobserved['uncovered']) == (0, 1)


def test_invalid_json_is_retained_as_incomplete(tmp_path):
    frames, output = tmp_path / 'frames.jsonl', tmp_path / 'report.json'
    frames.write_text('{')
    with pytest.raises(json.JSONDecodeError):
        audit_file(frames, output)
    partial = json.loads(output.read_text())
    assert partial['status'] == 'INCOMPLETE'
    assert partial['frames'] == 1
    assert 'input_sha256' not in partial


def test_audit_respects_wheel_validity_and_checks_recorded_slope():
    pytest.importorskip('torch')
    frame = measured_frame()
    frame.update(valid=False, reason='expired')
    assert check_frame(frame)['reason'] == 'expired'
    frame.update(valid=True, left_slope_rad=math.atan(0.1), right_slope_rad=math.atan(-0.5))
    assert check_frame(frame)['status'] == 'PASS'
    frame['right_slope_rad'] *= -1
    assert check_frame(frame)['status'] == 'FAIL'


@pytest.mark.parametrize('name, status, frames, covered, nonflat', [
    ('static_frames.jsonl', 'PASS', 9, 9, 6),
    ('frames.jsonl', 'NO_COVERAGE', 924, 0, 0),
    ('translated_mount_frames.jsonl', 'PASS', 308, 80, 39),
])
def test_published_positive_control_and_real_run(tmp_path, name, status, frames, covered, nonflat):
    pytest.importorskip('torch')
    source = Path(__file__).resolve().parents[1] / 'docs/assets/project84' / name
    report = audit_file(source, tmp_path / 'audit.json')
    assert report['status'] == status
    assert (report['frames'], report['checked'], report['uncovered']) == (frames, covered, frames - covered)
    assert report['nonzero_slope_frames'] == nonflat
    assert report['passed'] == covered and report['failed'] == 0
