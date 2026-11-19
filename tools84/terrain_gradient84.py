"""Audit the spatial derivative of two wheel-track profiles in the capture frame.

Pass one JSONL row per newly acquired packet. x_m and z_m store two tracks of
four ordered beams in metres. Null or NaN marks an unavailable return. The
scalar query_x_m is the current or predicted axle location expressed in the
same capture frame. Agreement tests the spatial interpolant. A robot or policy
response requires a separate closed-loop perturbation experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

FD_STEP_M = 1e-5
ABS_TOL = 1e-8
REL_TOL = 1e-7


def _matrix(values, name):
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f'{name} must contain two tracks')
    tracks = []
    for row in values:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise ValueError(f'{name} must contain four beam positions per track')
        converted = []
        for value in row:
            if value is None:
                converted.append(math.nan)
            elif type(value) in (float, int) and not math.isinf(value):
                converted.append(float(value))
            else:
                raise ValueError(f'{name} requires numbers, null or NaN')
        tracks.append(tuple(converted))
    return tuple(tracks)


class MeasuredProfile:
    def __init__(self, x_m, z_m):
        self.x_m = _matrix(x_m, 'x_m')
        self.z_m = _matrix(z_m, 'z_m')

    def segments(self, query: float, margin: float = 0.0):
        """Find one adjacent measured segment per track containing the open interior query.

        The entire central-difference stencil must fit the same two segments.
        Counting intersections before selection rejects knots, enlarged stencils
        that encounter another segment, overlapping intervals, and zero spans.
        """
        if not math.isfinite(query) or not math.isfinite(margin) or margin < 0:
            raise ValueError('query and nonnegative margin must be finite')
        lo, hi = query - margin, query + margin
        if margin and not lo < query < hi:
            raise ValueError('difference step is not representable at query_x_m')
        result = []
        for xs, zs in zip(self.x_m, self.z_m, strict=True):
            hits = []
            for i in range(3):
                x0, x1, z0, z1 = xs[i], xs[i + 1], zs[i], zs[i + 1]
                if not all(math.isfinite(v) for v in (x0, x1, z0, z1)):
                    continue
                left, right = min(x0, x1), max(x0, x1)
                if right >= lo and left <= hi:
                    if not left < lo <= hi < right:
                        return None
                    hits.append((x0, x1, z0, z1))
            if len(hits) != 1:
                return None
            result.append(hits[0])
        return result

    def heights(self, query_x_m, *, stop_coordinate_gradient: bool = False):
        """Return two measured heights, or None where either wheel track is unknown.

        Only the interpolation coordinate's derivative is stopped in the
        ablation. A zero-gradient edge keeps ordinary downstream backprop valid.
        """
        import torch

        if query_x_m.ndim != 0 or not query_x_m.is_floating_point():
            raise ValueError('query_x_m must be a floating-point scalar tensor')
        segments = self.segments(query_x_m.detach().item())
        if segments is None:
            return None
        q = query_x_m.detach() + 0.0 * query_x_m if stop_coordinate_gradient else query_x_m
        heights = []
        for x0, x1, z0, z1 in segments:
            weight = (q - x0) / (x1 - x0)
            heights.append((1 - weight) * z0 + weight * z1)
        return torch.stack(heights)


def check_frame(frame: dict) -> dict:
    import torch

    if not isinstance(frame, dict) or not {'x_m', 'z_m', 'query_x_m'} <= frame.keys():
        raise ValueError('frame requires x_m, z_m and query_x_m')
    query = frame['query_x_m']
    if type(query) not in (float, int) or not math.isfinite(query):
        raise ValueError('query_x_m must be a finite scalar in the capture frame')
    if 'valid' in frame and type(frame['valid']) is not bool:
        raise ValueError('wheel packet validity must be a boolean')
    if frame.get('valid') is False:
        return {'status': 'UNCOVERED', 'query_x_m': query, 'reason': frame.get('reason', 'sensor_gate')}
    profile = MeasuredProfile(frame['x_m'], frame['z_m'])
    segments = profile.segments(query, FD_STEP_M)
    if segments is None:
        return {'status': 'UNCOVERED', 'query_x_m': query, 'reason': 'no_two_sided_stencil'}

    q = torch.tensor(query, dtype=torch.float64, device='cpu', requires_grad=True)
    slopes = [(z1 - z0) / (x1 - x0) for x0, x1, z0, z1 in segments]
    reference = [
        z0 + (query - x0) * slope for (x0, _, z0, _), slope in zip(segments, slopes, strict=True)
    ]
    live = profile.heights(q)
    stop = profile.heights(q, stop_coordinate_gradient=True)
    gradients = [torch.autograd.grad(value, q, retain_graph=True)[0].item() for value in live]
    stopped = [torch.autograd.grad(value, q, retain_graph=True)[0].item() for value in stop]
    with torch.no_grad():
        offset_heights = profile.heights(q + FD_STEP_M) - profile.heights(q - FD_STEP_M)
        finite_diff = (offset_heights / (2 * FD_STEP_M)).tolist()
    values = live.detach().tolist()
    outputs = (slopes, reference, values, gradients, stopped, finite_diff)
    if not all(math.isfinite(v) for group in outputs for v in group):
        raise ValueError('nonfinite interpolation or derivative on a covered segment')
    forward_equal = torch.equal(live, stop)
    pairs = [*zip(reference, values), *zip(slopes, gradients), *zip(slopes, finite_diff)]
    if 'left_slope_rad' in frame or 'right_slope_rad' in frame:
        pairs.extend((math.atan(slope), frame[key]) for slope, key in
                     zip(slopes, ('left_slope_rad', 'right_slope_rad'), strict=True))
    matched = all(math.isclose(a, b, abs_tol=ABS_TOL, rel_tol=REL_TOL) for a, b in pairs)
    passed = matched and forward_equal and all(derivative == 0.0 for derivative in stopped)
    return {
        'status': 'PASS' if passed else 'FAIL', 'query_x_m': query,
        'height_m': values, 'analytic_height_m': reference, 'forward_equal_when_stopped': forward_equal,
        'analytic_dz_dx': slopes, 'finite_difference_dz_dx': finite_diff,
        'autograd_dz_dx': gradients, 'stopped_dz_dx': stopped,
    }


def audit_file(frames: Path, output: Path) -> dict:
    """Create an exclusive report, retaining incomplete results if an input row fails."""
    report = {
        'format': 'measured-wheel-track-gradient84-v1', 'input': frames.name, 'status': 'INCOMPLETE',
        'fd_step_m': FD_STEP_M, 'abs_tol': ABS_TOL, 'rel_tol': REL_TOL,
        'frames': 0, 'checked': 0, 'passed': 0, 'failed': 0, 'uncovered': 0,
        'nonzero_slope_frames': 0, 'max_abs_fd_error': 0.0, 'max_abs_autograd_error': 0.0, 'records': [],
    }
    digest = hashlib.sha256()
    with output.open('x', encoding='utf-8') as dest:
        try:
            with frames.open('rb') as source:
                for line, raw in enumerate(source, 1):
                    digest.update(raw)
                    report['frames'] += 1
                    frame = json.loads(raw)
                    record = check_frame(frame)
                    for key in ('case', 'sequence', 'captured_s'):
                        if key in frame:
                            record[key] = frame[key]
                    record['line'] = line
                    report['records'].append(record)
                    if record['status'] == 'UNCOVERED':
                        report['uncovered'] += 1
                    else:
                        report['checked'] += 1
                        report['passed' if record['status'] == 'PASS' else 'failed'] += 1
                        if any(abs(v) > ABS_TOL for v in record['analytic_dz_dx']):
                            report['nonzero_slope_frames'] += 1
                        for field, metric in (('finite_difference_dz_dx', 'max_abs_fd_error'),
                                              ('autograd_dz_dx', 'max_abs_autograd_error')):
                            error = max(abs(a - b) for a, b in zip(record['analytic_dz_dx'], record[field]))
                            report[metric] = max(report[metric], error)
            report['input_sha256'] = digest.hexdigest()
            if report['failed']:
                report['status'] = 'FAIL'
            elif report['checked']:
                report['status'] = 'PASS'
            else:
                report['status'] = 'NO_COVERAGE'
        except Exception as exc:
            report.update(status='INCOMPLETE', error=str(exc), read_prefix_sha256=digest.hexdigest())
            raise
        finally:
            json.dump(report, dest, ensure_ascii=False, allow_nan=False)
            dest.write('\n')
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit_file(args.frames, args.output)
    print(f"{report['status']} ({report['passed']} passed, {report['failed']} failed, "
          f"{report['uncovered']} uncovered)")
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
