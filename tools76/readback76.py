"""Explain archived72 response failures without loading a model or changing a gate.

This standard-library log reader retains every registered model, point and
epsilon. It separates arithmetic agreement, magnitude, model sign and physical
reference qualification. Endpoint-neighbor sensitivity is not a full numerical
error bound. Missing conditional yaw JVPs remain explicitly missing.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import json
import math
import resource
import struct
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools74.runtime74 import run_preserving
from tools76.precision76 import endpoint_precision
from tools76.reasons76 import explain_response

ARCHIVE = 'evidence/yaw_weight72/execution_01'
PUBLISHED = 'reports/yaw_weight72_results_20261101.json'
MODEL_SEEDS = (64001, 64002, 64003)
ARMS = ('control', 'active')
ROLES = ('development', 'old_probe', 'old55_local', 'old56', 'old64', 'old69', 'fresh')
FIRST_ROLES = ('old56', 'old64', 'old69', 'fresh')
ROLE_SEEDS = {'development': (53201, 53202, 53203), 'old_probe': (53301, 53302, 53303),
              'old55_local': (55301, 55302, 55303), 'old56': (56301, 56302, 56303),
              'old64': (64301, 64302, 64303), 'old69': (69301, 69302, 69303),
              'fresh': (72301, 72302, 72303)}
EPS = (3.125e-5, 6.25e-5, 1.25e-4, 2.5e-4)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def f64_array(archive, key, shape):
    """Read only the exact small C-order F64 NPY member needed for a scalar audit."""
    name = key + '.npy'
    require(archive.namelist().count(name) == 1, 'missing or duplicate NPY member: ' + name)
    info = archive.getinfo(name)
    require(info.file_size <= 65536, 'unexpected large scalar-audit NPY member')
    data = archive.read(name)
    require(data[:6] == b'\x93NUMPY' and data[6:8] in (b'\x01\x00', b'\x02\x00'), 'unsupported NPY version')
    width = 2 if data[6] == 1 else 4
    length = int.from_bytes(data[8:8 + width], 'little')
    header = ast.literal_eval(data[8 + width:8 + width + length].decode('latin1'))
    require(header == {'descr': '<f8', 'fortran_order': False, 'shape': shape}, 'raw F64 shape/order differs')
    count = math.prod(shape)
    payload = data[8 + width + length:]
    require(len(payload) == 8 * count, 'NPY payload length differs')
    values = struct.unpack('<' + 'd' * count, payload)
    require(all(math.isfinite(x) for x in values), 'nonfinite archived endpoint')
    return values


def check_fields(point):
    return {key: point[key] for key in ('gradient', 'model_central_difference', 'autograd_agrees', 'direction_identifiable')}


def ordered(points, role):
    require([(p['seed'], p['tick']) for p in points] ==
            [(seed, tick) for seed in ROLE_SEEDS[role] for tick in (110, 260, 410)], 'incomplete ordered grid: ' + role)


def validate_report(source):
    require(source['schema'] == 'd1-yaw-weight72-result-v1'
            and source['status'] == source['verification_after'] == source['comparison']['status'] == 'COMPLETE',
            'complete verified72 evidence required')
    require(source['model_updates'] == 6000 and source['completed_model_fits'] == 6
            and source['counts']['policy_transitions'] == 4680, 'partial work cannot enter diagnostics')
    require({(m['seed'], m['arm']) for m in source['models']} ==
            {(s, a) for s in MODEL_SEEDS for a in ARMS} and len(source['models']) == 6, 'six complete paired models required')
    require(set(source['cost_references']) == set(ROLES[1:])
            and set(source['head_references']) == set(source['first_step_references']) == set(FIRST_ROLES),
            'registered physical reference roles missing')
    for group in ('cost_references', 'head_references', 'first_step_references'):
        for role, points in source[group].items():
            ordered(points, role)
    require(len(source['comparison']['models']) == 6
            and {(m['seed'], m['arm']) for m in source['comparison']['models']} ==
            {(s, a) for s in MODEL_SEEDS for a in ARMS}, 'complete original model decisions required')
    for model in source['models']:
        require(model['status'] == model['fit']['status'] == 'COMPLETE' and model['fit']['updates'] == 1000,
                'incomplete model')
        require(set(model['diagnostics']) == set(ROLES)
                and set(model['heads']) == set(model['first_step']) == set(FIRST_ROLES), 'registered roles missing')
        for group, roles in (('diagnostics', ROLES), ('heads', FIRST_ROLES), ('first_step', FIRST_ROLES)):
            for role in roles:
                ledger = model[group][role]
                require(ledger['status'] == 'COMPLETE' and ledger['completed_points'] == 9, 'partial point ledger')
                ordered(ledger['points'], role)


def summary(rows):
    groups = {}
    for row in rows:
        key = (row['model_seed'], row['arm'], row['kind'])
        current = groups.setdefault(key, {'rows': 0, 'reasons': collections.Counter(),
                                         'original_match_true': 0, 'original_match_false': 0, 'original_match_unknown': 0})
        current['rows'] += 1
        current['reasons'].update(row['diagnosis']['reasons'])
        original = row['diagnosis']['original_valid_match']
        current['original_match_true' if original is True else 'original_match_false' if original is False
                else 'original_match_unknown'] += 1
    return [{'model_seed': key[0], 'arm': key[1], 'kind': key[2], **value,
             'reasons': dict(value['reasons'])} for key, value in sorted(groups.items())]


def write_views(output, report):
    columns = ('model_seed', 'arm', 'role', 'seed', 'tick', 'kind', 'epsilon', 'model_fd',
               'reference_gradient', 'reference_resolved', 'original_valid_match', 'reasons', 'json_pointer')
    with (output / 'rows.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in report['rows']:
            writer.writerow({**{key: row[key] for key in columns if key not in ('original_valid_match', 'reasons')},
                'original_valid_match': row['diagnosis']['original_valid_match'],
                'reasons': '|'.join(row['diagnosis']['reasons'])})
    counts = collections.Counter(reason for row in report['rows'] for reason in row['diagnosis']['reasons'])
    text = '# Archived response diagnosis76\n\n'
    text += 'Descriptive readback only. Require COMPLETE status and verification_after in report.json before use. Original decisions are unchanged; no candidate is accepted by this report.\n\n'
    text += f"Rows: {len(report['rows'])}; model calls / physics / updates: 0 / 0 / 0.\n\n"
    text += '| Reason (overlapping) | Scalar rows |\n|---|---:|\n'
    text += ''.join(f'| {name} | {count} |\n' for name, count in sorted(counts.items()))
    text += '\nEach CSV row links to its original JSON pointer. Full selfchecks, raw signs, original gates, response gain and endpoint precision remain in report.json.\n'
    text += '\nA tolerance pass does not prove relative accuracy or physical agreement. An endpoint-neighbor interval is conditional on stored F64 values, not a full forward-error bound. Missing yaw JVPs and missing H32 perturbed endpoints stay missing.\n'
    with (output / 'README.md').open('x') as stream:
        stream.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    consumed = {}
    source_code = {str(p.relative_to(ROOT)): sha(p) for p in
                   (Path(__file__), ROOT / 'tools76/precision76.py', ROOT / 'tools76/reasons76.py', ROOT / 'tools74/runtime74.py')}
    started = time.process_time()

    def consume(name, manifest=None):
        path = ROOT / name
        identity = {'bytes': path.stat().st_size, 'sha256': sha(path)}
        if manifest is not None:
            require(manifest.get(name) == identity, 'published raw input identity differs: ' + name)
        consumed[name] = identity
        return path

    def body(report):
        published = json.loads(consume(PUBLISHED).read_text())
        manifest = published['raw_manifest']
        source = json.loads(consume(ARCHIVE + '/report.json', manifest).read_text())
        require(published['schema'] == 'd1-yaw-weight72-published-v1' and published['formal'] == source,
                'raw report differs from complete published evidence')
        validate_report(source)
        report.update(schema='d1-response-diagnosis76-v1', rows=[], consumed_inputs=consumed,
            source_code_sha256=source_code, original_decisions_changed=False,
            source_candidate_status=source['comparison']['candidate_status'],
            physical_references={'cost_h32': source['cost_references'], 'heads': source['head_references'],
                                 'yaw': source['first_step_references']},
            description='Read-only standard-library archived log diagnosis; no new model experiment',
            limitations=['conditional yaw JVP/selfcheck not recorded', 'H32 perturbed model endpoints not recorded',
                         'one-neighbor endpoint sensitivity is not a full computation error bound',
                         'physical unknowns remain unknown; no epsilon or seed selection'])
        comparisons = {(m['seed'], m['arm']): m for m in source['comparison']['models']}

        def add(model, role, point, kind, epsilon, fd, ref, resolved, checks, original, pointer, precision=None):
            require(time.process_time() - started < 30., 'readback CPU budget exhausted')
            report['rows'].append({'model_seed': model['seed'], 'arm': model['arm'], 'role': role,
                'seed': point['seed'], 'tick': point['tick'], 'kind': kind, 'epsilon': epsilon,
                'model_fd': fd, 'reference_gradient': ref, 'reference_resolved': resolved,
                'diagnosis': explain_response(fd, ref, resolved, checks, original),
                'endpoint_precision': precision, 'json_pointer': pointer})

        for mi, model in enumerate(source['models']):
            comparison = comparisons[(model['seed'], model['arm'])]
            base = ARCHIVE + '/' + model['directory']
            for role in ROLES:
                for pi, point in enumerate(model['diagnostics'][role]['points']):
                    ref = source['cost_references'][role][pi] if role != 'development' else None
                    original = comparison['cost'][role]['points'][pi]['valid_match'] if ref else None
                    add(model, role, point, 'cost_h32', .001, point['model_central_difference'],
                        ref['gradient'] if ref else None, ref['resolved'] if ref else None, {'cost': check_fields(point)}, original,
                        f'/models/{mi}/diagnostics/{role}/points/{pi}')
            for role in FIRST_ROLES:
                for pi, point in enumerate(model['heads'][role]['points']):
                    href = source['head_references'][role][pi]
                    comp = comparison['heads'][role]['points'][pi]
                    h1 = point['cost_h1']
                    path = consume(f'{base}/cost_h1_{role}/point_{pi:02d}.npz', manifest)
                    with zipfile.ZipFile(path) as archive:
                        plus = f64_array(archive, 'plus_costs', (1,))[0]
                        minus = f64_array(archive, 'minus_costs', (1,))[0]
                    add(model, role, point, 'cost_h1', .001, h1['model_central_difference'],
                        href['cost_h1']['gradient'], href['cost_h1']['resolved'], {'cost': check_fields(h1)},
                        comp['cost_h1']['valid_match'], f'/models/{mi}/heads/{role}/points/{pi}/cost_h1',
                        endpoint_precision(plus, minus, .001, h1['model_central_difference']))
                    path = consume(f'{base}/heads_{role}/point_{pi:02d}.npz', manifest)
                    with zipfile.ZipFile(path) as archive:
                        for ci, channel in enumerate(('vx', 'vy')):
                            require([e['epsilon'] for e in point['body'][channel]['eps']] == list(EPS), 'body epsilon grid changed')
                            for ei, entry in enumerate(point['body'][channel]['eps']):
                                for field in ('fd', 'jvp', 'line_fd'):
                                    require(f64_array(archive, f'epsilon_{ei}_body_{field}', (1, 2))[ci] == entry['model_' + field],
                                            'body scalar cache differs from report')
                                plus = f64_array(archive, f'epsilon_{ei}_body_plus', (1, 2))[ci]
                                minus = f64_array(archive, f'epsilon_{ei}_body_minus', (1, 2))[ci]
                                ref = href['body'][channel]
                                require(entry['numeric_valid'] == all(entry[k]['autograd_agrees'] and entry[k]['direction_identifiable']
                                    for k in ('endpoint_selfcheck', 'line_selfcheck')), 'body composite gate differs')
                                add(model, role, point, 'body_' + channel, entry['epsilon'], entry['model_fd'],
                                    ref['eps'][ei]['central64'], ref['resolved'],
                                    {'endpoint': entry['endpoint_selfcheck'], 'line': entry['line_selfcheck']},
                                    comp['body'][channel]['eps'][ei]['valid_match'],
                                    f'/models/{mi}/heads/{role}/points/{pi}/body/{channel}/eps/{ei}',
                                    endpoint_precision(plus, minus, entry['epsilon'], entry['model_fd']))
                    path = consume(f'{base}/first_step_{role}/point_{pi:02d}.npz', manifest)
                    yaw = model['first_step'][role]['points'][pi]
                    ref = source['first_step_references'][role][pi]
                    with zipfile.ZipFile(path) as archive:
                        require([e['epsilon'] for e in yaw['eps']] == list(EPS), 'yaw epsilon grid changed')
                        for ei, entry in enumerate(yaw['eps']):
                            plus = f64_array(archive, f'epsilon_{ei}_model_next_plus', (1, 45))[28]
                            minus = f64_array(archive, f'epsilon_{ei}_model_next_minus', (1, 45))[28]
                            add(model, role, point, 'yaw', entry['epsilon'], entry['model_fd'],
                                ref['eps'][ei]['central64'], ref['resolved'], None,
                                comparison['first_step'][role]['points'][pi]['eps'][ei]['valid_match'],
                                f'/models/{mi}/first_step/{role}/points/{pi}/eps/{ei}',
                                endpoint_precision(plus, minus, entry['epsilon'], entry['model_fd']))
        require(len(report['rows']) == 3186, 'all3186 scalar records required')
        report['summary'] = summary(report['rows'])
        checks = [check for row in report['rows'] for check in row['diagnosis']['selfchecks'].values()]
        require(len(checks) == 4050, 'complete original4050 selfchecks required')
        report['selfcheck_summary'] = {'total': len(checks),
            'arithmetic_agreement_passed': sum(c['autograd_agrees'] for c in checks),
            'direction_qualification_failed': sum(not c['direction_identifiable'] for c in checks),
            'opposite_self_signs': sum(c['opposite_nonzero_sign'] is True for c in checks)}
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        write_views(args.output, report)

    def after():
        for name, identity in consumed.items():
            path = ROOT / name
            require(path.stat().st_size == identity['bytes'] and sha(path) == identity['sha256'], 'input changed: ' + name)
        require(all(sha(ROOT / name) == digest for name, digest in source_code.items()), 'diagnostic source changed')

    report = run_preserving(args.output, body, after)
    print(json.dumps({'status': report['status'], 'failure': report['failure'], 'rows': len(report.get('rows', [])),
                      'cpu_seconds': report['cpu_seconds'], 'model_calls': 0, 'physics_steps': 0, 'updates': 0}))
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
