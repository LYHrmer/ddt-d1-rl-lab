"""Recompute both gates from the published data on a clean clone, with zero integration."""
import gzip
import hashlib
import json
from pathlib import Path

from tools90.launch90 import PLAN, PLAN_SHA256
from tools90.readback90 import verify_frames, verify_support_scores


ROOT = Path(__file__).resolve().parents[1]


def event_rows(report, name):
    path = ROOT / f'reports/stand_wbc90_{name}.jsonl.gz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report[name + '_gzip_sha256']
    raw = gzip.decompress(path.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == report[name + '_sha256']
    return [json.loads(line) for line in raw.splitlines()]


def test_signed_stand_endpoints_in_all_three_pairs():
    plan = json.loads(PLAN.read_bytes())['stand']
    report = json.loads((ROOT/'reports/stand_wbc90.json').read_bytes())
    assert report['protocol_sha256'] == hashlib.sha256(PLAN.read_bytes()).hexdigest() == PLAN_SHA256
    runs = {(c['reset_seed'], c['arm']):c for c in report['stand']['cases']}
    observed = []
    for parent, seed, pair in zip(plan['parent_seeds'], plan['reset_seeds'],report['stand']['pairs'], strict=True):
        a, b = (runs[seed, arm] for arm in plan['arms'])
        assert a['initial_state_sha256'] == b['initial_state_sha256']
        assert a['parent_seed'] == b['parent_seed'] == parent
        for c in (a, b):
            assert c['pd_updates'] == 5*c['ticks'] and c['native_steps'] == 10*c['ticks']
            assert c['rollback_m'] == max(0., c['initial_world_x_m'] - c['world_x_at_1s_m'])
        delta = a['rollback_m'] - b['rollback_m']
        complete = all(c['ticks'] == plan['ticks_per_episode'] and not (c['fallen'] or c['illegal']) for c in (a,b))
        observed.append(complete and delta >= plan['min_rollback_reduction_m'])
        assert pair['rollback_reduction_m'] == delta and pair['passed'] == observed[-1]
    assert len(runs) == 6 and all(observed) == report['stand']['passed']
    plot = ROOT/'docs/assets/stand90.png'
    assert hashlib.sha256(plot.read_bytes()).hexdigest() == report['plot_sha256']


def test_all_registered_wbc_timestamps_targets_and_cold_scores():
    plan = json.loads(PLAN.read_bytes())['shadow']
    report = json.loads((ROOT/'reports/stand_wbc90.json').read_bytes())
    frames, truth = (event_rows(report, name) for name in ('frames','truth'))
    measured = verify_frames(frames, plan)
    assert measured == report['shadow_readback']
    for key in ('attempted','accepted','mechanical_pass','timing_pass','bypass_bitwise_equal'):
        assert measured[key] == report['shadow'][key]
    support = verify_support_scores(frames, truth, report['truth_base_quats_wxyz'])
    for k, v in support.items():
        if isinstance(v,float):
            assert abs(v - report['support_readback'][k]) <= 1e-9
        else:
            assert v == report['support_readback'][k]
    audited = report['mechanics_readback']
    original = {(f['seed'], f['phase'], f['source_row']):f['enabled'] for f in frames}
    checked = set()
    passing = 0
    for row in audited['rows']:
        key = row['seed'], row['phase'], row['source_row']
        assert key not in checked and 'qdd' in original[key]
        checked.add(key)
        assert row['emitted'] == original[key]['accepted']
        residuals = row['violations'].copy()
        maximum = residuals.pop('max')
        assert maximum == max(residuals.values())
        within = maximum <= plan['hard_tolerance']
        assert within == row['within_original_hard_tolerance']
        passing += int(row['emitted'] and within)
    assert checked == {k for k, v in original.items() if 'qdd' in v}
    assert passing == audited['accepted_within_tolerance'] == report['independent_mechanical_gate']['passing_frames']
    assert report['independent_mechanical_gate']['passed'] == (passing == len(frames) == measured['bypass_bitwise_equal'])
    assert report['qp_solves_during_readback'] == report['native_steps_during_readback'] == 0
