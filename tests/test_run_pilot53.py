"""Selection and audit logic can be tested without numerical libraries or assets."""
import io
from types import SimpleNamespace

import pytest

from tools53.run_pilot53 import choose_proposal, make_step_recorder


def models():
    return [{'status': 'COMPLETE', 'development_gradients': [
        {'gradient': .1, 'autograd_agrees': True, 'direction_identifiable': True} for _ in range(9)],
        'test_gradients': [{'gradient': -1000.} for _ in range(9)]} for _ in range(3)]


def test_probe_gradient_values_cannot_select_or_reverse_theta():
    rows = models()
    first = choose_proposal(rows)
    for row in rows:
        row['test_gradients'] = [{'gradient': 1000.} for _ in range(9)]
    second = choose_proposal(rows)
    assert first['theta'] == second['theta'] == -.01
    assert not first['physical_probe_results_seen']


@pytest.mark.parametrize('kind', ['disagreement', 'weak', 'invalid', 'partial'])
def test_no_candidate_from_disagreement_weak_gradient_or_partial_model(kind):
    rows = models()
    if kind == 'partial':
        rows[-1]['status'] = 'INCOMPLETE'
        with pytest.raises(ValueError, match='complete'):
            choose_proposal(rows)
        return
    for point in rows[-1]['development_gradients']:
        if kind == 'disagreement':
            point['gradient'] = -.1
        elif kind == 'weak':
            point['direction_identifiable'] = False
        else:
            point['autograd_agrees'] = False
    assert choose_proposal(rows)['theta'] is None


def test_audit_retains_actual_command_and_termination_with_monotonic_work():
    import json
    stream = io.StringIO()
    env = SimpleNamespace(_episode_steps=5, counters=lambda: {'policy_transitions': 69})
    info = {'command_during_interval': [0., 0., -.2], 'applied_action': [0.] * 16,
            'fallen': False, 'illegal_body_contact': True, 'base_height_m': .35, 'tilt_rad': .3}
    record = make_step_recorder(stream, {'phase': 'physical_probes'})
    record(env, [.1] * 16, (None, -1., True, False, info))
    row = json.loads(stream.getvalue())
    assert row['terminated'] and row['illegal_body_contact'] and not row['fallen']
    assert row['episode_tick'] == 5 and row['counts']['policy_transitions'] == 69
    assert row['command_during_interval'] == [0., 0., -.2]
    assert row['issued_action'] != row['applied_action']
