"""Finite endpoint interpretation, with deterministic synthetic arrays only."""
import pytest

from tools81 import diagnostics81 as d


def test_fixed_fd_retains_wrong_side_and_zero_without_floor_or_step_selection():
    values = d.finite_response(1., 1.1, 1.2, .009)
    assert values['central'] < 0 and values['forward'] > 0
    assert not values['strong_monotone']
    tiny = d.finite_response(0., -1e-20, 1e-20, .001125)
    assert tiny['strong_monotone'] and tiny['central'] < 0
    zero = d.finite_response(1., 1., 1., .009)
    assert zero['stored_plus_equals_zero'] and not zero['strong_monotone']


def test_cross_scoring_uses_raw_yaw_and_post_state_same_tick_commands():
    np = pytest.importorskip('numpy')
    arrays = {'body_wz': np.full(32, .3), 'commands': np.tile([0., 0., .2], (32, 1)),
              'states': np.zeros((32, 45)), 'costs': np.arange(32, dtype=float),
              'cost_components': np.ones((32, 6)), 'issued_actions': np.zeros((32, 16))}
    result = d.trace_metrics(arrays)
    assert result['yaw_mae'] == pytest.approx(.1)
    assert result['prefix_metrics']['8']['cost'] == 3.5
    assert result['prefix_metrics']['32']['yaw_mae'] == pytest.approx(.1)


def test_max_mae_and_cost_are_cross_scored_without_confusing_aggregation():
    rows = [{'tick': t, 'prefix_metrics': {'32': {'cost': c, 'yaw_mae': y}}}
            for t, c, y in [(175, 10., .01), (325, 2., .3)]]
    assert d._pair(rows, 32) == {'C': 6., 'Q': .3, 'max_ticks': [325]}


def test_v_ad_shares_one_rollout_graph_and_only_projects_existing_directions():
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace

    anchor = torch.nn.Module()
    anchor.action_head = torch.nn.Linear(128, 16).eval().requires_grad_(False)
    calls = []
    def values(_model, actor, points, **kwargs):
        calls.append(points)
        w, b = actor.action_head.weight, actor.action_head.bias
        costs = [b[3]*2 + w.sum()*0 for _ in range(6)]
        qs = torch.stack([b[7]*3 + w.sum()*0 for _ in range(3)])
        return {'points': [{'seed': s, 'tick': t, 'cost': costs[i]}
                for i, (s, t) in enumerate((s, t) for s in (81207, 81208, 81209) for t in (175, 325))],
                'per_seed_q': qs}
    directions = {}
    for arm, index in [('C6', 3), ('Q6', 7)]:
        weight, bias = torch.zeros((16, 128), dtype=torch.float64), torch.zeros(16, dtype=torch.float64)
        bias[index] = -1
        directions[arm] = weight, bias
    rows = d.validation_ad(SimpleNamespace(), anchor, ['single-shared-input'], directions, 64001, values, 100.)
    assert len(calls) == 1 and len(rows) == 12
    for row in rows:
        expected = -2. if (row['metric'], row['arm']) == ('C', 'C6') else (
            -3. if (row['metric'], row['arm']) == ('Q', 'Q6') else 0.)
        assert row['model_directional_ad'] == expected and row['physical_ad'] is None
        assert row['V_used_for_updates'] is False
