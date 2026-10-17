"""Exercise the actual delay18 class with a simulator-free parent seam."""
import ast
import copy
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "training/d1_rl/delay_env18.py"


class Fake03d:
    """Only the inherited reset's random-stream and episode-index boundary."""

    def __init__(self, config=None, *, arm="B", seed=0, rank=0, eval_delay_ticks=None, **kwargs):
        if kwargs:
            raise TypeError("unknown original task option")
        self.config = config
        self.arm, self.base_seed, self.rank = arm, int(seed), int(rank)
        self.eval_delay_ticks = eval_delay_ticks
        self._episode_index = 0
        self.observation_marker = object()
        self.reward_marker = object()

    def _episode_rng(self, stream):
        return np.random.default_rng(np.random.SeedSequence(
            [self.base_seed, self.rank, self._episode_index, stream]))

    def reset(self, *, seed=None):
        if seed is not None:
            self.base_seed, self._episode_index = int(seed), 0
        self._reset_rng = self._episode_rng(0)
        self._command_rng = self._episode_rng(1)
        self._delay_rng = self._episode_rng(2)
        self._episode_index += 1
        self._delay_ticks = self._pick_delay()
        return self._actor_observation(), {"delay_ticks": self._delay_ticks}

    def _actor_observation(self):
        return self.observation_marker

    def _reward(self):
        return self.reward_marker

    def step(self, action):
        return action


@pytest.fixture
def delay_class():
    # Compile the production class and its constants, skipping imports of the
    # real task package. This neither imports MuJoCo nor builds a robot model.
    tree = ast.parse(SOURCE.read_text())
    body = [node for node in tree.body if isinstance(node, (ast.Assign, ast.ClassDef))]
    namespace = {"np": np, "D1Task03dEnv": Fake03d}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["D1Delay18Env"]


def draw(env, count):
    return [env.reset()[1]["delay_ticks"] for _ in range(count)]


def test_production_parent_reset_index_and_stream_boundary_matches_fake():
    tree = ast.parse((ROOT / "training/d1_rl/env.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "D1Env")
    reset = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "reset")
    text = ast.unparse(reset)
    assert text.index("self._reset_rng = self._episode_rng(0)") < text.index("self._episode_index += 1")
    assert text.index("self._command_rng = self._episode_rng(1)") < text.index("self._episode_index += 1")
    assert text.index("self._delay_rng = self._episode_rng(2)") < text.index("self._episode_index += 1")
    assert text.index("self._episode_index += 1") < text.index("self._delay_ticks = self._pick_delay()")
    episode_rng = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_episode_rng")
    assert ast.dump(episode_rng.body[0].value, include_attributes=False) == ast.dump(
        ast.parse("np.random.default_rng(np.random.SeedSequence([self.base_seed, self.rank, self._episode_index, stream]))").body[0].value,
        include_attributes=False)


def test_only_delay_selection_overrides_parent_behavior(delay_class):
    methods = {name for name in delay_class.__dict__ if callable(delay_class.__dict__[name])}
    assert methods == {"__init__", "_pick_delay", "get_delay_state18"}
    for method in ("reset", "step", "_actor_observation", "_reward", "_episode_rng"):
        assert getattr(delay_class, method) is getattr(Fake03d, method)
    marker = {"unchanged": object()}
    env = delay_class(config=marker, training_delay_mode="uniform_0_3")
    assert env.config is marker
    assert env.reset()[0] is env.observation_marker
    assert env._reward() is env.reward_marker


@pytest.mark.parametrize("mode", [None, True, 0, "R", "uniform", "uniform_0_2", "fixed_3", "ZERO", []])
def test_unregistered_mode_is_rejected_before_parent_constructor(delay_class, monkeypatch, mode):
    def forbidden(*args, **kwargs):
        raise AssertionError("parent constructor must not be entered")
    monkeypatch.setattr(Fake03d, "__init__", forbidden)
    with pytest.raises(ValueError, match="training_delay_mode"):
        delay_class(training_delay_mode=mode)


@pytest.mark.parametrize("arm", ["R", "M", None])
def test_unregistered_original_arm_cannot_change_another_variable(delay_class, arm):
    with pytest.raises(ValueError, match="original B arm"):
        delay_class(training_delay_mode="zero", arm=arm)


@pytest.mark.parametrize("fixed", [-1, 4, 1., True, "1", np.int64(1)])
def test_fixed_eval_requires_registered_python_integer(delay_class, fixed):
    with pytest.raises(ValueError, match="Python int"):
        delay_class(training_delay_mode="uniform_0_3", eval_delay_ticks=fixed)


@pytest.mark.parametrize("mode", ["zero", "uniform_0_3"])
@pytest.mark.parametrize("fixed", [0, 1, 2, 3])
def test_fixed_eval_precedes_training_distribution_and_counts(delay_class, mode, fixed):
    env = delay_class(training_delay_mode=mode, eval_delay_ticks=fixed, seed=101, rank=2)
    assert draw(env, 16) == [fixed] * 16
    state = env.get_delay_state18()
    assert state["delay_counts"] == {str(i): 16 if i == fixed else 0 for i in range(4)}
    assert state["last_selection"] == {"mode": mode, "seed": 101, "rank": 2,
                                       "episode_index": 15, "stream": 18018,
                                       "selected_delay_ticks": fixed, "selection": "fixed_eval"}


def test_zero_has_no_sampling_and_records_every_episode(delay_class, monkeypatch):
    env = delay_class(training_delay_mode="zero", seed=101)
    env._episode_index = 1
    monkeypatch.setattr(np.random, "default_rng", lambda *_: (_ for _ in ()).throw(AssertionError("no RNG")))
    assert env._pick_delay() == 0
    assert env.get_delay_state18()["delay_counts"] == {"0": 1, "1": 0, "2": 0, "3": 0}


def test_uniform_exact_seeded_stream_covers_both_endpoints(delay_class):
    env = delay_class(training_delay_mode="uniform_0_3", seed=101, rank=3)
    selected = draw(env, 128)
    expected = [int(np.random.default_rng(np.random.SeedSequence([101, 3, episode, 18018])).integers(0, 4))
                for episode in range(128)]
    assert selected == expected and set(selected) == {0, 1, 2, 3}
    assert env.get_delay_state18()["delay_counts"] == {str(i): selected.count(i) for i in range(4)}


@pytest.mark.parametrize("mode", ["zero", "uniform_0_3"])
def test_selection_and_reporting_do_not_advance_reset_command_or_old_delay_rng(delay_class, mode):
    env = delay_class(training_delay_mode=mode, seed=101, rank=1)
    for _ in range(12):
        env.reset()
        before = [copy.deepcopy(rng.bit_generator.state) for rng in (env._reset_rng, env._command_rng, env._delay_rng)]
        env.get_delay_state18()
        # The sampler's independent reconstruction must never touch any of the
        # three generators the inherited reset just created.
        env._pick_delay()
        after = [rng.bit_generator.state for rng in (env._reset_rng, env._command_rng, env._delay_rng)]
        assert before == after
        episode = env._episode_index - 1
        for stream, rng in enumerate((env._reset_rng, env._command_rng, env._delay_rng)):
            expected = np.random.default_rng(np.random.SeedSequence([101, 1, episode, stream])).random(8)
            assert np.array_equal(rng.random(8), expected)


def test_two_modes_leave_original_rng_draws_identical(delay_class):
    control = delay_class(training_delay_mode="zero", seed=101, rank=0)
    delayed = delay_class(training_delay_mode="uniform_0_3", seed=101, rank=0)
    for _ in range(24):
        control.reset(); delayed.reset()
        for name in ("_reset_rng", "_command_rng", "_delay_rng"):
            assert np.array_equal(getattr(control, name).random(12), getattr(delayed, name).random(12))


def test_seeded_reset_repeats_delay_sequence_and_lifetime_counts_remain(delay_class):
    env = delay_class(training_delay_mode="uniform_0_3", seed=101, rank=2)
    before = draw(env, 24)
    repeated = [env.reset(seed=101)[1]["delay_ticks"], *draw(env, 23)]
    assert repeated == before
    assert sum(env.get_delay_state18()["delay_counts"].values()) == 48
    assert env.get_delay_state18()["last_selection"]["episode_index"] == 23


def test_rank_and_seed_are_independent_sampling_identity(delay_class):
    base = draw(delay_class(training_delay_mode="uniform_0_3", seed=101, rank=0), 64)
    other_rank = draw(delay_class(training_delay_mode="uniform_0_3", seed=101, rank=1), 64)
    other_seed = draw(delay_class(training_delay_mode="uniform_0_3", seed=102, rank=0), 64)
    assert base != other_rank and base != other_seed
    assert base == draw(delay_class(training_delay_mode="uniform_0_3", seed=101, rank=0), 64)


def test_reporting_returns_copies_and_never_selects_a_delay(delay_class):
    env = delay_class(training_delay_mode="uniform_0_3")
    assert env.get_delay_state18() == {"last_selection": None, "delay_counts": dict.fromkeys(("0", "1", "2", "3"), 0)}
    env.reset()
    expected = env.get_delay_state18()
    copied = env.get_delay_state18()
    copied["delay_counts"]["0"] = 999
    copied["last_selection"]["rank"] = 999
    assert env.get_delay_state18() == expected


def test_bad_runtime_eval_value_is_rejected_before_delay_count(delay_class):
    env = delay_class(training_delay_mode="zero")
    env.eval_delay_ticks = True
    with pytest.raises(ValueError, match="Python int"):
        env.reset()
    assert sum(env.get_delay_state18()["delay_counts"].values()) == 0


def test_selection_before_first_reset_is_rejected(delay_class):
    with pytest.raises(RuntimeError, match="inherited reset"):
        delay_class(training_delay_mode="uniform_0_3")._pick_delay()
