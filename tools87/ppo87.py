"""One real, single-environment SB3 PPO rollout for the D1 whole-body actor."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))
from d1_rl.velocity_policy15 import VelocityActorCriticPolicy, VelocityMlpExtractor
from tools33.reference_shape33 import YAW_COLUMNS
from tools52.adapter52 import COMPLETE25_SHA256, load_actor

PARENT = ROOT / "evidence/low_memory_delay25/execution_01/random/train/policy.zip"
ANCHOR = ROOT / "evidence/cpp_reference34/execution_01/formal/forward_stop_delay1_seed1001_cpp_oracle.npz"
ANCHOR_SHA = "2894bee232190bd5da2f02526b368bb23176d7e6a036d903f7d2ff4e54b5c506"
ACTOR, RANGES, ACTIONS, STEPS = 627, 16, 16, 4096
WHEELS = (3, 7, 11, 15)
TERRAINS = ("flat", "up", "down")
SEEDS = (87011, 87023, 87037)
ARMS = ("masked", "range")
PPO_KWARGS = dict(n_steps=STEPS, batch_size=256, n_epochs=4, learning_rate=1e-4,
                  gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=0.,
                  vf_coef=.5, max_grad_norm=.5, target_kl=.005)
PROJECTION = "mlp_extractor.range_projection.weight"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_arrays(path, arrays):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def single_cpu():
    require(sb3.__version__ == "2.7.0", "the checkpoint requires SB3 2.7.0")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def action_space():
    return spaces.Box(-1., 1., (ACTIONS,), np.float32)


def observation_space():
    return spaces.Box(-np.inf, np.inf, (ACTOR + RANGES,), np.float32)


class RangeMlp(VelocityMlpExtractor):
    """Only the actor sees range features. The original critic still sees 627 floats."""

    def __init__(self):
        super().__init__()
        self.range_projection = nn.Linear(RANGES, self.latent_dim_pi, bias=False)
        nn.init.zeros_(self.range_projection.weight)

    def forward_actor(self, features):
        return (super().forward_actor(features[:, :ACTOR])
                + self.range_projection(features[:, ACTOR:]))

    def forward_critic(self, features):
        return super().forward_critic(features[:, :ACTOR])


class FullBodyPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = RangeMlp().to(self.device)


POLICY_KWARGS = dict(ortho_init=False, log_std_init=-1., normalize_images=False)


def weight_member(path, name):
    with zipfile.ZipFile(path) as archive:
        require(archive.namelist().count(name) == 1, f"expected one {name}")
        return torch.load(io.BytesIO(archive.read(name)), map_location="cpu", weights_only=True)


def verify_parent():
    receipt = json.loads((PARENT.parent / "result.json").read_text())
    require(not PARENT.is_symlink() and sha(PARENT) == COMPLETE25_SHA256
            and receipt["status"] == "COMPLETE" and receipt["checkpoint_sha256"] == COMPLETE25_SHA256
            and receipt["eligible_for_evaluation"] and receipt["cleanup_complete"], "COMPLETE25 identity")
    require(receipt["hyperparameters"] == dict(PPO_KWARGS, device="cpu", cpu_threads=1,
                                               aux_coef=0., log_std_init=-1.), "25 PPO settings")


def warm_start(policy):
    """Restore every original weight and its Adam slots, matching by parameter name."""
    verify_parent()
    native_space = spaces.Dict({
        "actor": spaces.Box(-np.inf, np.inf, (ACTOR,), np.float32),
        "velocity_target": spaces.Box(-np.inf, np.inf, (3,), np.float32),
    })
    with torch.random.fork_rng(devices=[]):
        original = VelocityActorCriticPolicy(native_space, action_space(), lambda _: 1e-4)
    weights = weight_member(PARENT, "policy.pth")
    original.load_state_dict(weights, strict=True)
    original.optimizer.load_state_dict(weight_member(PARENT, "policy.optimizer.pth"))

    missing, unexpected = policy.load_state_dict(weights, strict=False)
    require(missing == [PROJECTION] and not unexpected, "full actor/critic transfer mismatch")
    require(not bool(torch.count_nonzero(policy.mlp_extractor.range_projection.weight)), "range init")
    previous = dict(original.named_parameters())
    current = dict(policy.named_parameters())
    require(set(current) == set(previous) | {PROJECTION}, "unexpected new parameter")
    old_optimizer, new_optimizer = original.optimizer, policy.optimizer
    require(type(old_optimizer) is type(new_optimizer)
            and len(old_optimizer.param_groups) == len(new_optimizer.param_groups) == 1,
            "expected a single original Adam parameter group")
    require(len(old_optimizer.state) == len(previous) and not new_optimizer.state, "missing Adam slots")
    group = old_optimizer.param_groups[0]
    for key, value in group.items():
        if key != "params":
            new_optimizer.param_groups[0][key] = copy.deepcopy(value)
    require(new_optimizer.param_groups[0]["lr"] == PPO_KWARGS["learning_rate"], "Adam learning rate")
    for name, parameter in previous.items():
        slots = old_optimizer.state[parameter]
        require(torch.equal(current[name], parameter) and {"step", "exp_avg", "exp_avg_sq"} <= set(slots)
                and all(bool(torch.isfinite(v).all()) for v in slots.values() if torch.is_tensor(v)),
                f"invalid inherited state for {name}")
        new_optimizer.state[current[name]] = copy.deepcopy(slots)
    return {"parent_sha256": COMPLETE25_SHA256, "inherited_parameter_tensors": len(previous),
            "inherited_adam_parameters": len(new_optimizer.state),
            "new_projection_parameters": int(current[PROJECTION].numel())}


def actor_input(actor, features, arm):
    require(arm in ARMS, "unknown perception arm")
    x = np.asarray(actor, dtype=np.float32)
    z = np.asarray(features, dtype=np.float32)
    require(x.shape == (ACTOR,) and z.shape == (RANGES,)
            and np.isfinite(x).all() and np.isfinite(z).all(), "expected finite 627+16 input")
    require(np.all(x[YAW_COLUMNS] == 0), "straight-line scope requires zero yaw command history")
    return np.concatenate((x, z if arm == "range" else np.zeros_like(z)))


def predictor(policy, arm):
    policy.eval()

    @torch.no_grad()
    def predict(actor, features):
        obs = torch.from_numpy(actor_input(actor, features, arm)).unsqueeze(0)
        action = policy._predict(obs, deterministic=True).clamp(-1., 1.)[0].numpy().copy()
        require(np.isfinite(action).all(), "nonfinite deterministic action")
        return action

    return predict


def parent_predictor():
    single_cpu()
    verify_parent()
    actor = load_actor(PARENT)

    @torch.no_grad()
    def predict(proprioception, features):
        x = actor_input(proprioception, features, "masked")[:ACTOR]
        return actor(torch.from_numpy(x)[None])[0].numpy().copy()
    return predict


def load_weights(path):
    with torch.random.fork_rng(devices=[]):
        policy = FullBodyPolicy(observation_space(), action_space(), lambda _: 1e-4, **POLICY_KWARGS)
    policy.load_state_dict(weight_member(path, "policy.pth"), strict=True)
    return policy


def load_predictor(arm_dir):
    single_cpu()
    arm_dir = Path(arm_dir)
    report = json.loads((arm_dir / "report.json").read_text())
    path = arm_dir / "policy.zip"
    require(report["status"] == "COMPLETE" and report["arm"] in ARMS
            and not path.is_symlink() and sha(path) == report["policy_sha256"], "unqualified checkpoint")
    return predictor(load_weights(path), report["arm"])


def initial_probes(policy):
    """Use archived straight-line C++ outputs and valid/missing range patterns."""
    require(sha(ANCHOR) == ANCHOR_SHA, "C++34 anchor identity")
    ticks = [0, 50, 100, 140, 225, 399, 450, 599]
    with np.load(ANCHOR, allow_pickle=False) as saved:
        actors = saved["raw_actor627"][ticks]
        expected = saved["cpp_issued16"][ticks]
    raw_range = np.tile(np.r_[np.linspace(.1, .8, 8), np.ones(8)].astype(np.float32), (len(ticks), 1))
    raw_range[::2] = 0
    enabled, disabled = predictor(policy, "range"), predictor(policy, "masked")
    actions = np.stack([enabled(x, z) for x, z in zip(actors, raw_range)])
    blind = np.stack([disabled(x, z) for x, z in zip(actors, raw_range)])
    error = float(np.abs(actions - expected).max())
    require(np.array_equal(actions, blind) and error <= 1e-5, "initial action differs from C++34")
    return {"ticks": ticks, "anchor_sha256": ANCHOR_SHA, "max_abs_error_cpp34": error,
            "initial_masked_range_bitwise_equal": True}


def preflight():
    """Zero resets, integrations and optimizer updates."""
    single_cpu()
    with torch.random.fork_rng(devices=[]):
        policy = FullBodyPolicy(observation_space(), action_space(), lambda _: 1e-4, **POLICY_KWARGS)
    return {**warm_start(policy), **initial_probes(policy)}


class TrainingTrace(gym.Wrapper):
    """Record the measurements and motor requests that actually drove MuJoCo."""

    def __init__(self, env, arm, poll):
        super().__init__(env)
        self.arm = arm
        self.poll = poll
        self.obs = None
        self.proprio_digest = hashlib.sha256()
        self.rows = {key: [] for key in ("raw_range", "visible_range", "issued", "applied", "qpos",
                                         "reward", "terrain", "terminated", "truncated")}
        self.probes = []

    def reset(self, **kwargs):
        self.obs, info = self.env.reset(**kwargs)
        return self.obs, info

    def step(self, action):
        n = len(self.rows["reward"])
        if self.poll is not None and n % 64 == 0:
            self.poll()
        base = self.env.unwrapped
        x, visible = self.obs[:ACTOR], self.obs[ACTOR:].copy()
        raw = np.array(base.last_raw_features, dtype=np.float32, copy=True)
        require(np.array_equal(self.obs, actor_input(x, raw, self.arm)), "policy/sensor inputs differ")
        self.proprio_digest.update(x.tobytes())
        if (n % 512 == 0 or (len(self.probes) < 4 and np.any(raw[8:]))) and len(self.probes) < 16:
            self.probes.append((x.copy(), raw))
        terrain = TERRAINS.index(base.current_terrain)
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        require(np.array_equal(base._last_policy_action, action), "issued motor targets mismatch")
        for key, value in (("raw_range", raw), ("visible_range", visible),
                           ("issued", base._last_policy_action.copy()), ("applied", base._applied_action.copy()),
                           ("qpos", base.data.qpos.copy()), ("reward", float(reward)), ("terrain", terrain),
                           ("terminated", bool(terminated)), ("truncated", bool(truncated))):
            self.rows[key].append(value)
        return self.obs, reward, terminated, truncated, info

    def archive(self, output):
        if not self.rows["reward"]:
            return {"completed_recorded_ticks": 0}
        arrays = {key: np.asarray(value) for key, value in self.rows.items()}
        save_arrays(output / "train_trace.npz", arrays)
        return {
            "completed_recorded_ticks": len(arrays["reward"]),
            "proprio627_sha256": self.proprio_digest.hexdigest(),
            "array_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()},
            "valid_raw_range_ticks": int(np.any(arrays["raw_range"][:, 8:] != 0, axis=1).sum()),
            "valid_actor_range_ticks": int(np.any(arrays["visible_range"][:, 8:] != 0, axis=1).sum()),
            "steps_per_terrain": dict(zip(TERRAINS, np.bincount(arrays["terrain"], minlength=3).tolist())),
            "terminated_episodes": int(arrays["terminated"].sum()),
            "six_second_truncations": int(arrays["truncated"].sum()),
        }


def train(arm, seed, output, *, poll=None):
    """Train one registered arm once. The caller supplies resource supervision."""
    from tools87.terrain87 import make_env

    require(arm in ARMS and type(seed) is int and seed in SEEDS, "unregistered arm or training seed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(arm=arm, seed=seed, origin="full_body_joint_pd_87", status="PARTIAL",
                  budget=STEPS, hyperparameters=PPO_KWARGS, early_kl_allowed=True)
    write_json(output / "protocol.json", report)
    env = trace = model = trace_report = None
    optimizer_steps = 0
    gradient_max = {"projection": 0., "action_head": 0.}
    try:
        single_cpu()
        if poll is not None:
            poll()
        env = make_env("cycle", seed, perception=(arm == "range"), max_transitions=STEPS)
        trace = TrainingTrace(env, arm, poll)
        model = PPO(FullBodyPolicy, trace, policy_kwargs=POLICY_KWARGS, seed=seed,
                    device="cpu", **PPO_KWARGS)
        policy = model.policy
        report["warm_start"] = warm_start(policy)
        report["initial_probes"] = initial_probes(policy)
        before_head = policy.action_net.weight.detach().clone()
        before_bias = policy.action_net.bias.detach().clone()
        before_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}

        def after_step(optimizer, args, kwargs):
            nonlocal optimizer_steps
            optimizer_steps += 1
            for key, parameter in (("projection", policy.mlp_extractor.range_projection.weight),
                                   ("action_head", policy.action_net.weight)):
                norm = float(torch.linalg.vector_norm(parameter.grad))
                require(np.isfinite(norm), "nonfinite PPO gradient")
                gradient_max[key] = max(norm, gradient_max[key])

        with policy.optimizer.register_step_post_hook(after_step):
            model.learn(total_timesteps=STEPS)
        counts = env.get_counts()
        require(model.num_timesteps == STEPS and counts["policy_transitions"] == STEPS
                and counts["native_steps"] == 10 * STEPS and counts["pd_updates"] == 5 * STEPS,
                "incomplete real PPO rollout")
        require(0 < optimizer_steps <= PPO_KWARGS["n_epochs"] * STEPS // PPO_KWARGS["batch_size"],
                "missing or excessive Adam updates")
        delta = policy.action_net.weight.detach() - before_head
        delta_bias = policy.action_net.bias.detach() - before_bias
        rows = torch.sqrt(delta.square().sum(1) + delta_bias.square())
        projection_norm = float(policy.mlp_extractor.range_projection.weight.detach().norm())
        changed = {k: float((v - before_state[k]).norm()) for k, v in policy.state_dict().items()}
        require(all(np.isfinite(list(changed.values()))) and bool(torch.any(rows > 0)),
                "nonfinite or unchanged whole-body actor")
        require((projection_norm > 0) == (arm == "range")
                and (gradient_max["projection"] > 0) == (arm == "range"), "unexpected range update")
        model.save(output / "policy.zip")
        report["policy_sha256"] = sha(output / "policy.zip")
        final_predictor = predictor(policy, arm)
        reloaded = predictor(load_weights(output / "policy.zip"), arm)
        actors, sensed = (np.stack(values) for values in zip(*trace.probes))
        actions = np.stack([final_predictor(x, z) for x, z in zip(actors, sensed)])
        replayed = np.stack([reloaded(x, z) for x, z in zip(actors, sensed)])
        masked = np.stack([final_predictor(x, np.zeros_like(z)) for x, z in zip(actors, sensed)])
        original = parent_predictor()
        original_actions = np.stack([original(x, z) for x, z in zip(actors, sensed)])
        delta_action = float(np.abs(actions - original_actions).max())
        require(np.array_equal(actions, replayed) and delta_action > 0., "reload or policy-change probe")
        save_arrays(output / "probes.npz", dict(actor=actors, raw_range=sensed, parent=original_actions,
                                                actions=actions, masked=masked, reloaded=replayed))
        trace_report = trace.archive(output)
        require(trace_report["valid_raw_range_ticks"] > 0
                and trace_report["valid_actor_range_ticks"] == (trace_report["valid_raw_range_ticks"]
                                                               if arm == "range" else 0),
                "sensor coverage check failed")
        report.update(status="COMPLETE", postclip_max_gradient_norm=gradient_max,
                      changed_parameter_l2=changed, action_head_row_delta_l2=rows.tolist(),
                      leg_rows_changed=sum(float(rows[i]) > 0 for i in range(ACTIONS) if i not in WHEELS),
                      wheel_rows_changed=sum(float(rows[i]) > 0 for i in WHEELS),
                      reload_probe_actions_bitwise_equal=True, max_action_change_from_parent=delta_action,
                      max_measured_vs_masked_action=float(np.abs(actions - masked).max()),
                      sb3_train_log={k: float(v) for k, v in model.logger.name_to_value.items()
                                     if k.startswith("train/") and np.isfinite(v)})
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if model is not None and optimizer_steps and not (output / "policy.zip").exists():
            try:
                model.save(output / "partial_policy.zip")
                report["partial_policy_sha256"] = sha(output / "partial_policy.zip")
            except Exception as save_error:
                report["partial_save_error"] = str(save_error)
        raise
    finally:
        report["optimizer_steps"] = optimizer_steps
        if env is not None:
            report["counts"] = env.get_counts()
            env.close()
        if trace is not None:
            if trace_report is None and not (output / "train_trace.npz").exists():
                trace_report = trace.archive(output)
            report["trace"] = trace_report
        write_json(output / "report.json", report)
    return report
