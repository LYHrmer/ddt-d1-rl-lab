"""One matched, native 4,096-step perceptive PPO update from a completed 87 policy."""
from __future__ import annotations

import hashlib
import signal
from pathlib import Path

import numpy as np
import torch

from tools87 import ppo87 as pilot
from tools89.common89 import PLAN, parent_folder, read, registered_plan


def parent_checkpoint(seed, plan):
    folder = parent_folder(seed, plan)
    report = read(folder / "report.json")
    path = folder / "policy.zip"
    pilot.require(report["counts"]["policy_transitions"] == pilot.STEPS and report["optimizer_steps"] > 0
                  and report["reload_probe_actions_bitwise_equal"] and report["hyperparameters"] == pilot.PPO_KWARGS
                  and not path.is_symlink(), "parent PPO training contract")
    return path


def inherit_state(policy, path):
    """Restore the exact policy and optimizer groups/slots of the same architecture."""
    weights = pilot.weight_member(path, "policy.pth")
    policy.load_state_dict(weights, strict=True)
    for name, tensor in policy.state_dict().items():
        pilot.require(torch.equal(tensor, weights[name]) and torch.isfinite(tensor).all(), f"parent parameter {name}")
    source = pilot.weight_member(path, "policy.optimizer.pth")
    optimizer = policy.optimizer
    optimizer.load_state_dict(source)
    restored = optimizer.state_dict()
    pilot.require(len(source["param_groups"]) == 1 and restored["param_groups"] == source["param_groups"]
                  and source["param_groups"][0]["lr"] == pilot.PPO_KWARGS["learning_rate"], "Adam group mismatch")
    ids = restored["param_groups"][0]["params"]
    params = list(policy.parameters())
    pilot.require(len(ids) == len(params) and set(ids) == set(source["state"]) == set(restored["state"]),
                  "missing Adam parameter slots")
    for i, parameter in zip(ids, params, strict=True):
        old, new = source["state"][i], restored["state"][i]
        pilot.require(new.keys() == old.keys() and {"step", "exp_avg", "exp_avg_sq"} <= new.keys()
                      and new["exp_avg"].shape == new["exp_avg_sq"].shape == parameter.shape
                      and all(torch.equal(v, old[k]) and torch.isfinite(v).all() for k, v in new.items()),
                      f"Adam values changed for parameter {i}")
    return {"parameter_tensors": len(weights), "adam_parameters": len(ids),
            "inherited_adam_steps": {k: int(optimizer.state[p]["step"].item()) for k, p in policy.named_parameters()}}


class CurriculumTrace(pilot.TrainingTrace):
    """Extend the frozen recorder with the actual motor history and wheel-support evidence."""

    def __init__(self, env):
        super().__init__(env, "range", None)
        self.rows.update({k: [] for k in ("proprio627", "qvel", "last_torque_nm", "loaded_wheels_past_toe",
                         "episode_tick", "start_x_m", "slope_start", "base_origin_past_toe", "command_vx_m_s",
                         "body_vx_m_s", "fallen", "illegal_contact")})

    def step(self, action):
        proprio = self.obs[:pilot.ACTOR].copy()
        obs, reward, done, truncated, info = super().step(action)
        values = {
            "proprio627": proprio, "qvel": self.env.unwrapped.data.qvel.copy(),
            "last_torque_nm": np.asarray(info["last_torque_nm"]), "loaded_wheels_past_toe": info["loaded_wheels_past_toe"],
            "episode_tick": info["episode_policy_tick"], "start_x_m": info["start_x_m"], "slope_start": info["slope_start"],
            "base_origin_past_toe": info["base_origin_past_toe"], "command_vx_m_s": info["command_during_interval"][0],
            "body_vx_m_s": info["body_vx_m_s"], "fallen": info["fallen"], "illegal_contact": info["illegal_body_contact"],
        }
        for k, v in values.items():
            self.rows[k].append(v)
        return obs, reward, done, truncated, info


def coverage(rows):
    terrain, command, support, ticks, slope = (np.asarray(rows[k]) for k in
        ("terrain", "command_vx_m_s", "loaded_wheels_past_toe", "episode_tick", "slope_start"))
    n = len(terrain)
    pilot.require(n and support.shape == (n, 4) and support.dtype == bool and len(command) == len(ticks) == n,
                  "wheel support or command alignment")
    uphill = terrain == pilot.TERRAINS.index("up")
    moving = uphill & (command > 0)
    both = support[:, :2].any(axis=1) & support[:, 2:].any(axis=1)
    all_four = support.all(axis=1)
    base = np.asarray(rows["base_origin_past_toe"])
    front = int((moving & support[:, :2].any(axis=1)).sum())
    rear = int((moving & support[:, 2:].any(axis=1)).sum())
    starts = np.flatnonzero(ticks == 1)
    pilot.require(len(starts) and starts[0] == 0 and not support[~uphill].any(), "episode boundary or terrain support")
    episodes = []
    for a, b in zip(starts, [*starts[1:], n], strict=True):
        section = slice(a, b)
        pilot.require(np.array_equal(ticks[section], np.arange(1, b-a+1))
                      and np.all(terrain[section] == terrain[a]) and np.all(slope[section] == slope[a]),
                      "episode clock, start pose or terrain changed")
        if uphill[a]:
            epochs = (moving & all_four)[section]
            episodes.append({"first_rollout_row": int(a), "ticks": int(b-a), "slope_start": bool(slope[a]),
                             "start_x_m": float(rows["start_x_m"][a]), "moving_ticks": int(moving[section].sum()),
                             "four_loaded_moving_ticks": int(epochs.sum()),
                             "terminated": bool(rows["terminated"][b-1]), "truncated": bool(rows["truncated"][b-1])})
    denominator, numerator = int(moving.sum()), int((moving & all_four).sum())
    return {"uphill_ticks": int(uphill.sum()), "uphill_moving_ticks": denominator,
            "uphill_base_past_toe_ticks": int((uphill & base).sum()),
            "uphill_moving_base_past_toe_ticks": int((moving & base).sum()),
            "uphill_moving_front_loaded_ticks": front, "uphill_moving_rear_loaded_ticks": rear,
            "uphill_moving_both_axles_loaded_ticks": int((moving & both).sum()),
            "uphill_moving_four_loaded_ticks": numerator,
            "uphill_moving_four_loaded_fraction": numerator / denominator if denominator else None,
            "fallen_ticks": int(np.sum(rows["fallen"])), "illegal_contact_ticks": int(np.sum(rows["illegal_contact"])),
            "steps_per_terrain": dict(zip(pilot.TERRAINS, np.bincount(terrain, minlength=3).tolist())),
            "uphill_episodes": episodes}


def load_predictor89(folder):
    folder = Path(folder)
    report = read(folder / "report.json")
    path = folder / "policy.zip"
    pilot.require(report["status"] == "COMPLETE" and report["protocol_sha256"] == pilot.sha(PLAN)
                  and report["arm"] in read(PLAN)["training"]["arms"]
                  and not path.is_symlink() and pilot.sha(path) == report["policy_sha256"], "unqualified 89 checkpoint")
    pilot.single_cpu()
    return pilot.predictor(pilot.load_weights(path), "range")


def train_one(arm, parent_seed, train_seed, output):
    """Create one fresh run. Resource admission and serial scheduling belong to launch89."""
    from tools89.course89 import make_env

    plan = registered_plan()
    params = plan["training"]
    pilot.require(arm in params["arms"] and (parent_seed, train_seed) in zip(params["parent_seeds"], params["seeds"])
                  and params["transitions_per_model"] == pilot.STEPS, "unregistered arm, seed pair or budget")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "PARTIAL", "arm": arm, "training_seed": train_seed, "parent_seed": parent_seed,
              "protocol_sha256": pilot.sha(PLAN), "source_sha256": pilot.sha(__file__), "budget": pilot.STEPS,
              "hyperparameters": pilot.PPO_KWARGS, "optimizer_steps": 0}
    pilot.write_json(output / "protocol.json", report)
    env = trace = model = trace_report = None

    def stop(signum, frame):
        raise InterruptedError(f"owned training worker signal {signum}")

    old_handler = signal.signal(signal.SIGTERM, stop)
    try:
        pilot.single_cpu()
        checkpoint = parent_checkpoint(parent_seed, plan)
        report["parent_policy_sha256"] = pilot.sha(checkpoint)
        env = make_env("cycle", train_seed, curriculum=(arm == "mixed"),
                       slope_start_x_m=plan["curriculum"]["slope_start_x_m"], perception=True, max_transitions=pilot.STEPS)
        trace = CurriculumTrace(env)
        model = pilot.PPO(pilot.FullBodyPolicy, trace, policy_kwargs=pilot.POLICY_KWARGS, seed=train_seed,
                          device="cpu", **pilot.PPO_KWARGS)
        policy = model.policy
        report["inheritance"] = inherit_state(policy, checkpoint)
        predict = pilot.predictor(policy, "range")
        with np.load(checkpoint.parent / "probes.npz", allow_pickle=False) as z:
            actors, ranges, before = (z[k] for k in ("actor", "raw_range", "actions"))
        actual = np.stack([predict(x, r) for x, r in zip(actors, ranges, strict=True)])
        pilot.require(np.array_equal(actual, before), "initial policy differs on archived parent actions")
        report["initial_probe_count"] = len(actual)
        report["initial_probe_actions_sha256"] = hashlib.sha256(actual.tobytes()).hexdigest()
        report["pre_rollout_torch_rng_sha256"] = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        before_head = policy.action_net.weight.detach().clone()
        before_bias = policy.action_net.bias.detach().clone()
        before_projection = policy.mlp_extractor.range_projection.weight.detach().clone()
        expected_base_step = report["inheritance"]["inherited_adam_steps"][pilot.PROJECTION]

        def count_step(optimizer, args, kwargs):
            report["optimizer_steps"] += 1

        with policy.optimizer.register_step_post_hook(count_step):
            model.learn(total_timesteps=pilot.STEPS)
        counts = env.get_counts()
        report["counts"] = counts
        pilot.require(model.num_timesteps == counts["policy_transitions"] == pilot.STEPS
                      and counts["pd_updates"] == 5 * pilot.STEPS and counts["native_steps"] == 10 * pilot.STEPS,
                      "incomplete PPO or physics budget")
        projection = policy.mlp_extractor.range_projection.weight
        steps = report["optimizer_steps"]
        pilot.require(0 < steps <= pilot.PPO_KWARGS["n_epochs"] * pilot.STEPS // pilot.PPO_KWARGS["batch_size"]
                      and int(policy.optimizer.state[projection]["step"]) == expected_base_step + steps,
                      "inherited versus new Adam step count")
        rows = torch.sqrt((policy.action_net.weight.detach() - before_head).square().sum(1)
                          + (policy.action_net.bias.detach() - before_bias).square())
        proj_delta = float(torch.linalg.vector_norm(projection.detach() - before_projection))
        pilot.require(np.isfinite(proj_delta) and proj_delta > 0 and torch.isfinite(rows).all()
                      and (rows > 0).all(), "missing or invalid full-body/sensor update")
        report.update(action_head_row_delta_l2=rows.tolist(), range_projection_delta_l2=proj_delta,
                      leg_rows_changed=int((rows > 0).sum()) - len(pilot.WHEELS), wheel_rows_changed=len(pilot.WHEELS))
        with (output / "policy.zip").open("xb") as stream:
            model.save(stream)
        report["policy_sha256"] = pilot.sha(output / "policy.zip")
        after = np.stack([predict(x, r) for x, r in zip(actors, ranges, strict=True)])
        reloaded = pilot.predictor(pilot.load_weights(output / "policy.zip"), "range")
        replay = np.stack([reloaded(x, r) for x, r in zip(actors, ranges, strict=True)])
        masked = np.stack([reloaded(x, np.zeros_like(r)) for x, r in zip(actors, ranges, strict=True)])
        pilot.require(np.array_equal(after, replay) and np.any(after != before) and np.any(after != masked),
                      "final action, signal or reload failure")
        pilot.save_arrays(output / "probes.npz", dict(actor=actors, raw_range=ranges, parent=before, actions=after,
                                                      reloaded=replay, masked=masked))
        trace_report = trace.archive(output)
        report.update(coverage=coverage(trace.rows), reload_probe_actions_bitwise_equal=True, trace=trace_report,
                      max_probed_sensor_action_change=float(np.max(np.abs(after - masked))),
                      max_probed_change_from_parent=float(np.max(np.abs(after - before))))
        pilot.require(trace_report["completed_recorded_ticks"] == pilot.STEPS
                      and trace_report["valid_actor_range_ticks"] == trace_report["valid_raw_range_ticks"] > 0,
                      "missing recorded rollout or sensor")
        report["status"] = "COMPLETE"
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if model is not None and report["optimizer_steps"] and not (output / "policy.zip").exists():
            try:
                with (output / "partial_policy.zip").open("xb") as stream:
                    model.save(stream)
            except Exception as save_error:
                report["partial_save_error"] = str(save_error)
        raise
    finally:
        try:
            if env is not None:
                report["counts"] = env.get_counts()
                env.close()
            if trace is not None and trace_report is None and not (output / "train_trace.npz").exists():
                report["trace"] = trace.archive(output)
        finally:
            pilot.write_json(output / "report.json", report)
            signal.signal(signal.SIGTERM, old_handler)
    return report
