#!/usr/bin/env python3
"""Independently bounded fine-tuning from the completed control checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common18 import ROOT, atomic_json, load_plan, require, sha, validate_launch  # noqa: E402
from tools16.contract16 import expected_counts as expected_counts, read_memory  # noqa: E402
from tools16.train_comparison16 import (  # noqa: E402
    RealRuntime as PreviousRuntime, StopRequested, Watchdog, bounded_call,
    close_workers, complete_counts, count_totals, termination_scope, training_phase,
)

STAGES = {"preflight_zero": ("zero", 32768), "preflight_random": ("uniform_0_3", 32768),
          "train_zero": ("zero", 1048576), "train_random": ("uniform_0_3", 1048576)}
HYPERPARAMETERS = {"aux_coef": 0., "device": "cpu", "cpu_threads": 1, "learning_rate": 3e-4,
                   "n_steps": 1024, "batch_size": 256, "n_epochs": 4, "gamma": .99,
                   "gae_lambda": .95, "clip_range": .2, "ent_coef": 0., "vf_coef": .5,
                   "max_grad_norm": .5, "log_std_init": -1.}


def resolved(root, name):
    path = Path(name)
    return (path if path.is_absolute() else root/path).resolve()


def validate_loaded_model(model, hyperparameters, initial):
    """Loading may restore optimizer state, but must not change the PPO settings."""
    require(type(model.num_timesteps) is int and model.num_timesteps == initial,
            "checkpoint timestep offset changed")
    for key in ("aux_coef", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
                "ent_coef", "vf_coef", "max_grad_norm", "learning_rate"):
        require(getattr(model, key) == hyperparameters[key], "loaded PPO setting changed: "+key)
    require(model.clip_range(1.) == model.clip_range(0.) == hyperparameters["clip_range"],
            "loaded clipping schedule changed")
    require(str(model.device) == "cpu" and model.n_envs == 4 and model.seed == 181
            and model.normalize_advantage is True and model.clip_range_vf is None
            and model.target_kl is None and model.use_sde is False, "unsupported loaded PPO state")


def optimizer_sha(optimizer):
    """Hash restored Adam values independently of ZIP/tensor pickle encoding."""
    digest = hashlib.sha256()

    def visit(value):
        if hasattr(value, "detach"):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(b"tensor"+str(array.dtype).encode()+str(array.shape).encode()+array.tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict")
            for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
                visit(key)
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(type(value).__name__.encode())
            for item in value:
                visit(item)
        else:
            require(value is None or type(value) in (str, int, float, bool), "unsupported optimizer value")
            require(type(value) is not float or math.isfinite(value), "nonfinite optimizer state")
            digest.update(json.dumps(value, allow_nan=False).encode()+b";")
    state = optimizer.state_dict()
    require(bool(state.get("state")) and bool(state.get("param_groups")), "missing restored optimizer state")
    visit(state)
    return digest.hexdigest()


def valid_delay_state(rows, envs, mode):
    if not isinstance(rows, list) or len(rows) != envs:
        return False
    for rank, row in enumerate(rows):
        if not isinstance(row, dict):
            return False
        counts, last = row.get("delay_counts"), row.get("last_selection")
        if (not isinstance(counts, dict) or set(counts) != {"0", "1", "2", "3"}
                or not all(type(value) is int and value >= 0 for value in counts.values())
                or not isinstance(last, dict) or last.get("mode") != mode
                or type(last.get("rank")) is not int or last["rank"] != rank
                or type(last.get("selected_delay_ticks")) is not int
                or last["selected_delay_ticks"] not in (0, 1, 2, 3)
                or counts[str(last["selected_delay_ticks"])] < 1
                or sum(counts.values()) < 1):
            return False
        if mode == "zero" and (last["selected_delay_ticks"] != 0 or any(counts[key] for key in ("1", "2", "3"))):
            return False
    return True


class RealRuntime(PreviousRuntime):
    def create_vec(self, plan, transitions, output, root, mode):
        import gymnasium as gym
        from d1_rl.delay_env18 import D1Delay18Env
        from d1_rl.velocity_labels15 import VelocityLabelWrapper
        from stable_baselines3.common.vec_env import SubprocVecEnv
        sys.path.insert(0, str(root/"tools09"))
        from train_d1_ppo09 import CountedMonitor

        class PhaseWrapper(gym.Wrapper):
            def step(self, action):
                base = self.env.unwrapped
                phase = training_phase(base._training_schedule, base._episode_steps)
                obs, reward, terminated, truncated, info = self.env.step(action)
                return obs, reward, terminated, truncated, dict(info, training_phase=phase)

            def get_delay_state18(self):
                return self.env.unwrapped.get_delay_state18()

        def factory(rank):
            def make():
                base = D1Delay18Env(training_delay_mode=mode, config=root/plan["config"], arm="B",
                                   seed=plan["seed"], rank=rank, max_transitions=transitions//plan["envs"])
                return PhaseWrapper(VelocityLabelWrapper(CountedMonitor(base, str(output/f"env_{rank:02d}.monitor.csv"))))
            return make
        return SubprocVecEnv([factory(rank) for rank in range(plan["envs"])], start_method="forkserver")

    def create_model(self, vec, plan, hyperparameters, output):
        from d1_rl.velocity_policy15 import AuxiliaryVelocityPPO
        from stable_baselines3.common.logger import configure
        model = AuxiliaryVelocityPPO.load(plan["base_checkpoint"]["path"], env=vec, device="cpu",
                                         seed=plan["seed"], force_reset=True)
        validate_loaded_model(model, hyperparameters, plan["base_checkpoint"]["trained_transitions"])
        self.initial_identity = {"num_timesteps": model.num_timesteps,
                                 "policy_state_sha256": self.policy_sha(model),
                                 "optimizer_state_sha256": optimizer_sha(model.policy.optimizer),
                                 "optimizer_restored": True, "updates": model._n_updates}
        model.set_logger(configure(str(output), ["csv"]))
        return model

    def learn(self, model, transitions, watchdog):
        from stable_baselines3.common.callbacks import BaseCallback
        initial = model.num_timesteps

        class Callback(BaseCallback):
            def _on_step(self):
                return watchdog.on_step(self.locals["infos"], self.locals["rewards"], self.num_timesteps-initial)

            def _on_rollout_end(self):
                watchdog.on_rollout_end(self.num_timesteps-initial)
        if not watchdog.poll(force=True, phase="before_learn"):
            raise StopRequested(watchdog.reason)
        model.learn(total_timesteps=transitions, callback=Callback(), progress_bar=False, reset_num_timesteps=False)

    def delay_state(self, vec):
        return vec.env_method("get_delay_state18")


def run(plan_path, stage, output, *, runtime=None, loader=load_plan, launch_validator=validate_launch,
        memory_reader=read_memory, clock=time.monotonic, root=ROOT):
    started = clock()
    root, plan_path, output = Path(root).resolve(), Path(plan_path).resolve(), Path(output).absolute()
    plan = loader(plan_path)
    require(stage in STAGES, "unknown training stage")
    mode, transitions = STAGES[stage]
    spec = plan["stages"][stage]
    require(spec["mode"] == mode and spec["transitions"] == transitions and plan["envs"] == 4
            and plan["seed"] == 181, "training stage scope changed")
    require(output.resolve() == (root/plan["output_root"]/stage).resolve() and not output.exists(),
            "output must be the exclusive fixed stage directory")
    require(output.is_relative_to(root), "output outside root")
    cursor = output
    while cursor != root:
        require(not cursor.is_symlink(), "symlink stage output forbidden")
        cursor = cursor.parent
    launch_validator(plan, plan_path, stage, output)
    baseline = plan["base_checkpoint"]
    checkpoint = resolved(root, baseline["path"])
    base_protocol = plan["base_protocol"]
    protocol_path = resolved(root, base_protocol["path"])
    require(sha(checkpoint) == baseline["sha256"] and baseline["trained_transitions"] == 1048576
            and sha(protocol_path) == base_protocol["sha256"], "baseline identity changed")
    baseline_protocol = json.loads(protocol_path.read_text())
    hyper = baseline_protocol["hyperparameters"]
    require(hyper == HYPERPARAMETERS, "baseline PPO hyperparameters changed")
    require(memory_reader()["effective_available_bytes"] >= plan["resources"]["start_min_effective_bytes"],
            "NOT_READY: insufficient start RAM")
    plan_sha = sha(plan_path)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"format": "d1-delay18-training-protocol-v1", "plan_sha256": plan_sha,
                "stage": stage, "mode": mode, "seed": plan["seed"], "envs": plan["envs"],
                "planned_new_transitions": transitions, "planned_native_steps": transitions*10,
                "base_checkpoint": baseline, "base_protocol": base_protocol,
                "hyperparameters": hyper, "source_identity": plan["source_identity"],
                "warm_start": True, "teacher_policy_used": False, "reset_num_timesteps": False,
                "preflight_weights_discarded": True, "versions": None,
                "delay_sampling": "per episode; private stream 18018; inherited FIFO", "phase_tick_s": .02}
    atomic_json(output/"protocol.json", protocol)
    vec = model = watcher = workers = totals = delays = initial = None
    status, error, cleanup, inputs_unchanged, spawn_started = "ERROR", None, True, False, False
    checkpoint_sha = startup_stop = None
    with termination_scope() as termination:
        try:
            runtime = runtime or RealRuntime()
            runtime.prepare()
            protocol["versions"] = runtime.versions
            atomic_json(output/"protocol.json", protocol)
            require(runtime.versions == baseline_protocol["versions"], "runtime versions differ from completed baseline")
            loader(plan_path)
            require(sha(plan_path) == plan_sha, "plan changed before spawn")
            sample = memory_reader()
            require(sample["effective_available_bytes"] >= plan["resources"]["start_min_effective_bytes"],
                    "NOT_READY: insufficient RAM after imports")
            require(clock()-started < spec["max_wall_s"], "WALL_LIMIT before spawn")
            termination["defer"] = True
            spawn_started = True
            try:
                vec = runtime.create_vec(plan, transitions, output, root, mode)
            finally:
                termination["defer"] = False
            if termination["received"]:
                raise StopRequested(termination["received"])
            sample = dict(memory_reader(), phase="after_spawn", elapsed_s=clock()-started)
            if sample["effective_available_bytes"] < plan["resources"]["reserve_bytes"]:
                startup_stop = dict(sample, reason="RAM_RESERVE_BELOW_3_GIB")
                atomic_json(output/"stop_sample.json", startup_stop)
                raise StopRequested(startup_stop["reason"])
            model = runtime.create_model(vec, plan, hyper, output)
            initial = runtime.initial_identity
            require(initial["num_timesteps"] == 1048576 and initial["optimizer_restored"] is True,
                    "baseline model or optimizer was not restored")
            require(all(isinstance(initial[key], str) and len(initial[key]) == 64
                        for key in ("policy_state_sha256", "optimizer_state_sha256")), "missing initial state identities")
            watcher = Watchdog(output, plan, stage, started, runtime, vec, model, memory_reader, clock)
            runtime.learn(model, transitions, watcher)
            watcher.poll(force=True, phase="after_learn")
            status = watcher.reason or ("COMPLETE" if model.num_timesteps == 1048576+transitions else "PARTIAL")
        except BaseException as exc:
            status = watcher.reason if watcher and watcher.reason else str(exc) if isinstance(exc, StopRequested) else "ERROR"
            error = f"{type(exc).__name__}: {exc}"
            (output/"error.txt").write_text(traceback.format_exc())
        finally:
            termination["defer"] = True
            if vec is not None:
                try:
                    workers = bounded_call(lambda: runtime.counters(vec))
                    totals = count_totals(workers)
                except BaseException as exc:
                    error = (error or "")+f"; counter collection failed: {exc}"
                try:
                    delays = bounded_call(lambda: runtime.delay_state(vec))
                    require(valid_delay_state(delays, plan["envs"], mode), "invalid delay provenance")
                except BaseException as exc:
                    error = (error or "")+f"; delay provenance collection failed: {exc}"
                cleanup, close_error = close_workers(vec)
                if close_error:
                    error = (error or "")+"; "+close_error
            elif spawn_started:
                cleanup = False
            try:
                loader(plan_path)
                require(sha(plan_path) == plan_sha and sha(checkpoint) == baseline["sha256"]
                        and sha(protocol_path) == base_protocol["sha256"], "plan or baseline changed during stage")
                inputs_unchanged = True
            except BaseException as exc:
                error = (error or "")+f"; inputs changed: {exc}"
            final = getattr(model, "num_timesteps", None)
            new = final-1048576 if type(final) is int and final >= 1048576 else None
            if termination["received"]:
                status = termination["received"]
            if status == "COMPLETE" and (error or not cleanup or not inputs_unchanged
                    or not complete_counts(workers, plan["envs"], transitions)
                    or not watcher or watcher.rollouts*4096 != transitions
                    or not valid_delay_state(delays, plan["envs"], mode)):
                status = "INCOMPLETE"
            if model is not None:
                saved = output/("policy.zip" if status == "COMPLETE" else "partial_policy.zip")
                try:
                    model.save(str(saved))
                    loader(plan_path)
                    require(sha(plan_path) == plan_sha and sha(checkpoint) == baseline["sha256"]
                            and sha(protocol_path) == base_protocol["sha256"], "inputs changed during save")
                    checkpoint_sha = sha(saved)
                except BaseException as exc:
                    status, inputs_unchanged = "ERROR", False
                    error = (error or "")+f"; checkpoint save/verification failed: {exc}"
                if termination["received"]:
                    status = termination["received"]
                if status != "COMPLETE" and (output/"policy.zip").exists():
                    (output/"policy.zip").replace(output/"partial_policy.zip")
            result = {"format": "d1-delay18-training-result-v1", "plan_sha256": plan_sha,
                      "protocol_sha256": sha(output/"protocol.json"), "stage": stage, "mode": mode,
                      "seed": plan["seed"], "envs": plan["envs"], "aux_coef": 0., "status": status,
                      "source_identity": plan["source_identity"], "versions": protocol["versions"],
                      "hyperparameters": hyper,
                      "base_checkpoint_sha256": baseline["sha256"], "new_trained_transitions": new,
                      "initial_num_timesteps": initial["num_timesteps"] if initial else None,
                      "final_num_timesteps": final, "initial_state": initial,
                      "actual_counts": totals, "worker_counts": workers, "per_worker_counts": workers,
                      "delay_state18": delays, "inputs_unchanged": inputs_unchanged,
                      "cleanup_complete": cleanup, "error": error, "checkpoint_sha256": checkpoint_sha,
                      "eligible_for_evaluation": status == "COMPLETE" and stage.startswith("train_"),
                      "elapsed_total_s": clock()-started, "stop_sample": watcher.stop_sample if watcher else startup_stop,
                      "min_available_ram_bytes": watcher.min_available if watcher else None,
                      "min_effective_available_bytes": watcher.min_effective if watcher else None,
                      "preflight_weights_used_for_training": False}
            atomic_json(output/"result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args.plan, args.stage, args.output)
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({"status": "REFUSED_BEFORE_SPAWN", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
