#!/usr/bin/env python3
"""Authorized, exclusive CPU comparison stages; imports no simulator at startup."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract16 import ROOT, atomic_json, expected_counts, load_release, read_memory, sha, validate_sources  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


class StopRequested(Exception):
    pass


@contextmanager
def termination_scope():
    control = {"defer": False, "received": None}
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def stop(signum, _frame):
        control["received"] = signal.Signals(signum).name
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        if not control["defer"]:
            raise StopRequested(control["received"])

    try:
        for sig in previous:
            signal.signal(sig, stop)
        yield control
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def bounded_call(function, timeout=3.):
    """A dead worker must not block final accounting/cleanup indefinitely."""
    result = []

    def invoke():
        try:
            result.append((True, function()))
        except BaseException as exc:
            result.append((False, exc))
    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError("worker operation exceeded bounded wait")
    if not result[0][0]:
        raise result[0][1]
    return result[0][1]


def close_workers(vec):
    error = None
    try:
        bounded_call(vec.close)
    except BaseException as exc:
        error = f"vector close failed: {type(exc).__name__}: {exc}"
    processes = getattr(vec, "processes", [])
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1.)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.)
        except BaseException as exc:
            error = (error or "")+f"; worker cleanup failed: {exc}"
    try:
        complete = not any(process.is_alive() for process in processes)
    except BaseException as exc:
        complete, error = False, (error or "")+f"; worker liveness unknown: {exc}"
    return complete, error


def count_totals(workers):
    require(isinstance(workers, list) and bool(workers), "missing worker counters")
    keys = set(workers[0])
    require(set(expected_counts(0)) <= keys and all(isinstance(row, dict) and set(row) == keys
            and all(type(value) is int and value >= 0 for value in row.values()) for row in workers),
            "invalid worker counters")
    return {key: sum(row[key] for row in workers) for key in sorted(keys)}


def complete_counts(workers, envs, transitions):
    try:
        totals = count_totals(workers)
        return (len(workers) == envs and all(totals[key] == value for key, value in expected_counts(transitions).items())
                and all(all(row[key] == value for key, value in expected_counts(transitions//envs).items())
                        for row in workers))
    except (ValueError, TypeError, KeyError):
        return False


def training_phase(course, tick):
    """Use the actual pre-step 03d course, not a regenerated RNG schedule."""
    require(isinstance(course, dict) and isinstance(tick, int), "missing actual training course")
    previous = None
    for index, (begin, end, command) in enumerate(course["segments"]):
        vx, vy, wz = command
        require(vy == 0 and not (vx and wz), "unexpected paired-command axes")
        direction = ("forward" if vx > 0 else "backward") if vx else ("turn_left" if wz > 0 else "turn_right") if wz else None
        if begin <= tick < end:
            if direction:
                return "drive_"+direction
            if index == 0:
                return "stationary_initial"
            require(previous is not None, "zero course has no preceding drive")
            return ("cut_zero_after_" if tick-begin < 25 else "stationary_after_")+previous
        if direction:
            previous = direction
    raise ValueError("training tick is outside its actual course")


class Watchdog:
    def __init__(self, output, release, stage, started, runtime, vec, model, memory_reader, clock):
        self.output, self.release, self.stage = output, release, stage
        self.started, self.runtime, self.vec, self.model = started, runtime, vec, model
        self.memory_reader, self.clock = memory_reader, clock
        self.reason = self.stop_sample = self.last_sample = None
        self.last_poll, self.rollouts = -float("inf"), 0
        self.min_available = self.min_effective = None
        self.phases = {}

    def poll(self, *, force=False, phase="training"):
        elapsed = self.clock()-self.started
        if force or elapsed-self.last_poll >= self.release["resources"]["poll_s"]:
            sample = dict(self.memory_reader(), elapsed_s=elapsed, phase=phase)
            for key in ("available_ram_bytes", "effective_available_bytes"):
                require(type(sample[key]) is int and sample[key] >= 0, "invalid RAM accounting")
            self.last_poll, self.last_sample = elapsed, sample
            self.min_available = min(self.min_available, sample["available_ram_bytes"]) if self.min_available is not None else sample["available_ram_bytes"]
            self.min_effective = min(self.min_effective, sample["effective_available_bytes"]) if self.min_effective is not None else sample["effective_available_bytes"]
            with (self.output/"resource_samples.jsonl").open("a") as stream:
                stream.write(json.dumps(sample, allow_nan=False)+"\n")
            reason = ("RAM_RESERVE_BELOW_3_GIB" if sample["effective_available_bytes"] < self.release["resources"]["reserve_bytes"]
                      else "WALL_LIMIT" if elapsed >= self.release["stages"][self.stage]["max_wall_s"] else None)
            if reason and self.reason is None:
                self.reason, self.stop_sample = reason, sample
                atomic_json(self.output/"stop_sample.json", dict(sample, reason=reason))
        return self.reason is None

    def on_step(self, infos, rewards, transitions):
        require(len(infos) == len(rewards) == self.release["envs"], "callback worker alignment changed")
        for info, reward in zip(infos, rewards):
            label = info["training_phase"]
            require(isinstance(label, str) and math.isfinite(float(reward)), "invalid phase reward")
            terms = info["reward_terms"]["contributions_per_tick"]
            require(isinstance(terms, dict) and all(math.isfinite(float(value)) for value in terms.values()), "invalid reward components")
            cell = self.phases.setdefault(label, {"ticks": 0, "reward_sum": 0., "contribution_sum": defaultdict(float)})
            cell["ticks"] += 1
            cell["reward_sum"] += float(reward)
            for name, value in terms.items():
                cell["contribution_sum"][name] += float(value)
        return self.poll()

    def on_rollout_end(self, transitions):
        require(transitions == (self.rollouts+1)*4096 and sum(cell["ticks"] for cell in self.phases.values()) == 4096,
                "rollout or phase denominator is not exactly 4096")
        workers = bounded_call(lambda: self.runtime.counters(self.vec))
        totals = count_totals(workers)
        require(complete_counts(workers, self.release["envs"], transitions), "rollout worker counters mismatch")
        phases = {name: dict(cell, contribution_sum=dict(cell["contribution_sum"]),
                           reward_mean_per_tick=cell["reward_sum"]/cell["ticks"],
                           contribution_mean_per_tick={key: value/cell["ticks"] for key, value in cell["contribution_sum"].items()})
                  for name, cell in sorted(self.phases.items())}
        self.rollouts += 1
        deterministic = {"rollout": self.rollouts, "transitions": transitions, "actual_counts": totals,
                         "per_worker_counts": workers, "phase_rewards": phases,
                         "policy_state_sha256": self.runtime.policy_sha(self.model)}
        self.poll(force=True, phase="rollout_end")
        record = dict(deterministic, deterministic=deterministic, elapsed_training_s=self.clock()-self.started,
                      resource_sample=self.last_sample)
        with (self.output/"ledger.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False)+"\n")
        atomic_json(self.output/"live_status.json", record)
        self.phases.clear()
        if self.reason:
            raise StopRequested(self.reason)


class RealRuntime:
    """Heavy imports and all physical execution stay behind authorized run()."""
    def prepare(self):
        import multiprocessing as mp
        import platform
        import torch
        import stable_baselines3
        import numpy
        import gymnasium
        import mujoco
        # Import the complete runtime before the final 6 GiB admission sample.
        import d1_rl.factory03  # noqa: F401
        import d1_rl.velocity_labels15  # noqa: F401
        import d1_rl.velocity_policy15  # noqa: F401
        import stable_baselines3.common.vec_env  # noqa: F401
        sys.path.insert(0, str(ROOT/"tools09"))
        import train_d1_ppo09  # noqa: F401
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        mp.set_forkserver_preload(["d1_rl.forkserver_bootstrap"])
        self.versions = {"python": platform.python_version(), "torch": torch.__version__,
                         "sb3": stable_baselines3.__version__, "numpy": numpy.__version__,
                         "gymnasium": gymnasium.__version__, "mujoco": mujoco.__version__}

    def create_vec(self, release, transitions, output, root):
        import gymnasium as gym
        from d1_rl.factory03 import environment_class
        from d1_rl.velocity_labels15 import VelocityLabelWrapper
        from stable_baselines3.common.vec_env import SubprocVecEnv
        sys.path.insert(0, str(root/"tools09"))
        from train_d1_ppo09 import CountedMonitor

        class PhaseWrapper(gym.Wrapper):
            def step(self, action):
                base = self.env.unwrapped
                phase = training_phase(base._training_schedule, base._episode_steps)
                observation, reward, terminated, truncated, info = self.env.step(action)
                return observation, reward, terminated, truncated, dict(info, training_phase=phase)

        def factory(rank):
            def make():
                config_path = root/release["config"]
                config = json.loads(config_path.read_text())
                base = environment_class(config)(config=config_path, arm="B", seed=release["seed"],
                                                  rank=rank, max_transitions=transitions//release["envs"])
                return PhaseWrapper(VelocityLabelWrapper(CountedMonitor(base, str(output/f"env_{rank:02d}.monitor.csv"))))
            return make
        return SubprocVecEnv([factory(rank) for rank in range(release["envs"])], start_method="forkserver")

    def create_model(self, vec, release, hyperparameters, output):
        from d1_rl.velocity_policy15 import AuxiliaryVelocityPPO
        from stable_baselines3.common.logger import configure
        hyper = {key: value for key, value in hyperparameters.items() if key not in ("cpu_threads", "log_std_init")}
        model = AuxiliaryVelocityPPO(env=vec, seed=release["seed"], verbose=0,
                                    policy_kwargs={"log_std_init": hyperparameters["log_std_init"]}, **hyper)
        model.set_logger(configure(str(output), ["csv"]))
        return model

    def learn(self, model, transitions, watchdog):
        from stable_baselines3.common.callbacks import BaseCallback

        class Callback(BaseCallback):
            def _on_step(self):
                return watchdog.on_step(self.locals["infos"], self.locals["rewards"], self.num_timesteps)

            def _on_rollout_end(self):
                watchdog.on_rollout_end(self.num_timesteps)
        if not watchdog.poll(force=True, phase="before_learn"):
            raise StopRequested(watchdog.reason)
        model.learn(total_timesteps=transitions, callback=Callback(), progress_bar=False)

    def counters(self, vec):
        return vec.env_method("counters")

    def policy_sha(self, model):
        digest = hashlib.sha256()
        for name, value in sorted(model.policy.state_dict().items()):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(name.encode()+str(array.dtype).encode()+str(array.shape).encode()+array.tobytes())
        return digest.hexdigest()


def run(release_path, arm, stage, output, *, root=ROOT, runtime=None, memory_reader=read_memory,
        clock=time.monotonic, loader=load_release, validator=validate_sources, ticket_validator=None):
    started = clock()  # Includes authorization checks, imports, spawn and NN setup.
    root, release_path, output = Path(root).resolve(), Path(release_path).resolve(), Path(output).resolve()
    release = loader(release_path, require_authorized=True, root=root)
    validator(release, root=root)
    require(arm in ("control", "velocity") and stage in ("preflight", "train"), "invalid arm or stage")
    expected = root/release["output_root"]/f"{stage}_{arm}"
    require(output == expected.resolve() and output.is_relative_to(root) and not output.exists(), "output must be the new fixed stage path")
    cursor = expected
    while cursor != root:
        require(not cursor.is_symlink(), "symlink output paths are forbidden")
        cursor = cursor.parent
    if ticket_validator is None:
        from run_comparison16 import validate_stage_launch
        ticket_validator = validate_stage_launch
    ticket_validator(release, release_path, f"{stage}_{arm}", output, root=root)
    for _ in range(2):
        sample = memory_reader()
        require(sample["effective_available_bytes"] >= release["resources"]["start_min_effective_bytes"], "NOT_READY: insufficient start RAM")
        if runtime is None:
            runtime = RealRuntime()
        if _ == 0:
            runtime.prepare()
    validator(release, root=root)
    require(clock()-started < release["stages"][stage]["max_wall_s"], "WALL_LIMIT before spawn")
    release_sha = sha(release_path)
    transitions = release["stages"][stage]["transitions"]
    hyper = {"aux_coef": release["arms"][arm]["aux_coef"], "device": "cpu", "cpu_threads": 1,
             "learning_rate": 3e-4, "n_steps": 4096//release["envs"], "batch_size": 256, "n_epochs": 4,
             "gamma": .99, "gae_lambda": .95, "clip_range": .2, "ent_coef": 0., "vf_coef": .5,
             "max_grad_norm": .5, "log_std_init": -1.}
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"format": "d1-comparison16-training-protocol-v1", "release_sha256": release_sha,
                "arm": arm, "stage": stage, "envs": release["envs"], "seed": release["seed"],
                "planned_transitions": transitions, "planned_native_steps": transitions*10,
                "inputs_sha256": release["sources_sha256"], "hyperparameters": hyper,
                "random_initialization": True, "warm_start": False, "teacher_policy_used": False,
                "versions": runtime.versions,
                "environment_arm": "B", "phase_tick_s": .02, "cut_zero_ticks": 25,
                "config": json.loads((root/release["config"]).read_text())}
    atomic_json(output/"protocol.json", protocol)
    vec, model, watcher, workers, totals = None, None, None, None, None
    status, error, cleanup, inputs_unchanged, spawn_started = "ERROR", None, True, False, False
    checkpoint_sha = startup_stop = None
    with termination_scope() as termination:
        try:
            termination["defer"] = True
            spawn_started = True
            try:
                vec = runtime.create_vec(release, transitions, output, root)
            finally:
                termination["defer"] = False
            if termination["received"]:
                raise StopRequested(termination["received"])
            sample = dict(memory_reader(), phase="after_spawn", elapsed_s=clock()-started)
            if sample["effective_available_bytes"] < release["resources"]["reserve_bytes"]:
                startup_stop = dict(sample, reason="RAM_RESERVE_BELOW_3_GIB")
                atomic_json(output/"stop_sample.json", startup_stop)
                raise StopRequested(startup_stop["reason"])
            model = runtime.create_model(vec, release, hyper, output)
            watcher = Watchdog(output, release, stage, started, runtime, vec, model, memory_reader, clock)
            runtime.learn(model, transitions, watcher)
            watcher.poll(force=True, phase="after_learn")
            status = watcher.reason or ("COMPLETE" if model.num_timesteps == transitions else "PARTIAL")
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
                cleanup, close_error = close_workers(vec)
                if close_error:
                    error = (error or "")+"; "+close_error
            elif spawn_started:
                cleanup = False  # Partial constructor failure needs outer session cleanup.
            try:
                validator(release, root=root)
                require(sha(release_path) == release_sha, "release changed during stage")
                inputs_unchanged = True
            except BaseException as exc:
                error = (error or "")+f"; sources changed: {exc}"
            if termination["received"]:
                status = termination["received"]
            trained = int(model.num_timesteps) if model is not None else 0
            if status == "COMPLETE" and (error or not cleanup or not inputs_unchanged
                    or not complete_counts(workers, release["envs"], transitions)
                    or not watcher or watcher.rollouts*4096 != transitions):
                status = "INCOMPLETE"
            if model is not None:
                checkpoint = output/("policy.zip" if status == "COMPLETE" else "partial_policy.zip")
                try:
                    model.save(str(checkpoint))
                except BaseException as exc:
                    status = "ERROR"
                    error = (error or "")+f"; checkpoint save failed: {exc}"
                else:
                    try:
                        validator(release, root=root)
                        require(sha(release_path) == release_sha, "release changed during save")
                        checkpoint_sha = sha(checkpoint)
                    except BaseException as exc:
                        status, inputs_unchanged = "ERROR", False
                        error = (error or "")+f"; checkpoint source verification failed: {exc}"
                if status != "COMPLETE" and (output/"policy.zip").exists():
                    (output/"policy.zip").replace(output/"partial_policy.zip")
            if termination["received"] and status == "COMPLETE":
                status = termination["received"]
                (output/"policy.zip").replace(output/"partial_policy.zip")
            result = {"format": "d1-comparison16-training-result-v1", "release_sha256": release_sha,
                      "protocol_sha256": sha(output/"protocol.json"), "arm": arm, "stage": stage,
                      "seed": release["seed"], "envs": release["envs"], "aux_coef": hyper["aux_coef"],
                      "status": status, "trained_transitions": trained, "actual_counts": totals,
                      "per_worker_counts": workers, "inputs_unchanged": inputs_unchanged,
                      "cleanup_complete": cleanup, "error": error, "checkpoint_sha256": checkpoint_sha,
                      "eligible_for_evaluation": status == "COMPLETE" and stage == "train",
                      "elapsed_total_s": clock()-started, "stop_sample": watcher.stop_sample if watcher else startup_stop,
                      "min_available_ram_bytes": watcher.min_available if watcher else None,
                      "min_effective_available_bytes": watcher.min_effective if watcher else None,
                      "phase_denominator": "observed policy transitions in each phase, not episode hazard"}
            atomic_json(output/"result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "velocity"), required=True)
    parser.add_argument("--stage", choices=("preflight", "train"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args.release, args.arm, args.stage, args.output)
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "REFUSED_BEFORE_SPAWN", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
