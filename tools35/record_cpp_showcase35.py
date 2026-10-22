"""Record two independent C++34 demonstrations without changing frozen sources.

This is a presentation capture, not the single-use 34 qualification runner.
Only COMPLETE34 and its completed25 weights are accepted. Partial attempts are
kept, never resumed, and cannot be passed to the companion renderer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASKS = ("forward_stop", "turn_stop")
SEED = 1001
TICKS = 600
EXECUTION = ROOT / "evidence/cpp_reference34/execution_01"
POLICY = ROOT / "evidence/low_memory_delay25/execution_01/random/export/policy.onnx"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "LP_NUM_THREADS")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Replace only this attempt's journal, never another attempt's directory."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checked_files(entries):
    for name, digest in entries.items():
        path = Path(name)
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "source/artifact changed: " + str(path))


def load_inputs():
    """Read-only provenance closure; constructs no model or inference session."""
    from tools30.run_notch30 import inputs
    report = json.loads((EXECUTION / "report.json").read_text())
    require(report["status"] == "COMPLETE" and report["error"] is None
            and report["cpp_original_and_holdout_qualified"] and report["cleanup_complete"]
            and not report["automatic_model_replacement"], "complete qualified34 execution required")
    identity, bundle = inputs()
    files = {str(EXECUTION / "report.json"): sha(EXECUTION / "report.json")}
    files.update({str(ROOT / name): digest
                  for name, digest in report["source"]["sources_sha256"].items()})
    files.update({str(ROOT / name): digest
                  for name, digest in bundle["old_release"]["sources_sha256"].items()})
    files.update(report["cpp_build"]["inputs_sha256"])
    files[report["cpp_build"]["binary"]] = report["cpp_build"]["binary_sha256"]
    cases = {}
    for name in TASKS:
        selected = [case for case in report["stages"]["formal"]["cases"]
                    if (case["sequence"], case["delay_ticks"], case["reset_seed"]) == (name, 0, SEED)]
        require(len(selected) == 1, "expected one complete saved case: " + name)
        case = selected[0]
        require(case["status"] == "COMPLETE" and case["done_reason"] == "HORIZON_12S"
                and case["policy_transitions"] == TICKS and case["reference_thresholds_met"]
                and not case["failure"], "saved demonstration case is incomplete")
        files[str(EXECUTION / "formal" / case["trace_file"])] = case["trace_sha256"]
        cpp = case["cpp_reference34"]
        files[str(EXECUTION / "formal" / cpp["sidecar"])] = cpp["sha256"]
        model_sha = cpp["transport"]["identities"]["model_sha256"]
        require(model_sha == sha(POLICY), "complete34 model differs from complete25 export")
        files[str(POLICY)] = model_sha
        cases[name] = case
    checked_files(files)
    return {"files_sha256": files, "completed25_identity": identity,
            "qualification_report_sha256": sha(EXECUTION / "report.json")}, bundle, report, cases


class ResourceGuard:
    def __init__(self, output, scope):
        self.output, self.scope = Path(output), scope
        self.started, self.last = time.monotonic(), -float("inf")

    def poll(self, force=False, admission=False):
        from tools27.resource_guard27 import check_memory, snapshot_memory
        now = time.monotonic()
        require(now - self.started < 600, "demonstration wall budget exceeded")
        if not force and now - self.last < 1:
            return
        sample = snapshot_memory(self.scope)
        with (self.output / "resources.jsonl").open("a") as stream:
            stream.write(json.dumps({**sample, "elapsed_s": now - self.started}) + "\n")
        check_memory(sample)
        require(sample["effective_available_bytes"] >= (4 if admission else 2) << 30,
                "effective memory admission/reserve failed")
        require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "Nice must be at least 10")
        require(all(os.environ.get(name) == "1" for name in THREAD_VARIABLES),
                "single-thread environment required")
        self.last = now


class StateCapture:
    """Observe reset/step boundaries without modifying the wrapped environment."""
    def __init__(self, environment, stream=None):
        self.env = environment
        self.rows = []
        self.stream = stream
        self.inflight = False
        self.known_counts = environment.counters() if hasattr(environment, "counters") else {}

    def __getattr__(self, name):
        return getattr(self.env, name)

    def capture(self):
        row = (self.env.data.qpos.copy(), self.env.data.qvel.copy(), float(self.env.data.time))
        self.rows.append(row)
        if self.stream is not None:
            self.stream.write(json.dumps({"qpos": row[0].tolist(), "qvel": row[1].tolist(),
                                          "time_s": row[2]}, allow_nan=False) + "\n")
            self.stream.flush()
        if hasattr(self.env, "counters"):
            self.known_counts = self.env.counters()

    def reset(self, **kwargs):
        self.inflight = True
        result = self.env.reset(**kwargs)
        self.capture()
        self.inflight = False
        return result

    def step(self, action):
        self.inflight = True
        result = self.env.step(action)
        self.capture()
        self.inflight = False
        return result

    def save(self, path):
        import numpy as np
        if self.rows:
            with Path(path).open("xb") as stream:
                np.savez_compressed(stream, qpos=np.asarray([r[0] for r in self.rows]),
                                    qvel=np.asarray([r[1] for r in self.rows]),
                                    time_s=np.asarray([r[2] for r in self.rows]))


def validate_complete_states(qpos, qvel, time_s, nq, nv):
    import numpy as np
    require(qpos.shape == (601, nq) and qvel.shape == (601, nv)
            and time_s.shape == (601,), "demonstration requires all 601 synchronized states")
    require(all(np.isfinite(array).all() for array in (qpos, qvel, time_s)), "nonfinite states")
    require(np.allclose(time_s, np.arange(601) * .02, rtol=0, atol=1e-9),
            "demonstration must contain the complete 0..12 second clock")


def case_journal(name):
    return {"status": "INCOMPLETE", "error": None, "task": name, "seed": SEED,
            "delay_ticks": 0, "role": "presentation_only", "new_qualification_claimed": False,
            "training_transitions": 0, "actual_counts": None, "known_actual_counts": {},
            "cleanup_complete": False}


def record_case(name, output, bundle, saved_report, saved_case, helpers, guard, result):
    import numpy as np
    from d1_rl.factory03 import environment_class

    import mujoco
    from tools34.cpp_reference34 import SOURCE, CppReferenceActor

    output.mkdir()
    write_json(output / "report.json", result)
    actor, env, constructing = None, None, False
    state_stream = None
    raw_inputs, actions = [], []
    try:
        guard.poll(True)
        identities = saved_case["cpp_reference34"]["transport"]["identities"]
        actor = CppReferenceActor(saved_report["cpp_build"]["binary"], POLICY,
            output=output / "actor", expected_binary_sha256=identities["binary_sha256"],
            expected_model_sha256=identities["model_sha256"],
            expected_source_sha256=identities["source_sha256"], source=SOURCE, max_requests=TICKS)
        actor.start()
        constructing = True
        cls = environment_class(ROOT / bundle["old_release"]["config"])
        state_stream = (output / "states.jsonl").open("x")
        env = StateCapture(cls(config=ROOT / bundle["old_release"]["config"], arm="B", seed=SEED,
                               rank=0, max_transitions=TICKS, eval_delay_ticks=0,
                               eval_command_sequence=helpers.SEQUENCES[name]), state_stream)
        constructing = False
        result["construction_counts"] = env.counters()
        mujoco.mj_saveModel(env.model, str(output / "model.mjb"), None)

        def predict(observation):
            guard.poll()
            raw_inputs.append(observation.copy())
            action = actor.predict(observation)
            actions.append(action.copy())
            return action

        item = helpers.run_episode(env, predict, name, 0, SEED, output / "trace.npz", 6000)
        result["trace_checks"] = item
        require(item["done_reason"] == "HORIZON_12S" and item["policy_transitions"] == TICKS
                and not item["failure"], "capture did not complete the full demonstration")
        result["status"] = "COMPLETE"
    except BaseException as exc:  # noqa: BLE001 -- Preserve interruptions and partial evidence.
        result["error"] = f"{type(exc).__name__}: {exc}"
        (output / "error.txt").write_text(traceback.format_exc())
    finally:
        cleanup = not constructing
        if actor is not None:
            try:
                closed = actor.close()
                result["actor"] = closed
                cleanup &= closed["cleanup_complete"] and closed["status"] == "CLOSED"
                if result["status"] == "COMPLETE":
                    require(closed["requests_completed"] == closed["inference_returns"] == TICKS,
                            "capture inference counts differ")
            except BaseException as exc:  # noqa: BLE001 -- Cleanup failures must enter the journal.
                cleanup = False
                result["error"] = f"actor cleanup {type(exc).__name__}: {exc}"
        if env is not None:
            result["attempted_counts"] = env.counters()
            result["known_actual_counts"] = env.known_counts
            result["actual_counts"] = None if env.inflight else env.counters()
            result["step_or_reset_inflight"] = env.inflight
            try:
                env.save(output / "states.npz")
                if result["status"] == "COMPLETE":
                    with np.load(output / "states.npz", allow_pickle=False) as states:
                        validate_complete_states(states["qpos"], states["qvel"], states["time_s"],
                                                 env.model.nq, env.model.nv)
            except BaseException as exc:  # noqa: BLE001 -- Keep partial states even after interruption.
                cleanup = False
                result["error"] = f"state/cleanup {type(exc).__name__}: {exc}"
            finally:
                try:
                    env.close()
                except BaseException as exc:  # noqa: BLE001 -- Report an interrupted cleanup explicitly.
                    cleanup = False
                    result["error"] = f"env cleanup {type(exc).__name__}: {exc}"
        if state_stream is not None:
            state_stream.close()
        with (output / "actor_io.npz").open("xb") as stream:
            np.savez_compressed(stream, raw_actor627=np.asarray(raw_inputs), issued16=np.asarray(actions))
        result["cleanup_complete"] = bool(cleanup)
        if not cleanup or result["error"]:
            result["status"] = "INCOMPLETE"
        artifacts = {p.name: sha(p) for p in (output / "model.mjb", output / "states.npz", output / "states.jsonl",
                     output / "actor_io.npz", output / "trace.npz") if p.is_file()}
        write_json(output / "manifest.json", {"format": "mujoco-state-showcase-v1",
            "status": result["status"], "task": name, "source": "complete34_cpp_actor",
            "policy_sha256": sha(POLICY), "nq": env.model.nq if env else None,
            "nv": env.model.nv if env else None, "states": len(env.rows) if env else 0,
            "free_qpos": env.free_qpos if env else None,
            "policy_dt_s": .02, "duration_s": 12.0, "artifacts_sha256": artifacts,
            "new_qualification_claimed": False, "source_report_sha256": sha(EXECUTION / "report.json")})
        write_json(output / "report.json", result)
    return result


def execute(output, owned_scope):
    require(not output.exists(), "output exists: no overwrite, retry, or resume")
    output.mkdir(parents=True)
    started = time.monotonic()
    result = {"status": "INCOMPLETE", "error": None, "cases": [], "training_transitions": 0,
              "role": "presentation_only", "new_qualification_claimed": False,
              "automatic_model_replacement": False, "owned_scope": owned_scope,
              "planned_counts": {"policy_transitions": 1200, "native_steps": 12000, "pd_updates": 6000}}
    write_json(output / "report.json", result)
    from tools16.run_comparison16 import termination_guard
    with termination_guard() as control:
        try:
            guard = ResourceGuard(output, owned_scope)
            guard.poll(True, admission=True)
            for path in (ROOT / ".python-deps", ROOT / "training"):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            identity, bundle, saved_report, cases = load_inputs()
            result["inputs"] = identity
            result["recorder_sha256"] = sha(__file__)
            import gymnasium
            import numpy as np

            import mujoco
            versions = {"python": platform.python_version(), "gymnasium": gymnasium.__version__,
                        "mujoco": mujoco.__version__, "numpy": np.__version__}
            require(all(bundle["protocol"]["versions"][key] == value for key, value in versions.items()),
                    "runtime versions differ from completed25 training")
            result["versions"] = versions
            from tools16.evaluate_comparison16 import frozen_helpers
            helpers = frozen_helpers(bundle["old_release"])
            for name in TASKS:
                item = case_journal(name)
                result["cases"].append(item)
                # Register before starting: a signal in finalization must not erase
                # the current case or turn its unknown executed work into zero.
                write_json(output / "report.json", result)
                record_case(name, output / name, bundle, saved_report, cases[name], helpers, guard, item)
                write_json(output / "report.json", result)
                require(item["status"] == "COMPLETE", "incomplete capture: " + name)
            for key, value in result["planned_counts"].items():
                require(sum(c["actual_counts"][key] for c in result["cases"]) == value,
                        "final capture counters differ: " + key)
            checked_files(identity["files_sha256"])
            require(sha(__file__) == result["recorder_sha256"], "recorder changed during capture")
            guard.poll(True)
            result.update(status="COMPLETE", inputs_unchanged=True)
        except BaseException as exc:  # noqa: BLE001 -- Interrupted attempts remain incomplete on disk.
            result["error"] = f"{type(exc).__name__}: {exc}"
            (output / "error.txt").write_text(traceback.format_exc())
        finally:
            control["defer"] = True
            if control["received"]:
                result.update(status="INCOMPLETE", error=control["received"])
            result["elapsed_s"] = time.monotonic() - started
            result["cleanup_complete"] = all(c["cleanup_complete"] for c in result["cases"])
            result["known_actual_counts"] = {key: sum(c["known_actual_counts"].get(key, 0)
                for c in result["cases"])
                for key in ("policy_transitions", "native_steps", "pd_updates", "forward_calls", "resets")}
            result["actual_counts"] = ({key: sum(c["actual_counts"].get(key, 0) for c in result["cases"])
                for key in result["known_actual_counts"]}
                if all(c["actual_counts"] is not None for c in result["cases"]) else None)
            write_json(output / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="read-only provenance checks, no physics")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--owned-scope")
    args = parser.parse_args(argv)
    if args.check_only:
        identity, _, _, _ = load_inputs()
        print(json.dumps({"status": "PASS", "files_checked": len(identity["files_sha256"]),
                          "physics_steps": 0, "inference_calls": 0}, indent=2))
        return 0
    require(args.output is not None and args.owned_scope, "--output and --owned-scope required")
    result = execute(args.output.resolve(), args.owned_scope)
    print(json.dumps({k: result[k] for k in ("status", "error", "actual_counts", "elapsed_s")}, indent=2))
    return int(result["status"] != "COMPLETE")


if __name__ == "__main__":
    raise SystemExit(main())
