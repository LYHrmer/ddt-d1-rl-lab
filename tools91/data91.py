"""Admit complete training episodes and export the six pre-decision branch points."""
from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from tools87.run87 import ROOT, read, sha
from training.d1_rl.interface import HISTORY_LENGTH, build_observation, pack_actor_input


FIT_KEYS = ("qpos", "qvel", "applied", "issued", "episode_tick", "terrain", "terminated", "truncated")
START_KEYS = ("state45", "observation627", "previous_filtered", "fifo16", "start_tick", "parent_seed", "reset_seed")
PLAN87 = ROOT / "docs/experiments87/full_body_perception.json"
PLAN88 = ROOT / "docs/experiments88/uphill_wheel_probe.json"
PLAN89 = ROOT / "docs/experiments89/uphill_curriculum.json"
PLAN90 = ROOT / "docs/experiments90/stand_wbc.json"

# Published in reports/uphill_{curriculum89,diagnosis88}.json and reports/stand_wbc90.json.
FIT_STAGE_SHA = "2c518e335cae080d055042c530ff62f52fd3046e6e191929980e1717eb18d4b3"
MOVE_STAGE_SHA = "71549890935482997a61b0fbf83945084f480a33acb207c7c4b10f3f94839b50"
STAND_STAGE_SHA = "0008db012bd3ad3fb9923cfa22029d814c84c681912bfc5ebf65727e16f3914b"


def _check(ok, message):
    if not ok:
        raise ValueError(message)


def _digest(path, expected):
    _check(not path.is_symlink() and sha(path) == expected, f"archive SHA256 mismatch: {path}")


def _stage(folder, expected, plan):
    source = folder / "report.json"
    _digest(source, expected)
    receipt = read(source)
    _check(receipt["status"] == "COMPLETE" and receipt["protocol_sha256"] == sha(plan),
           f"incomplete or unregistered stage: {folder.name}")
    return receipt


def _episode_pairs(data, first_id):
    """Return transitions between adjacent post-interval rows of completed episodes."""
    q, v = data["qpos"], data["qvel"]
    tick, terrain = data["episode_tick"], data["terrain"]
    issued, applied = data["issued"], data["applied"]
    terminal, time_limit = data["terminated"], data["truncated"]
    n = len(tick)
    _check(n > 0 and q.shape == (n, 23) and v.shape == (n, 22)
           and issued.shape == applied.shape == (n, 16)
           and tick.shape == terrain.shape == terminal.shape == time_limit.shape == (n,),
           "invalid state, action or episode columns")
    _check(np.issubdtype(tick.dtype, np.integer) and np.issubdtype(terrain.dtype, np.integer)
           and terminal.dtype == time_limit.dtype == bool
           and all(np.isfinite(x).all() for x in (q, v, issued, applied))
           and (np.abs(issued) <= 1.).all() and (np.abs(applied) <= 1.).all()
           and np.isin(terrain, [0, 1, 2]).all(), "invalid physical values or episode labels")
    done = terminal | time_limit
    starts = np.flatnonzero(tick == 1)
    _check(np.array_equal(starts, np.r_[0, 1 + np.flatnonzero(done[:-1])]),
           "missing terminal marker or reset boundary")
    _check(not np.any(applied[starts]), "reset FIFO must contain zeros")
    states = np.column_stack((q, v))
    parts, episodes = [], []
    discarded = 0
    for start, stop in zip(starts, np.r_[starts[1:], n], strict=True):
        length = stop - start
        _check(np.array_equal(tick[start:stop], np.arange(1, length + 1))
               and np.all(terrain[start:stop] == terrain[start]), "noncontiguous episode or terrain switch")
        _check(np.array_equal(issued[start:stop-1], applied[start+1:stop]), "one-tick FIFO mismatch")
        if not done[stop - 1]:
            discarded = int(length)
            continue
        episode = first_id + len(episodes)
        episodes.append({"id": episode, "start_row": int(start), "end_row_exclusive": int(stop),
                         "terrain": int(terrain[start]), "terminated": bool(terminal[stop - 1]),
                         "truncated": bool(time_limit[stop - 1]), "transitions": int(length - 1)})
        # Row j is the END of interval j. No initial qpos/qvel was saved in these training traces.
        src, dst = slice(start, stop - 1), slice(start + 1, stop)
        parts.append(dict(s=states[src], a=applied[dst], target=states[dst],
                          episode=np.full(length - 1, episode, dtype=np.int64),
                          tick=tick[src], done=done[dst], terrain=terrain[src]))
    return parts, episodes, discarded


def fit_data():
    """Return (arrays, provenance) using only 89 mixed. Target is the absolute next state45."""
    base = ROOT / "evidence/uphill89/training_01"
    stage = _stage(base, FIT_STAGE_SHA, PLAN89)
    admitted = read(PLAN90)["shadow"]["sources"]  # Prior registration binds the three raw files.
    _check([x["seed"] for x in admitted] == [89011, 89023, 89037], "mixed source seeds differ")
    registered = {m["model"]: m for m in stage["models"]}
    parts, sources, episodes = [], [], []
    for entry in admitted:
        seed = entry["seed"]
        folder = base / f"mixed_{seed}"
        model = registered[folder.name]
        record = folder / "report.json"
        _digest(record, model["report_sha256"])
        receipt = read(record)
        _check(receipt["status"] == "COMPLETE" and receipt["arm"] == "mixed"
               and receipt["training_seed"] == seed and receipt["parent_seed"] == entry["parent"]
               and receipt["protocol_sha256"] == stage["protocol_sha256"]
               and receipt["trace"]["completed_recorded_ticks"] == receipt["budget"] == 4096,
               "unqualified mixed PPO rollout")
        source = folder / "train_trace.npz"
        _digest(source, entry["trace_sha256"])
        with np.load(source, allow_pickle=False) as trace:
            data = {key: trace[key] for key in FIT_KEYS}
        _check(len(data["episode_tick"]) == receipt["budget"], "wrong training trace length")
        chunk, details, discarded = _episode_pairs(data, len(episodes))
        parts.extend(chunk)
        episodes.extend(dict(training_seed=seed, **row) for row in details)
        sources.append(dict(training_seed=seed, parent_seed=entry["parent"], model=folder.name,
                            path=str(source.relative_to(ROOT)), trace_sha256=entry["trace_sha256"],
                            report_sha256=model["report_sha256"], completed_episodes=len(details),
                            discarded_unfinished_rows=discarded))
    arrays = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    _check(len(episodes) == 42 and len(arrays["s"]) == 11766 and arrays["done"].sum() == 42,
           "registered whole-episode cover differs")
    return arrays, {"stage_sha256": FIT_STAGE_SHA, "sources": sources, "episodes": episodes,
                    "transitions": len(arrays["s"]), "completed_episodes": len(episodes),
                    "terminated_episodes": sum(row["terminated"] for row in episodes),
                    "truncated_episodes": sum(row["truncated"] for row in episodes)}


def _prefix(archive, key, rows):
    """Decode only the first rows of a registered C-order numeric .npy member."""
    with archive.open(key + ".npy") as member:
        version = np.lib.format.read_magic(member)
        readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
        _check(version in readers, "unexpected archived npy format")
        shape, fortran, dtype = readers[version](member)
        _check(not fortran and dtype.kind in "fiu" and 0 < rows <= shape[0], "invalid prefix format or extent")
        count = rows * int(np.prod(shape[1:], dtype=np.int64))
        raw = member.read(count * dtype.itemsize)
        _check(len(raw) == count * dtype.itemsize, "truncated source prefix")
        return np.frombuffer(raw, dtype=dtype).reshape((rows, *shape[1:]))


def command(tick):
    """Frozen 50 Hz schedule used by both original controllers."""
    return np.array([0. if tick < 50 else .2, 0., 0.], dtype=np.float64)


def _frame(q, v, tick, last_issued, filtered):
    return build_observation(v[3:6], q[3:7], command(tick), q[7:23], v[6:22], last_issued, filtered)


def _moving_start(source, tick, predict, tolerance):
    # The full-file digest admits the original data. Only this bounded prefix is decoded.
    columns = ("qpos", "qvel", "issued_action", "applied_action", "raw_cpp_action", "command_vx_m_s", "time_s")
    with ZipFile(source) as archive:
        data = {key: _prefix(archive, key, tick + 1) for key in columns}
    q, v, issued = data["qpos"], data["qvel"], data["issued_action"]
    _check(q.shape == (tick + 1, 23) and v.shape == (tick + 1, 22)
           and issued.shape == (tick + 1, 16) and np.isfinite(q).all() and np.isfinite(v).all()
           and np.all(np.abs(issued) <= 1.) and not np.any(v[0]) and not np.any(issued[0])
           and np.array_equal(issued, data["raw_cpp_action"]) and not np.any(data["applied_action"][0])
           and np.array_equal(data["applied_action"][1:], issued[:-1]), "original prefix or FIFO mismatch")
    _check(np.allclose(data["time_s"], np.arange(tick + 1) / 50., atol=1e-9, rtol=0.)
           and np.array_equal(data["command_vx_m_s"][1:], np.array([command(k)[0] for k in range(tick)])),
           "prefix command schedule or clock mismatch")
    filtered = np.zeros(3)
    current, filtered = _frame(q[0], v[0], 0, issued[0], filtered)
    history = np.tile(current, (HISTORY_LENGTH, 1))
    for t in range(1, tick + 1):
        if t == tick:
            # At decision tick-1 the actor issued the action archived at endpoint row tick.
            raw = predict(pack_actor_input(current, history), np.zeros(16, np.float32))
            error = float(np.max(np.abs(issued[t] - raw)))
            _check(error <= tolerance, f"parent action disagreement at last warmup decision: {error}")
        history[:-1] = history[1:].copy()
        history[-1] = current
        current, filtered = _frame(q[t], v[t], t, issued[t], filtered)
    return dict(state45=np.r_[q[-1], v[-1]], observation627=pack_actor_input(current, history),
                previous_filtered=filtered, fifo16=issued[-1]), error


def _standing_start(source, initial_digest):
    # These are the only three reset members written before action inference or the first integration.
    with np.load(source, allow_pickle=False) as data:
        q, v, obs = (data[key] for key in ("initial_qpos", "initial_qvel", "initial_obs"))
    digest = hashlib.sha256(q.tobytes() + v.tobytes() + obs.tobytes()).hexdigest()
    _check(initial_digest == digest and q.shape == (23,) and v.shape == (22,) and obs.shape == (643,)
           and q[0] == 1.10 and not np.any(v) and not np.any(obs[627:]), "standing reset identity differs")
    fifo = np.zeros(16, dtype=np.float32)
    current, filtered = _frame(q, v, 0, fifo, np.zeros(3))
    reconstructed = pack_actor_input(current, np.tile(current, (HISTORY_LENGTH, 1)))
    _check(np.array_equal(reconstructed, obs[:627]), "reset actor history, last action or gyro differs")
    return dict(state45=np.r_[q, v], observation627=obs[:627], previous_filtered=filtered, fifo16=fifo)


def prepare_starts(out):
    """Create a fresh directory with six observation/FIFO branch points and return their provenance."""
    out = Path(out)
    out.mkdir(exist_ok=False)
    from tools87.ppo87 import parent_predictor
    from tools89.common89 import parent_folder, registered_plan

    move_root = ROOT / "evidence/uphill88/probe_01"
    stand_root = ROOT / "evidence/wbc90/stand_01"
    moving, standing = _stage(move_root, MOVE_STAGE_SHA, PLAN88), _stage(stand_root, STAND_STAGE_SHA, PLAN90)
    protocol87 = read(PLAN87)
    course = protocol87["course"]
    _check((course["policy_hz"], course["stand_ticks"], course["command_forward_m_s"], course["action_delay_ticks"])
           == (50, 50, .2, 1) and read(PLAN88)["parent_protocol_sha256"] == sha(PLAN87),
           "parent course or command changed")
    start_tick = read(PLAN88)["issue_start_tick"]
    _check(start_tick == 180, "unexpected moving intervention cut")
    parent_plan = registered_plan()
    predict = parent_predictor()
    tolerance = protocol87["training"]["valid_initial_action_max_abs_error"]
    claims = []
    stages = (("move", moving, move_root), ("stand", standing, stand_root))
    for task, stage, root in stages:
        originals = [case for case in stage["cases"] if case["arm"] == "original"]
        seed_field = "seed" if task == "move" else "reset_seed"
        seeds = [case[seed_field] for case in originals]
        expected = read(PLAN88)["reset_seeds"] if task == "move" else read(PLAN90)["stand"]["reset_seeds"]
        _check(seeds == expected and len(seeds) == 3, "registered branch seeds or order differ")
        for case, seed in zip(originals, seeds, strict=True):
            source = root / f"original_{seed}/trace.npz"
            _digest(source, case["trace_sha256"])
            if task == "move":
                start, error = _moving_start(source, start_tick, predict, tolerance)
                tick, parent_seed = start_tick, 25
                parent_sha = protocol87["parent"]["checkpoint_sha256"]
                check = {"final_prefix_action_max_abs_error": error}
            else:
                start = _standing_start(source, case["initial_state_sha256"])
                tick, parent_seed = 0, case["parent_seed"]
                folder = parent_folder(parent_seed, parent_plan)
                parent_sha = case["policy_sha256"]
                _digest(folder / "policy.zip", parent_sha)
                check = {"initial_state_sha256": case["initial_state_sha256"], "reset_observation_exact": True}
            case_id = f"{task}_{seed}"
            path = out / (case_id + ".npz")
            start.update(start_tick=np.int64(tick), parent_seed=np.int64(parent_seed), reset_seed=np.int64(seed))
            with path.open("xb") as stream:
                np.savez_compressed(stream, **start)
            claims.append({"case": case_id, "source_trace_sha256": case["trace_sha256"],
                           "parent_policy_sha256": parent_sha, "start_sha256": sha(path), **check})
    return {"source_stage_sha256": {"move": MOVE_STAGE_SHA, "stand": STAND_STAGE_SHA}, "starts": claims}
