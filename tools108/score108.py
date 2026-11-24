"""Read the actual contact, actuator FIFO and axle/base displacement from each matched D1 pair."""
from pathlib import Path

import numpy as np

from tools91.launch91 import require
from tools94.common94 import arrays, digest, read, write


def measurements(path, meta, plan):
    require(digest(path) == meta["trace_sha256"], "trajectory digest mismatch")
    with np.load(path, allow_pickle=False) as a:
        ticks = meta["ticks_completed"]
        forces = a["step_force_n"]
        times = a["time_s"]
        require(forces.shape == (10 * ticks, 4) and times.shape == (ticks + 1,)
                and np.allclose(times, np.arange(ticks + 1) * .02, rtol=0, atol=1e-9), "clock mismatch")
        first = np.flatnonzero((forces[:, :2] >= plan["trigger_n"]).any(axis=1))
        sub = int(first[0]) if len(first) else None
        k = None if sub is None else sub // 10 + 1
        require(sub == meta["first_contact_substep"] and k == meta["trigger_tick"], "first contact mismatch")
        nominal, issued, applied = (a[name] for name in ("nominal_action", "issued_action", "applied_action"))
        require(np.array_equal(issued[:-1], applied[1:]), "existing motor FIFO changed")
        offset = (issued - nominal) * .5
        require(not offset[:, [0,1,3,4,5,7,8,9,10,11,12,13,14,15]].any(), "perturbed another actuator")
        want = np.zeros_like(issued)
        if meta["arm"] == "flexion" and k is not None:
            want[k:k + plan["pulse_ticks"], [2, 6]] = plan["pulse_rad"] / .5
        require(np.allclose(issued, np.clip(nominal + want, -1., 1.), atol=2e-7, rtol=0),
                "issued actions disagree with registered intervention")
        end = None if k is None else k + plan["horizon"]
        ok = end is not None and end <= ticks
        # A loaded wheel above the front edge on the horizontal face is stronger evidence
        # than axle projection or a wheel pushing on the vertical wall.
        raw = a["step_contacts"]
        geo = plan["geometry"]
        top = raw[(raw[:, 1] >= 0) & (raw[:, 3] >= plan["trigger_n"]) & (raw[:, 4] >= geo["front_x_m"])
                  & (raw[:, 6] >= geo["height_m"] - .002) & (raw[:, 9] >= .9)]
        wheel_top = [bool((top[:, 1] == i).any()) for i in range(4)]
        row = dict(seed=meta["seed"], arm=meta["arm"], status=meta["status"], ticks=ticks,
                   terminal_reason=meta["terminal_reason"], trigger_tick=k, first_contact_s=(sub+1)*.002 if sub is not None else None,
                   max_front_contact_force_n=float(forces[:, :2].max()), window_complete=ok, top_by_wheel=wheel_top,
                   fallen=bool(a["fallen"].any()), illegal_contact=bool(a["illegal_contact"].any()),
                   max_nonwheel_step_force_n=float(a["nonwheel_step_max_n"].max()),
                   axle_mm=None, base_mm=None, changed_knee_ticks=int((np.abs(offset[:, 2]) > 1e-6).sum()),
                   final_axle_x_m=float(a["front_axle_x_m"][-1]), final_base_x_m=float(a["qpos"][-1, 0]),
                   sha256=meta["trace_sha256"])
        if ok:
            row.update(axle_mm=1000 * float(a["front_axle_x_m"][end] - a["front_axle_x_m"][k]),
                       base_mm=1000 * float(a["qpos"][end, 0] - a["qpos"][k, 0]))
        prefix = None if not ok else {name: a[name][:k+1] for name in ("qpos", "qvel", "actor_observation")}
        share = dict(time_s=times, axle_x_m=a["front_axle_x_m"], base_x_m=a["qpos"][:, 0],
                     step_force_n=forces, knee_offset_rad=offset[:, [2, 6]], fallen=a["fallen"],
                     illegal_contact=a["illegal_contact"])
        return row, prefix, share


def collect_summary(episodes, out, plan, split):
    index = read(episodes / "index.json")
    require(index["reset_seeds"] == plan[split] and index["arms"] == plan["arms"], "unregistered reset roster")
    rows, pairs, audit = [], [], {}
    for seed in plan[split]:
        signals = []
        for arm in plan["arms"]:
            key = f"seed{seed}_{arm}"
            m = read(episodes / key / "report.json")
            require(m["seed"] == seed and m["arm"] == arm and m["geometry"] == plan["geometry"]
                    and m["checkpoint_sha256"] == plan["checkpoint_sha256"], "wrong rollout")
            row, prefix, data = measurements(episodes / key / m["trace_file"], m, plan)
            rows.append(row)
            signals.append(prefix)
            audit.update({f"{key}_{name}": val for name,val in data.items()})
        lo, hi = rows[-2:]
        matched = bool(all(s is not None for s in signals) and lo["trigger_tick"] == hi["trigger_tick"]
                       and all(np.array_equal(signals[0][key], signals[1][key]) for key in signals[0]))
        paired = bool(matched and lo["status"] == hi["status"] == "COMPLETE" and lo["changed_knee_ticks"] == 0
                      and hi["changed_knee_ticks"] == plan["pulse_ticks"])
        pairs.append(dict(seed=seed, matched_prefix=matched, qualified=paired,
                          axle_effect_mm=hi["axle_mm"] - lo["axle_mm"] if matched else None,
                          base_effect_mm=hi["base_mm"] - lo["base_mm"] if matched else None))
    result = dict(split=split, data_eligible=all(p["qualified"] for p in pairs), episodes=rows, pairs=pairs,
                  native_decisions=sum(r["ticks"] for r in rows), physics_substeps=10 * sum(r["ticks"] for r in rows),
                  archival_index_sha256=digest(episodes / "index.json"))
    out = Path(out)
    arrays(out / "measurements.npz", audit)
    write(out / "contacts.json", result)
    return result
