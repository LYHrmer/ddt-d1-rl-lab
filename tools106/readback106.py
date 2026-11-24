"""Seal readings before opening reference heights and evaluate the registered capture history."""
import numpy as np

from tools94.common94 import ROOT, arrays, digest, read, write
from tools101.score101 import qualifying, require
from tools104.motion104 import reference_motion
from tools105 import score105
from tools106.footprint106 import exit_footprints
from tools106.history106 import history_inputs, oracle_capture_delivery
from tools106.history_readout import after_abstention, packet_readouts


PUBLIC = ROOT / "reports/history106"
PLAN = ROOT / "docs/experiments106/history.json"
COLUMNS = ("split", *score105.IDS, "single_mm", "history_mm")


def measured_predictions(measured_path):
    parts = []
    for split in score105.SEEDS:
        rows, points, queries, previous, motion = history_inputs(split, measured_path)
        single, history = packet_readouts(points, queries, previous, motion)
        parts.append(dict(**rows, split=np.repeat(split, len(single)), single_mm=single, history_mm=history))
    return {k: np.concatenate([p[k] for p in parts]) for k in COLUMNS}


def counts(truth, values):
    passed = qualifying(truth, values)
    available = np.isfinite(values)
    return dict(n=len(truth), correct=int(passed.sum()), unknown=int((~available).sum()),
                wrong_numeric=int((available & ~passed).sum()),
                flat_false_positive=int(((truth == 0) & available & ~passed).sum()))


def compare(truth, single, arms, indices):
    reference = single[indices]
    y = truth[indices]
    result = {"single": counts(y, reference)}
    for k, values in arms.items():
        v = values[indices]
        added = ~np.isfinite(reference) & np.isfinite(v)
        correct = qualifying(y, v)
        result[k] = dict(**counts(y, v), filled_correct=int((added & correct).sum()),
                         filled_wrong=int((added & ~correct).sum()))
    return result


def groups(series, truth):
    first = np.r_[True, series[1:] != series[:-1]]
    switch = ~first & (np.roll(truth, 1) != truth)
    return dict(all=np.ones(len(truth), dtype=bool), first=first, enter=switch & (truth != 0),
                exit=switch & (truth == 0), stable=~first & ~switch)


def split_scores(rows, single, arms):
    truth = rows["target_mm"]
    result = {}
    masks = groups(rows["series_id"], truth)
    for seed in np.unique(rows["source_seed"]):
        views = {}
        for visibility in ("clear", "masked"):
            selected = (rows["source_seed"] == seed) & (rows["visibility"] == visibility)
            views[visibility] = {k: compare(truth, single, arms, selected & m) for k, m in masks.items()}
        result[str(seed)] = views
    return result


def diagnose(folder, predicted):
    archived = score105.load_predictions(score105.PUBLIC / "predictions.npz")
    for k in ("split", *score105.IDS):
        require(np.array_equal(archived[k], predicted[k]), "changed #105 packet identities")
    np.testing.assert_allclose(archived["candidate_mm"], predicted["single_mm"], rtol=0., atol=1e-7, equal_nan=True)
    single, history = predicted["single_mm"], predicted["history_mm"]
    known = np.isfinite(single)
    require(np.array_equal(single[known], history[known]), "candidate overwrote current evidence")
    joined = score105.with_truth(archived)
    results, footprints = {}, []
    plan = read(PLAN)
    for split in score105.SEEDS:
        mask = archived["split"] == split
        rows = {k: v[mask] for k, v in joined.items()}
        current = single[mask]
        oracle, carry = current.copy(), current.copy()
        ids, points, queries, prior, motion = history_inputs(split, folder / "measured.npz")
        poses = oracle_capture_delivery(ids, folder / "reference101.npz")
        for i in np.flatnonzero(~np.isfinite(current) & (prior >= 0)):
            j = prior[i]
            estimate = after_abstention(None, queries[i], points[j], reference_motion(poses[[j, i], 0]))
            if estimate is not None:
                oracle[i] = estimate
            carry[i] = current[j]
        arms = dict(history=history[mask], oracle=oracle, carry=carry, nearest=rows["baseline_mm"])
        results.update(split_scores(rows, current, arms))
        if split == "validation":
            y = rows["target_mm"]
            exits = groups(rows["series_id"], y)["exit"]
            missing = np.flatnonzero(exits & (rows["visibility"] == "masked") & ~np.isfinite(current))
            require(missing.tolist() == plan["known_exit_row_ids"], "old failure population changed")
            footprints = exit_footprints(rows, prior, motion, queries, poses, missing)
            for entry, i in zip(footprints, missing, strict=True):
                for key in ("history", "oracle", "carry"):
                    value = arms[key][i]
                    entry[key + "_mm"] = None if not np.isfinite(value) else float(value)
    checks = {}
    for seed in score105.SEEDS["validation"]:
        views = results[str(seed)]
        checks[str(seed)] = dict(masked_exit_correct_fill=views["masked"]["exit"]["history"]["filled_correct"] >= 1,
                                 no_wrong_fill=all(views[v]["all"]["history"]["filled_wrong"] == 0 for v in views))
    # Reuse the original qualification gate against the same nearest-return comparator.
    graded = score105.score({**joined, "candidate_mm": history})
    return dict(per_source=results, retrospective_checks=checks,
                checked_sources_passed=sum(all(v.values()) for v in checks.values()),
                original_gate_against_nearest=graded["gate"], old_masked_exit_footprints=footprints)


def external_scores(path):
    from tools106.geometry100 import geometry_rows
    meta, points, queries, previous, motions = [], [], [], [], []
    for info, cloud, query in geometry_rows(path):
        i = len(meta)
        same = bool(meta) and meta[-1]["series"] == info["series"]
        require(same == (info["reference_motion"] is not None), "wrong external series boundary")
        previous.append(i - 1 if same else -1)
        motions.append(info["reference_motion"])
        meta.append(info)
        points.append(cloud)
        queries.append(query)
    require(len(meta) == 216, "incomplete external grid")
    single, history = packet_readouts(points, queries, previous, motions)
    series = np.cumsum(np.asarray(previous) < 0)
    rows = {"target_mm": np.asarray([r["target_mm"] for r in meta]), "series_id": series,
            "source_seed": np.asarray([r["seed"] for r in meta]), "visibility": np.asarray([r["visibility"] for r in meta])}
    return split_scores(rows, single, {"oracle_history": history})


def record(folder):
    path = folder / "predictions.npz"
    arrays(path, measured_predictions(folder / "measured.npz"))
    before_join = digest(path)
    with np.load(path, allow_pickle=False) as saved:
        report = diagnose(folder, {k: saved[k] for k in COLUMNS})
    external = external_scores(folder / "reference100.npz")
    require(digest(path) == before_join, "scoring modified the sealed readings")
    write(folder / "report.json", dict(status="COMPLETE", predictions_sha256=before_join,
          **report, external_reference_pose_only=external,
          native_decisions=0, new_rays=0, model_updates=0, policy_updates=0))
