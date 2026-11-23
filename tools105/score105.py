"""Replay two height readouts on identical measured returns and score sealed row predictions."""
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import numpy as np

from tools94.common94 import ROOT, arrays, digest, read, write
from tools100.score100 import TOLERANCE_MM, estimate_delta_mm
from tools101.score101 import TICKS, VISIBILITIES, paired, qualifying, require


PUBLIC = ROOT / "reports/surface105"
PROTOCOL = ROOT / "docs/experiments105/support_surface.json"
DESIGN_WINDOWS = ROOT / "reports/motion104/windows.npz"
PACKETS = {"design": ROOT / "reports/jeplo101/holdout.npz",
           "validation": ROOT / "reports/jeplo101/train.npz"}
PACKET_SHA = {"design": "fa518b23665ead8c1f77288aef163285bde85e8bdcfab3d2383e8762d3dd8223",
              "validation": "a73f2838e37ae94f7ac23a63768710289998f688e4261f63beef09e754e82907"}
SEEDS = {"design": (89037,), "validation": (89011, 89023)}
EPISODES = {"design": (0, 3, 6, 9, 12), "validation": (2, 5, 8, 11)}
LAYOUTS = {"design": ("unseen_left", "unseen_right"),
           "validation": ("narrow_left", "narrow_right", "long_left", "long_right")}
IDS = ("row_id", "series_id", "cloud_id", "source_seed", "episode", "layout", "side",
       "visibility", "capture_tick", "delivery_tick")
INTEGER_IDS = ("row_id", "series_id", "cloud_id", "source_seed", "episode", "capture_tick", "delivery_tick")
ARMS = ("baseline_mm", "candidate_mm")
PREDICTION_KEYS = ("split", *IDS, *ARMS)
REPLAY_ATOL_MM = 1e-7


def readouts(rows, points, queries, baseline, candidate):
    """Call each readout with exactly the same returns and two queries, independent of row identities."""
    require(set(rows) == set(IDS), "expected only packet identity and clock columns")
    n = len(rows["row_id"])
    require(all(rows[k].shape == (n,) for k in IDS), "invalid packet identity column")
    require(points.shape == (n, 520, 3) and queries.shape == (n, 2, 2)
            and points.dtype == queries.dtype == np.float64, "invalid measured returns or wheel queries")
    require(not np.isinf(points).any() and np.isfinite(queries).all(), "invalid measurement")
    points.setflags(write=False)
    queries.setflags(write=False)
    output = dict(rows)
    for name, reader in zip(ARMS, (baseline, candidate), strict=True):
        values = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            estimate = reader(points[i], queries[i])
            if estimate is not None:
                require(isinstance(estimate, (float, np.floating)) and np.isfinite(estimate),
                        f"{name} must return finite millimetres or None")
                values[i] = estimate
        output[name] = values
    return output


def predictions(validation_windows):
    from tools105.data105 import inputs
    from tools105.surface105 import surface_delta_mm

    parts = []
    for split in SEEDS:
        rows, points, queries = inputs(split, DESIGN_WINDOWS if split == "design" else validation_windows)
        part = readouts(rows, points, queries, estimate_delta_mm, surface_delta_mm)
        part["split"] = np.repeat(split, len(part["row_id"]))
        parts.append(part)
    return {key: np.concatenate([part[key] for part in parts]) for key in PREDICTION_KEYS}


def load_predictions(path):
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == set(PREDICTION_KEYS), "invalid sealed prediction columns")
        return {key: archive[key] for key in PREDICTION_KEYS}


def with_truth(predicted):
    """Join signed targets after inference and check every prediction against its archived packet identity."""
    labels = np.empty(len(predicted["row_id"]), dtype=np.float64)
    require(set(np.unique(predicted["split"])) == set(PACKETS), "missing or unregistered prediction split")
    for split, path in PACKETS.items():
        require(digest(path) == PACKET_SHA[split], "changed round-101 packet archive")
        selected = predicted["split"] == split
        with np.load(path, allow_pickle=False) as archive:
            for key in IDS:
                require(np.array_equal(predicted[key][selected], archive[key]), f"{split} {key} alignment changed")
            labels[selected] = archive["target_mm"]
    return {**predicted, "target_mm": labels}


def _check_roster(rows):
    n = len(rows["row_id"])
    require(set(rows) == set((*PREDICTION_KEYS, "target_mm"))
            and all(v.shape == (n,) for v in rows.values()), "unexpected row columns or shapes")
    require(all(rows[k].dtype.kind in "iu" for k in INTEGER_IDS), "integer identities and clocks required")
    require(all(rows[k].dtype.kind == "U" for k in ("split", "layout", "side", "visibility")),
            "invalid text identities")
    require(all(rows[k].dtype == np.float64 and not np.isinf(rows[k]).any() for k in ARMS),
            "predictions must be float64 with NaN only for unknown")
    truth = rows["target_mm"]
    require(truth.dtype.kind == "f" and np.isfinite(truth).all(), "invalid target")
    require(np.array_equal(rows["delivery_tick"], rows["capture_tick"] + 2), "changed packet latency")
    require(set(np.unique(rows["split"])) == set(SEEDS), "missing or unregistered split")
    for split, seeds in SEEDS.items():
        ii = np.flatnonzero(rows["split"] == split)
        streams = list(product(seeds, EPISODES[split], LAYOUTS[split], VISIBILITIES))
        m = len(streams)
        require(np.array_equal(rows["row_id"][ii], np.arange(m * len(TICKS)))
                and np.array_equal(rows["series_id"][ii], np.repeat(np.arange(m), len(TICKS)))
                and np.array_equal(rows["capture_tick"][ii], np.tile(TICKS, m)), "incomplete or reordered sequence")
        for k, column in enumerate(("source_seed", "episode", "layout", "visibility")):
            expected = np.repeat([stream[k] for stream in streams], len(TICKS))
            require(np.array_equal(rows[column][ii], expected), f"{split} {column} roster changed")
        side = np.array([layout.rsplit("_", 1)[1] for layout in rows["layout"][ii]])
        raised = np.where(side == "left", 30., -30.)
        require(np.array_equal(rows["side"][ii], side) and np.all((truth[ii] == 0) | (truth[ii] == raised)),
                "invalid signed side or height")
        # #101 captures a cloud once per (seed, episode, layout, tick), then applies both visibility masks.
        cloud_grid = np.arange(m // 2 * len(TICKS)).reshape(-1, len(TICKS)).repeat(2, axis=0)
        require(np.array_equal(rows["cloud_id"][ii], cloud_grid.ravel()), "invalid paired cloud identity")
        twin_labels = truth[ii].reshape(-1, 2, len(TICKS))
        require(np.array_equal(twin_labels[:, 0], twin_labels[:, 1]), "visibility changed physical truth")


def _groups(rows, selected):
    ii = np.flatnonzero(selected)
    series = rows["series_id"][ii]
    truth = rows["target_mm"][ii]
    first = np.r_[True, series[1:] != series[:-1]]
    switch = ~first & (truth != np.roll(truth, 1))
    return {"all_arrivals": ii, "first": ii[first], "switch": ii[switch],
            "enter": ii[switch & (truth != 0)], "exit": ii[switch & (truth == 0)],
            "stable": ii[~first & ~switch]}


def _paired(rows, ii):
    result = paired(rows["target_mm"][ii], rows["baseline_mm"][ii], rows["candidate_mm"][ii])
    result["candidate"] = result.pop("predictive")
    result["candidate_mae_both_mm"] = result.pop("predictive_mae_both_mm")
    for arm in ("baseline", "candidate"):
        stats = result[arm]
        stats["unknown"] = stats.pop("missing")
        stats["raised_unknown"] = stats.pop("raised_missing")
    result["packet_pairs"] = len(ii)
    result["distinct_native_windows"] = len(set(zip(rows["episode"][ii], rows["capture_tick"][ii], strict=True)))
    return result


def gate(views):
    masked = views["masked"]
    clear = views["clear"]["all_arrivals"]
    switch, enter, exit_, stable = (masked[name] for name in ("switch", "enter", "exit", "stable"))
    return {
        "masked_switch_strict_gain": switch["candidate"]["passed"] > switch["baseline"]["passed"],
        "masked_enter_no_loss": enter["candidate"]["passed"] >= enter["baseline"]["passed"],
        "masked_exit_no_loss": exit_["candidate"]["passed"] >= exit_["baseline"]["passed"],
        "masked_stable_no_extra_failures": stable["candidate"]["failed"] <= stable["baseline"]["failed"],
        "clear_all_no_extra_failures": clear["candidate"]["failed"] <= clear["baseline"]["failed"],
    }


def score(rows):
    _check_roster(rows)
    per_seed = {}
    for split, seeds in SEEDS.items():
        for seed in seeds:
            selected = (rows["split"] == split) & (rows["source_seed"] == seed)
            views = {}
            for vis in VISIBILITIES:
                groups = _groups(rows, selected & (rows["visibility"] == vis))
                views[vis] = {name: _paired(rows, ii) for name, ii in groups.items()}
            checks = gate(views) if split == "validation" else None
            per_seed[str(seed)] = {
                "split": split, "native_episodes": list(EPISODES[split]),
                "physical_clouds": len(np.unique(rows["cloud_id"][selected])),
                "distinct_native_windows": len(EPISODES[split]) * len(TICKS),
                "packet_pairs": int(selected.sum()), "visibility": views, "checks": checks,
                "passed": all(checks.values()) if checks is not None else None,
            }
    votes = [per_seed[str(seed)]["passed"] for seed in SEEDS["validation"]]
    window_count = sum(item["distinct_native_windows"] for item in per_seed.values())
    return {
        "tolerance_mm": TOLERANCE_MM,
        "definitions": {
            "switch": "A later arrival in the same series with a changed signed reference. Enter and exit partition it.",
            "stable": "A later arrival with the previous signed reference. All arrivals also include every first packet.",
            "errors": "Unknown always fails. Raised missed side includes called-flat, wrong-side and unknown. "
                      "Correct side outside the tolerance is a separate magnitude failure. "
                      "A flat false positive is an available estimate outside the inclusive flat band.",
            "unit": "One paired comparison per rendered packet view. Layouts and visibility share native motion.",
        },
        "per_seed": per_seed,
        "gate": {"validation_seeds": list(SEEDS["validation"]), "design_seeds": list(SEEDS["design"]),
                 "seeds_passed": sum(votes), "seeds_required": len(votes), "passed": all(votes)},
        "counts": {"packet_pairs": len(rows["row_id"]), "physical_clouds": sum(v["physical_clouds"] for v in per_seed.values()),
                   "native_episodes": sum(len(v["native_episodes"]) for v in per_seed.values()),
                   "distinct_native_windows": window_count, "new_native_decisions": 0, "new_physics_steps": 0,
                   "new_sensor_rays": 0, "model_updates": 0, "policy_updates": 0},
    }


def score_file(path):
    return score(with_truth(load_predictions(path)))


def _report(folder, result, sealed_sha):
    return {"status": "COMPLETE", "protocol_sha256": digest(PROTOCOL), "predictions_sha256": sealed_sha,
            "packet_archives_sha256": dict(PACKET_SHA), "design_windows_sha256": digest(DESIGN_WINDOWS),
            "validation_windows_sha256": digest(folder / "windows.npz"), **result}


def record(folder):
    path = folder / "predictions.npz"
    report_path = folder / "report.json"
    require(folder.is_dir() and (folder / "windows.npz").is_file(), "missing validation motion windows")
    require(not path.exists() and not report_path.exists(), "refusing to overwrite predictions or a report")
    arrays(path, predictions(folder / "windows.npz"))
    sealed_sha = digest(path)
    result = score_file(path)
    require(digest(path) == sealed_sha, "predictions changed during reference scoring")
    report = _report(folder, result, sealed_sha)
    write(report_path, report)
    return report


def check(folder):
    path = folder / "predictions.npz"
    report = read(folder / "report.json")
    sealed_sha = digest(path)
    require(report["predictions_sha256"] == sealed_sha, "changed sealed predictions")
    saved = load_predictions(path)
    replay = predictions(folder / "windows.npz")
    for key in PREDICTION_KEYS:
        left, right = saved[key], replay[key]
        require(left.shape == right.shape and left.dtype == right.dtype, f"replay {key} format changed")
        if key in ARMS:
            require(np.allclose(left, right, rtol=0., atol=REPLAY_ATOL_MM, equal_nan=True), f"replay {key} changed")
        else:
            require(np.array_equal(left, right), f"replay identity {key} changed")
    joined = with_truth(saved)
    truth = joined["target_mm"]
    for key in ARMS:
        require(np.array_equal(qualifying(truth, saved[key]), qualifying(truth, replay[key])),
                f"replay changed a row grade in {key}")
        require(np.array_equal(saved[key] > TOLERANCE_MM, replay[key] > TOLERANCE_MM)
                and np.array_equal(saved[key] < -TOLERANCE_MM, replay[key] < -TOLERANCE_MM),
                f"replay changed a reported raised side in {key}")
    require(_report(folder, score(joined), sealed_sha) == report, "archived inputs, counts, votes or scores changed")
    return report


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", type=Path, metavar="DIR")
    mode.add_argument("--check", type=Path, nargs="?", const=PUBLIC, metavar="DIR")
    args = parser.parse_args()
    result = record(args.record) if args.record is not None else check(args.check)
    print("Verified paired height readouts:", result["counts"]["packet_pairs"], "packets,",
          result["gate"]["seeds_passed"], "/", result["gate"]["seeds_required"], "validation seeds passed")
