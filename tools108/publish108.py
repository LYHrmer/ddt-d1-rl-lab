"""Publish the contact-response evidence and a byte-identical copy of the recorded movie."""
import argparse
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np

from tools91.launch91 import require
from tools94.common94 import ROOT, arrays, digest, read, write
from tools108.run108 import BASE, NATIVE_FILES, PLAN_SHA, STAGES, completed, locations, protocol


PUBLIC = ROOT / "reports/contact108"
MOVIE = ROOT / "docs/assets/contact108.mp4"
DESKTOP_NAME = "D1_三厘米台阶触阶实测_原动作与前腿屈曲_第108轮.mp4"


def copy_new(source, destination):
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    require(digest(source) == digest(destination), "published copy changed")


def publish():
    from tools108.diagnose108 import diagnose
    from tools108.preview108 import preview
    from tools108.readback108 import check

    plan = protocol()
    paths, resources, seals, hashes = {}, {}, {}, {}
    for stage in STAGES:
        folder, usage = completed(stage)
        _, guard, unit = locations(stage, int(folder.name.rsplit("_", 1)[1]))
        paths[stage] = folder
        resources[stage] = {**usage, **plan["resources"], "unit":unit, "guard_sha256":digest(guard/"report.json")}
        seals[stage], hashes[stage] = read(folder/"seal.json"), digest(folder/"seal.json")
    fit_seal = read(paths["blind"]/"fit_binding.json")["fit_seal_sha256"]
    require(fit_seal == hashes["fit"], "blind acquisition used a different fit")
    require(not PUBLIC.exists() and not MOVIE.exists(), "preserving the prior public evidence")
    video = paths["video"]/"step108.mp4"
    require(read(paths["video"]/"video.json")["sha256"] == digest(video), "invalid movie")
    pairs = {
        "development/contacts.json": ("development","contacts.json"),
        "development/measurements.npz": ("development","measurements.npz"),
        "blind/contacts.json": ("score","contacts.json"), "blind/measurements.npz": ("score","measurements.npz"),
        "fit/fit.json": ("fit","fit/fit.json"), "fit/models.npz": ("fit","fit/models.npz"),
        "score/score.json": ("score","score.json"), "score/predictions.npz": ("score","forecast/predictions.npz"),
        "score/forecast_seal.json": ("score","forecast/forecast_seal.json"), "video.json": ("video","video.json")
    }
    with TemporaryDirectory(dir=PUBLIC.parent, prefix=".contact108-") as temp:
        out = Path(temp)/"contact108"
        out.mkdir()
        for relative, (stage, archived) in pairs.items():
            target = out/relative
            target.parent.mkdir(exist_ok=True)
            copy_new(paths[stage]/archived, target)
        for guard in sorted(BASE.glob("guard_*/report.json")):
            if read(guard)["status"] != "COMPLETE":
                failures = out/"failed_attempts"
                failures.mkdir(exist_ok=True)
                copy_new(guard, failures/(guard.parent.name+".json"))
        for split in ("development", "blind"):
            samples, ticks, terrain_contacts, motion = [], [], {}, {}
            for seed in plan[split]:
                for arm in plan["arms"]:
                    label = f"seed{seed}_{arm}"
                    with np.load(paths[split]/"episodes"/label/"trace.npz", allow_pickle=False) as trace:
                        terrain_contacts[label] = trace["step_contacts"]
                        positions = trace["qpos"]
                        motion[label+"_calves_rad"] = positions[:, [9, 13]]
                        motion[label+"_wheel_z_m"] = trace["wheel_xyz_m"][:, :2, 2]
                        motion[label+"_base_z_m"] = positions[:, 2]
                        motion[label+"_base_quat"] = positions[:, 3:7]
                        if arm == "original":
                            k = int(trace["trigger_tick"])
                            require(k >= 1, "missing original pre-contact observation")
                            ticks.append(k)
                            observations = trace["actor_observation"]
                            samples.append(observations[k-1])
                            deliveries = np.flatnonzero(trace["sensor_delivery_tick"][:k] == np.arange(k))
                            captures = trace["sensor_capture_tick"][deliveries]
                            require(np.all(captures >= 0) and np.all(deliveries - captures == 2),
                                    "invalid optical timestamps")
                            motion[label+"_packet_ticks"] = np.column_stack((captures, deliveries))
                            motion[label+"_range_features"] = observations[deliveries, 627:]
                            motion[label+"_capture_pose"] = positions[captures, :7]
                            motion[label+"_delivery_pose"] = positions[deliveries, :7]
            arrays(out/split/"step_contacts.npz", terrain_contacts)
            arrays(out/split/"motion_observations.npz", motion)
            arrays(out/split/"onset.npz", {"resets":plan[split], "trigger_ticks":ticks, "observations":samples})
            write(out/split/"physical_readback.json", diagnose(paths[split]/"episodes", plan, split))
            write(out/split/"preview_readback.json", preview(paths[split]/"episodes", plan, split))
        write(out/"resources.json", resources)
        source_paths = {*NATIVE_FILES, "tools77/geometry77.py", "tools100/sensor100.py",
                        "docs/experiments100/pointcloud_density.json", "training/d1_rl/task03d.py",
                        *(str(p.relative_to(ROOT)) for p in (ROOT/"tools108").glob("*.py"))}
        write(out/"provenance.json", {
            "plan_sha256":PLAN_SHA, "sources_sha256":{name:digest(ROOT/name) for name in sorted(source_paths)},
            "stage_seal_sha256":hashes, "stage_seals":seals, "blind_fit_seal_sha256":fit_seal,
            "public_files_sha256":{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob("*")) if p.is_file()},
            "film":{"path":str(MOVIE.relative_to(ROOT)), "sha256":digest(video)}
        })
        check(out)
        out.rename(PUBLIC)
    copy_new(video, MOVIE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desktop", type=Path, help="place a verified copy in this directory after publishing")
    args = parser.parse_args()
    publish()
    if args.desktop:
        copy_new(MOVIE, args.desktop/DESKTOP_NAME)
