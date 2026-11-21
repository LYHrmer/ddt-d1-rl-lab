"""Publish the exact sealed measurements and movie. Existing different files are preserved."""
import argparse
import os
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

from tools91.launch91 import require
from tools94.common94 import ROOT, digest, read, write
from tools100.common100 import (PLAN_SHA, PUBLIC_CLOUDS, PUBLIC_MANIFEST, PUBLIC_MOVIE_RECEIPT,
                                PUBLIC_REPORT, PUBLIC_VIDEO, protocol)
from tools100.launch100 import completed
from tools100.readback100 import verify_pair


def copy_verified(source, target):
    checksum = digest(source)
    if not target.exists():
        with NamedTemporaryFile(dir=target.parent) as temporary:
            with source.open("rb") as stream:
                copyfileobj(stream, temporary)
            temporary.flush()
            require(digest(temporary.name) == checksum, "incomplete artifact copy")
            os.fchmod(temporary.fileno(), 0o644)
            try:
                os.link(temporary.name, target)
            except FileExistsError:
                pass
    require(digest(target) == checksum, f"preserving different file at {target}")
    return checksum


def publish(desktop_path=None):
    plan = protocol(private=True)
    replay, replay_limits = completed("replay", plan)
    video, video_limits = completed("video", plan)
    cloud = replay / "clouds.npz"
    record = read(replay / "report.json")
    require(verify_pair(plan, record, cloud) == record["instrument_checks"], "independent verification failed")
    receipt = read(video / "video.json")
    movie = video / "pointcloud100.mp4"
    require(receipt["status"] == "COMPLETE" and receipt["protocol_sha256"] == PLAN_SHA
            and receipt["video_sha256"] == digest(movie) and receipt["report_sha256"] == digest(replay / "report.json")
            and receipt["cloud_sha256"] == digest(cloud), "movie is not from the sealed measurements")
    assets = [(replay / "report.json", PUBLIC_REPORT), (cloud, PUBLIC_CLOUDS), (movie, PUBLIC_VIDEO),
              (video / "video.json", PUBLIC_MOVIE_RECEIPT)]
    manifest = {"schema": 1, "protocol_sha256": PLAN_SHA,
                "files": {str(target.relative_to(ROOT)): copy_verified(source, target) for source, target in assets},
                "seals": {stage: digest(folder / "seal.json") for stage, folder in (("replay", replay), ("video", video))},
                "producers": {stage: read(folder / "seal.json")["sources"] for stage, folder in (("replay", replay), ("video", video))},
                "resources": {"replay": replay_limits, "video": video_limits}}
    if not PUBLIC_MANIFEST.exists():
        write(PUBLIC_MANIFEST, manifest)
    require(read(PUBLIC_MANIFEST) == manifest, "existing publication differs")
    if desktop_path is not None:
        copy_verified(movie, desktop_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desktop-path", type=Path)
    args = parser.parse_args()
    publish(args.desktop_path)
