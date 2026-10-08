#!/usr/bin/env python3
"""Compare the owner's ZIP with its extracted source and prepare a text patch.

With no --output, this is a read-only audit. An explicit new --output writes
only a unified patch; it never edits the ZIP or the extracted third-party tree.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ZIP_PREFIX = "workspace2/DDT_Lab/ddt_ros2_control/"
SOURCE_PREFIX = "src/ddt_ros2_control/"
EXPECTED_CHANGED = (
    "controller/rl_controller/launch/sim_mujoco.launch.py",
    "simulation/mujoco_bridge/mujoco_ros2_control/CMakeLists.txt",
    "simulation/mujoco_bridge/mujoco_ros2_control/src/mujoco_ros2_control_plugin.cpp",
    "simulation/mujoco_bridge/mujoco_ros2_control/src/mujoco_ros2_control.cpp",
    "simulation/mujoco_bridge/mujoco_ros2_control/include/mujoco_ros2_control/mujoco_ros2_control.hpp",
    "simulation/mujoco_bridge/mujoco_ros2_control/include/mujoco_ros2_control/mujoco_ros2_control_plugin.hpp",
    "simulation/mujoco_bridge/mujoco_ros2_control/package.xml",
    "simulation/mujoco_bridge/mujoco_sim_ros2/src/simulate.h",
    "simulation/mujoco_bridge/mujoco_sim_ros2/src/simulate.cc",
    "simulation/mujoco_bridge/mujoco_sim_ros2/src/main.cc",
    "urdfs/d1_description/CMakeLists.txt",
    "urdfs/d1_description/package.xml",
)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def relative_source_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (path.is_absolute() or not value or ".." in path.parts or
            "\\" in value or str(path) != value):
        raise ValueError(f"Unsafe original source path: {value!r}")
    return path


def decode_text(data: bytes, label: str) -> list[str]:
    if b"\x00" in data:
        raise ValueError(f"Binary/NUL data refused for unified patch: {label}")
    try:
        lines = data.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError as error:
        raise ValueError(f"Non-UTF-8 data refused for unified patch: {label}") from error
    return lines


def prepare(archive: Path) -> tuple[bytes, dict]:
    receipt = json.loads((ROOT / "source_archive_receipt.json").read_text(encoding="utf-8"))
    if Path(receipt["destination"]).parts[-2:] != ("src", "ddt_ros2_control"):
        raise ValueError("Unexpected extracted source destination in receipt")
    archive_sha = sha_file(archive)
    if archive_sha != receipt["archive_sha256"]:
        raise ValueError("Original ZIP SHA256 differs from source_archive_receipt.json")
    changed = []
    diffs = []
    with zipfile.ZipFile(archive) as source_zip:
        for item in receipt["files"]:
            relative = str(relative_source_path(item["path"]))
            original_name = ZIP_PREFIX + relative
            try:
                original = source_zip.read(original_name)
            except KeyError as error:
                raise ValueError(f"Original file missing from ZIP: {original_name}") from error
            if sha_bytes(original) != item["sha256"]:
                raise ValueError(f"ZIP entry differs from original receipt: {relative}")
            current_path = ROOT / SOURCE_PREFIX / relative
            if not current_path.resolve().is_relative_to((ROOT / SOURCE_PREFIX).resolve()):
                raise ValueError(f"Extracted path escapes source tree: {relative}")
            if not stat.S_ISREG(current_path.lstat().st_mode):
                raise ValueError(f"Extracted file is missing/symlinked/nonregular: {relative}")
            current = current_path.read_bytes()
            current_sha = sha_bytes(current)
            if current_sha == item["sha256"]:
                continue
            if max(len(original), len(current)) > 1024 * 1024:
                raise ValueError(f"Changed file exceeds text patch size limit: {relative}")
            before = decode_text(original, relative + " (ZIP)")
            after = decode_text(current, relative + " (working tree)")
            label = SOURCE_PREFIX + relative
            diff_lines = difflib.unified_diff(
                before, after, fromfile="a/" + label, tofile="b/" + label,
                lineterm="\n")
            # difflib preserves a source's missing final newline but does not
            # emit the GNU patch marker. Add it explicitly for a valid patch.
            diff = "".join(line if line.endswith("\n") else
                           line + "\n\\ No newline at end of file\n"
                           for line in diff_lines)
            if not diff:
                raise ValueError(f"Changed file produced empty text diff: {relative}")
            diffs.append(f"diff --git a/{label} b/{label}\n" + diff)
            changed.append({"path": label, "original_sha256": item["sha256"],
                            "current_sha256": current_sha,
                            "original_bytes": len(original), "current_bytes": len(current)})
    actual = tuple(item["path"].removeprefix(SOURCE_PREFIX) for item in changed)
    if set(actual) != set(EXPECTED_CHANGED) or len(actual) != len(EXPECTED_CHANGED):
        raise ValueError(f"Original source change set differs from reviewed 12 paths: {actual}")
    patch = "".join(diffs).encode("utf-8")
    summary = {"format": "d1-upstream-text-reproduction-patch-03-v1",
               "owner": "original workspace2.zip owner; local changes are a separate derivative patch",
               "original_archive_name": archive.name,
               "original_archive_sha256": archive_sha,
               "source_archive_receipt_sha256": sha_file(ROOT / "source_archive_receipt.json"),
               "original_files_compared": len(receipt["files"]),
               "changed_text_files": changed,
               "changed_text_file_count": len(changed),
               "binary_patches": 0,
               "patch_bytes": len(patch), "patch_sha256": sha_bytes(patch),
               "third_party_tree_modified_by_this_tool": False}
    return patch, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=Path("/home/lyh/下载/workspace2.zip"),
                        help="owner's original ZIP; SHA must match receipt")
    parser.add_argument("--output", type=Path,
                        help="new .patch path; omit for read-only comparison")
    args = parser.parse_args()
    patch, summary = prepare(args.archive.resolve())
    if args.output:
        output = args.output.resolve()
        if output.suffix != ".patch":
            parser.error("--output must end in .patch")
        with output.open("xb") as stream:
            stream.write(patch)
        summary["output"] = str(output)
        summary["status"] = "PATCH_WRITTEN_NOT_APPLIED_OR_PUBLISHED"
    else:
        summary["status"] = "READ_ONLY_PATCH_PLAN"
    print(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
