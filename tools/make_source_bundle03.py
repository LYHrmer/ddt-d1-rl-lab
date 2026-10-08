#!/usr/bin/env python3
"""Plan or create a small, explicit-allowlist source bundle for D1 contract 03.

No Git repository is initialized, no upstream ZIP is copied, and no patch is
generated. With no --output this command is read-only. --audit-upstream lists
changed extracted files from source_archive_receipt.json for later review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 20 * 1024 * 1024

# Every archive payload path is named here. No directory traversal or globbing
# can silently add build outputs, policy weights, meshes or evidence traces.
ALLOWLIST = {
    "provenance": (
        ".gitignore",
        "README_LOCAL.md",
        "source_archive_receipt.json",
    ),
    "entrypoints": (
        "ros_env.sh",
        "run_d1_keyboard.sh",
        "run_d1_mujoco.sh",
        "run_d1_training_course.sh",
        "run_training03.sh",
    ),
    "training": (
        "training/d1_rl/__init__.py",
        "training/d1_rl/env.py",
        "training/d1_rl/interface.py",
        "training/d1_rl/forkserver_bootstrap.py",
        "training/d1_rl/fast_checks.py",
        "training/configs/pilot03.json",
        "training/configs/pilot03b.json",
        "training/configs/interface03b.json",
        "training/configs/diagnostic03b_zero.json",
        "training/configs/diagnostic03b_static_support.json",
        "training/configs/diagnostic03b_wheel_pulse.json",
        "training/DEPENDENCIES03.md",
        "evidence/interface_audit_03/golden_run_01/schema.json",
    ),
    "runtime03_source": (
        "runtime03/CMakeLists.txt",
        "runtime03/package.xml",
        "runtime03/README.md",
        "runtime03/uncrustify.cfg",
        "runtime03/include/runtime03/interface.hpp",
        "runtime03/src/main.cpp",
        "runtime03/schema_pilot03.json",
        "runtime03/schema_pilot03b.json",
        "runtime03/compare_python_interface.py",
        "runtime03/compare_dynamic_traces.py",
        "runtime03/check_cli_zero_physics.py",
        "runtime03/prepare_pilot_deployment03.py",
        "runtime03/pilot_deployment_plan03b.md",
        "runtime03/live_watchdog03.py",
        "runtime03/evaluate_watchdog03.py",
        "runtime03/sequences03b/forward_stop.json",
        "runtime03/sequences03b/reverse_stop.json",
        "runtime03/sequences03b/turn_stop.json",
    ),
    "tools": (
        "tools/audit_d1_interface.py",
        "tools/interface_harness03.cpp",
        "tools/generate_video_course.py",
        "tools/generate_training_course.py",
        "tools/validate_training_course.py",
        "tools/render_training_course.py",
        "tools/plot_course_map.py",
        "tools/build_runtime03.sh",
        "tools/install_system_deps.sh",
        "tools/bounded_training_run.py",
        "tools/train_d1_ppo.py",
        "tools/export_d1_ppo.py",
        "tools/evaluate_d1_ppo.py",
        "tools/run_d1_onnx_probe.py",
        "tools/run_selftrained03.py",
        "tools/render_runtime03_trace.py",
        "tools/run_d1_fixed_probe.py",
        "tools/evaluate_runtime03_trace.py",
        "tools/summarize_training03.py",
        "tools/profile_env03.py",
        "tools/validate_fast_checks03.py",
        "tools/benchmark_fast_checks03.py",
        "tools/check_fast_checks03_static.py",
        "tools/generate_upstream_patch03.py",
        "tools/upstream_reproduction03.patch",
        "tools/upstream_reproduction03.md",
        "tools/make_source_bundle03.py",
    ),
    "contracts_and_tutorial": (
        "docs/career_execution_contract_03.md",
        "docs/career_execution_contract_03b_addendum.md",
        "docs/d1_interface_audit_03.md",
        "docs/tutorial/00_复现教程.md",
        "docs/tutorial/01_地形与操作.md",
        "docs/tutorial/02_观测动作与时基.md",
        "docs/tutorial/03_训练导出与评测.md",
    ),
    "generated_terrain_source": (
        "tools/course03_layout.json",
        "src/ddt_ros2_control/urdfs/d1_description/mujoco/scene_contract_plane_02.xml",
        "src/ddt_ros2_control/urdfs/d1_description/mujoco/scene_contract_step15_02.xml",
        "src/ddt_ros2_control/urdfs/d1_description/mujoco/scene_training_course_03.xml",
        "src/ddt_ros2_control/urdfs/d1_description/mujoco/scene_video_reconstruction.xml",
        "src/ddt_ros2_control/urdfs/d1_description/mujoco/training_course_03_assets/layout_manifest.json",
        "src/ddt_ros2_control/urdfs/d1_description/mujoco/training_course_03_assets/rough_10mm.png",
        "src/ddt_ros2_control/urdfs/d1_description/mujoco/training_course_03_assets/rough_25mm.png",
        "src/ddt_ros2_control/urdfs/d1_description/mujoco/training_course_03_assets/rough_50mm.png",
        "src/ddt_ros2_control/urdfs/d1_description/mujoco/video_course_assets/layout_estimated.json",
        "src/ddt_ros2_control/urdfs/d1_description/mujoco/video_course_assets/pattern_height.png",
        "src/ddt_ros2_control/urdfs/d1_description/mujoco/video_course_assets/rough_height.png",
        "src/ddt_ros2_control/urdfs/d1_description/mujoco/video_course_assets/README.md",
        "src/ddt_ros2_control/urdfs/d1_description/mujoco/training_course_03_assets/README.md",
        "src/ddt_ros2_control/simulation/mujoco_bridge/mujoco_ros2_control/include/mujoco_ros2_control/step_evidence.hpp",
        "src/ddt_ros2_control/simulation/mujoco_bridge/mujoco_ros2_control/src/step_evidence.cpp",
    ),
}

DEPENDENCIES = {
    "external_source_archive": "workspace2.zip; obtain separately from its owner",
    "external_source_archive_sha256_source": "source_archive_receipt.json",
    "python": "3.10",
    "python_packages": {"numpy": "2.2.6", "mujoco": "3.3.0",
                        "gymnasium": "1.2.0", "stable-baselines3": "2.7.0",
                        "torch": "2.12.1", "onnx": "1.18.0",
                        "onnxruntime": "1.23.2", "psutil": "5.9.0"},
    "native": {"ROS": "Humble", "MuJoCo": "3.3.0",
               "C++ ONNX Runtime": "1.10.0", "ffmpeg": "required for trace MP4"},
}

EXCLUSIONS = [
    "original robot.xml, meshes/STL, and colleague ONNX policies",
    "workspace2.zip and third-party source/dependency archives",
    "runtime03/build, install, .python-deps, bytecode and generated binaries",
    "training checkpoints and all evidence outputs except the single audited golden schema.json",
]


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def safe_member(root: Path, member: str) -> Path:
    pure = PurePosixPath(member)
    if (pure.is_absolute() or ".." in pure.parts or "." in pure.parts or
            "\\" in member or not member or str(pure) != member):
        raise ValueError(f"Unsafe archive member path: {member!r}")
    candidate = root.joinpath(*pure.parts)
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Archive member escapes workspace: {member}")
    if not stat.S_ISREG(candidate.lstat().st_mode):
        raise ValueError(f"Archive member is missing, symlinked or not a regular file: {member}")
    return candidate


def inventory(root: Path) -> list[dict]:
    seen = set()
    rows = []
    total = 0
    for category, members in ALLOWLIST.items():
        for member in members:
            if member in seen:
                raise ValueError(f"Duplicate allowlist path: {member}")
            seen.add(member)
            path = safe_member(root, member)
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                raise ValueError(f"Allowlisted file exceeds 2 MiB: {member}")
            total += size
            if total > MAX_TOTAL_BYTES:
                raise ValueError("Source bundle exceeds 20 MiB uncompressed")
            rows.append({"path": member, "category": category,
                         "bytes": size, "sha256": digest_file(path)})
    return sorted(rows, key=lambda item: item["path"])


def upstream_identity(root: Path) -> dict:
    receipt = json.loads((root / "source_archive_receipt.json").read_text(encoding="utf-8"))
    sha = receipt["archive_sha256"]
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("Invalid original archive identity in source_archive_receipt.json")
    return {"archive_name": Path(receipt["archive"]).name,
            "archive_sha256_from_receipt": sha,
            "receipt_sha256": digest_file(root / "source_archive_receipt.json"),
            "archive_bytes_included": 0,
            "original_archive_rehashed_by_this_tool": False}


def audit_upstream(root: Path) -> dict:
    """List extracted original files that differ from the immutable ZIP receipt.

    This is a read-only candidate list; no patch or original bytes are copied.
    """
    receipt = json.loads((root / "source_archive_receipt.json").read_text(encoding="utf-8"))
    recorded_destination = Path(receipt["destination"])
    if recorded_destination.parts[-2:] != ("src", "ddt_ros2_control"):
        raise ValueError("Unexpected original ZIP extraction destination")
    # The receipt remembers the original machine's absolute path. Rebase its
    # known extraction subdirectory so the read-only audit works in a copy.
    destination = root / "src/ddt_ros2_control"
    changed, missing = [], []
    for item in receipt["files"]:
        relative = item["path"]
        path = safe_receipt_path(destination, relative)
        if not path.exists():
            missing.append(relative)
            continue
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(f"Original extracted file is no longer regular: {relative}")
        current_sha = digest_file(path)
        if current_sha != item["sha256"]:
            changed.append({"path": relative, "zip_receipt_sha256": item["sha256"],
                            "current_sha256": current_sha,
                            "possible_text_patch": path.suffix.lower() in
                            {".cpp", ".cc", ".hpp", ".h", ".c", ".py", ".xml", ".yaml", ".yml",
                             ".txt", ".md", ".sh", ".bash", ".cmake"}})
    return {"read_only": True, "patch_generated": False,
            "changed_original_files": changed, "missing_original_files": missing,
            "files_compared": len(receipt["files"])}


def safe_receipt_path(destination: Path, member: str) -> Path:
    pure = PurePosixPath(member)
    if pure.is_absolute() or ".." in pure.parts or "\\" in member or str(pure) != member:
        raise ValueError(f"Unsafe original receipt path: {member!r}")
    path = destination.joinpath(*pure.parts)
    if not path.resolve().is_relative_to(destination):
        raise ValueError(f"Original receipt path escapes extraction: {member}")
    return path


def bundle_readme(upstream: dict) -> str:
    return ("# D1 contract 03 source bundle (not standalone)\n\n"
            "This ZIP contains explicitly listed new source, contracts, tutorials and small generated terrain assets. "
            "It intentionally contains no colleague robot model, meshes, ONNX weights, original project source, "
            "training checkpoints or large evidence. It cannot run by itself.\n\n"
            f"Obtain the original {upstream['archive_name']} separately from its owner and verify SHA256 "
            f"`{upstream['archive_sha256_from_receipt']}` against `source_archive_receipt.json`. "
            "Follow `README_LOCAL.md` for the original workspace setup, then the included `docs/tutorial/` "
            "and contract 03b for the new training/runtime path. "
            "`tools/upstream_reproduction03.patch` is a separately identified text delta for 12 original files; "
            "read `tools/upstream_reproduction03.md` before applying it to a clean extraction. "
            "The original source files themselves are not included.\n\n"
            "`manifest.json` records every included file's byte size and SHA256. It does not certify that a "
            "policy trained successfully or traversed terrain.\n")


def payload(root: Path, rows: list[dict], upstream: dict) -> tuple[bytes, bytes]:
    readme_bytes = bundle_readme(upstream).encode("utf-8")
    manifest = {"format": "d1-contract03-explicit-source-bundle-v1",
                "files": rows, "file_count": len(rows),
                "source_bytes": sum(row["bytes"] for row in rows),
                "bundle_readme_sha256": hashlib.sha256(readme_bytes).hexdigest(),
                "upstream": upstream, "dependencies": DEPENDENCIES,
                "excluded": EXCLUSIONS,
                "source_archive_is_required_separately": True,
                "upstream_patch_included": True}
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2,
                                 allow_nan=False) + "\n").encode("utf-8")
    return manifest_bytes, readme_bytes


def write_bundle(root: Path, output: Path, rows: list[dict], upstream: dict) -> dict:
    if output.suffix.lower() != ".zip":
        raise ValueError("--output must end in .zip")
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("Refusing to overwrite archive or receipt")
    if not output.parent.is_dir():
        raise FileNotFoundError("Archive parent directory must already exist")
    manifest_bytes, readme_bytes = payload(root, rows, upstream)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9, allowZip64=False) as bundle:
        for name, data in (("BUNDLE_README.md", readme_bytes),
                           ("manifest.json", manifest_bytes)):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            bundle.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED,
                            compresslevel=9)
        for row in rows:
            name = row["path"]
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            # Source files are capped at 2 MiB. Recheck after read to catch a
            # concurrent edit during bundling instead of making a stale manifest.
            data = safe_member(root, name).read_bytes()
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise RuntimeError(f"Source changed while bundling: {name}")
            bundle.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED,
                            compresslevel=9)
    receipt = {"status": "ARCHIVED", "archive": str(output),
               "archive_bytes": output.stat().st_size,
               "archive_sha256": digest_file(output),
               "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "bundle_readme_sha256": hashlib.sha256(readme_bytes).hexdigest(),
               "source_file_count": len(rows),
               "source_bytes": sum(row["bytes"] for row in rows),
               "original_archive_included": False,
               "original_archive_sha256_from_receipt": upstream["archive_sha256_from_receipt"],
               "upstream_patch_included": True}
    with receipt_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="new .zip path; omit for read-only dry-run")
    parser.add_argument("--audit-upstream", action="store_true",
                        help="read-only SHA comparison of original extracted files against receipt")
    args = parser.parse_args()
    rows = inventory(ROOT)
    upstream = upstream_identity(ROOT)
    manifest_bytes, readme_bytes = payload(ROOT, rows, upstream)
    summary = {"mode": "archive" if args.output else "dry_run",
               "format": "d1-contract03-explicit-source-bundle-v1",
               "file_count": len(rows),
               "source_bytes": sum(row["bytes"] for row in rows),
               "max_file_bytes": max(row["bytes"] for row in rows),
               "planned_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "planned_bundle_readme_sha256": hashlib.sha256(readme_bytes).hexdigest(),
               "files": rows,
               "upstream": upstream,
               "excluded": EXCLUSIONS}
    if args.audit_upstream:
        summary["upstream_audit"] = audit_upstream(ROOT)
    if args.output:
        summary["archive_receipt"] = write_bundle(ROOT, args.output.resolve(), rows, upstream)
    print(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
