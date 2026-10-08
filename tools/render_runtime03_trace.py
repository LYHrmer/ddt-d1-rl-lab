#!/usr/bin/env python3
"""Render an existing C++ runtime03 trace as a labeled, zero-integration MP4.

This is visual playback of recorded qpos/qvel, not a new policy or physics test.
Use --check-only to validate the trace without importing MuJoCo or a renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (ROOT / "src/ddt_ros2_control/urdfs/d1_description/mujoco/"
                 "scene_contract_plane_02.xml")
NQ, NV = 23, 22


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def finite_vector(value, length: int, label: str, line_number: int) -> None:
    if (not isinstance(value, list) or len(value) != length or
            any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in value)):
        raise ValueError(f"trace line {line_number}: {label} must have {length} finite numbers")


def inspect_source(source_run: Path, model_path: Path) -> dict:
    """Read-only validation; imports only the Python standard library."""
    report_path = source_run / "report.json"
    trace_path = source_run / "trace.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    status = report.get("status")
    if status not in ("COMPLETED", "FAILED"):
        raise ValueError(f"C++ report has no final outcome: {status!r}")
    model_hash = sha256(model_path)
    if report.get("model_sha256") != model_hash:
        raise ValueError("Model XML SHA256 differs from the C++ report")
    robot_path = model_path.parent / "robot.xml"
    robot_hash = sha256(robot_path)
    if report.get("robot_xml_sha256") != robot_hash:
        raise ValueError("Included robot XML SHA256 differs from the C++ report")
    dt = float(report["native_dt"])
    if not math.isfinite(dt) or not 0 < dt <= 0.01:
        raise ValueError("Invalid native_dt in C++ report")
    completed = report.get("native_steps_completed")
    if not isinstance(completed, int) or completed < 1:
        raise ValueError("C++ report has no completed native steps to replay")
    previous_time = -math.inf
    count = 0
    last_time = None
    with trace_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank trace line {line_number}")
            row = json.loads(line)
            count += 1
            if row.get("native_count") != count:
                raise ValueError(f"trace line {line_number}: native_count is not contiguous")
            finite_vector(row.get("qpos"), NQ, "qpos", line_number)
            finite_vector(row.get("qvel"), NV, "qvel", line_number)
            finite_vector(row.get("command"), 3, "command", line_number)
            for field in ("sim_time", "body_vx", "body_wz"):
                value = row.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"trace line {line_number}: invalid {field}")
            time_s = float(row["sim_time"])
            if time_s <= previous_time or abs(time_s - count * dt) > max(1e-8, time_s * 1e-7):
                raise ValueError(f"trace line {line_number}: sim_time/native_dt mismatch")
            previous_time = time_s
            last_time = time_s
    if count != completed:
        raise ValueError(f"C++ trace has {count} rows; report says {completed} completed steps")
    return {
        "source_status": status,
        "source_terminal_reason": report.get("terminal_reason") or report.get("error") or "UNSPECIFIED",
        "source_native_steps_completed": completed,
        "source_native_steps_attempted": report.get("native_steps_attempted"),
        "source_mode": report.get("mode"),
        "source_model_sha256": model_hash,
        "source_robot_sha256": robot_hash,
        "source_report_sha256": sha256(report_path),
        "source_trace_sha256": sha256(trace_path),
        "source_schema_sha256": report.get("schema_sha256"),
        "source_policy_sha256": report.get("policy_sha256"),
        "native_dt_s": dt,
        "last_recorded_sim_time_s": last_time,
        "trace_rows": count,
    }


def frame_indices(completed: int, dt: float, fps: int) -> list[int]:
    """Choose nearest completed native state for each video timestamp."""
    duration = completed * dt
    indices = []
    for frame in range(math.floor(duration * fps + 1e-10) + 1):
        native = min(completed, max(1, math.floor(frame / fps / dt + 0.5)))
        if not indices or native != indices[-1]:
            indices.append(native)
    if indices[-1] != completed:
        indices.append(completed)
    return indices


def selected_rows(trace_path: Path, indices: list[int]):
    """Stream rows; keep only the selected video frames in memory."""
    wanted = iter(indices)
    target = next(wanted, None)
    with trace_path.open("r", encoding="utf-8") as stream:
        for native_count, line in enumerate(stream, 1):
            if target is None:
                break
            if native_count == target:
                yield json.loads(line)
                target = next(wanted, None)
    if target is not None:
        raise ValueError("Trace ended before a selected video frame")


def render(source_run: Path, model_path: Path, output_dir: Path, source: dict,
           width: int, height: int, fps: int, camera_mode: str,
           distance: float, azimuth: float, elevation: float,
           end_hold_s: float) -> dict:
    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("Set MUJOCO_GL=egl before rendering")
    import mujoco  # Delayed: --help and --check-only load no MuJoCo/GL.
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    if mujoco.__version__ != "3.3.0":
        raise RuntimeError(f"MuJoCo 3.3.0 required; found {mujoco.__version__}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if (model.nq, model.nv) != (NQ, NV):
        raise ValueError(f"Model dimensions changed: nq={model.nq}, nv={model.nv}")
    # Offscreen buffer capacity only; geometry, dynamics and source XML stay untouched.
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    indices = frame_indices(source["trace_rows"], source["native_dt_s"], fps)
    hold_frames = round(end_hold_s * fps)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    font_file = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_file, 24) if Path(font_file).is_file() else ImageFont.load_default()
    small = ImageFont.truetype(font_file, 18) if Path(font_file).is_file() else ImageFont.load_default()
    mp4 = output_dir / "runtime_trace.mp4"
    command = ["rtk", "proxy", "ffmpeg", "-nostdin", "-loglevel", "error", "-n",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(mp4)]
    encoded = 0
    forwards = 0
    process = None
    last_frame = None
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        assert process.stdin is not None
        first_position = None
        for row in selected_rows(source_run / "trace.jsonl", indices):
            qpos = np.asarray(row["qpos"], dtype=np.float64)
            if first_position is None:
                first_position = qpos[:3].copy()
            data.qpos[:] = qpos
            data.qvel[:] = row["qvel"]
            data.time = float(row["sim_time"])
            data.ctrl[:] = 0
            data.qfrc_applied[:] = 0
            mujoco.mj_forward(model, data)
            forwards += 1
            position = qpos[:3] if camera_mode == "follow" else first_position
            camera.lookat[:] = (float(position[0]), float(position[1]),
                                float(position[2]) - 0.10)
            renderer.update_scene(data, camera=camera)
            canvas = Image.fromarray(renderer.render(), mode="RGB")
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, width, 116), fill=(12, 20, 29))
            draw.text((24, 12), "D1  |  RECORDED C++ RUNTIME TRACE", font=font,
                      fill=(238, 244, 248))
            command_vx, _, command_wz = row["command"]
            values = (f"sim {row['sim_time']:.3f}s   vx {row['body_vx']:+.3f} m/s   "
                      f"wz {row['body_wz']:+.3f} rad/s   "
                      f"cmd vx {command_vx:+.2f}, wz {command_wz:+.2f}")
            draw.text((24, 48), values, font=small, fill=(214, 229, 237))
            qualification = source.get("qualification_status", "NOT_SCORED")
            status = (f"RUN {source['source_status']}  |  QUALIFICATION {qualification}  |  "
                      f"{source['source_terminal_reason']}")
            status_color = ((255, 168, 155) if source["source_status"] == "FAILED" or
                            qualification == "FAIL" else (184, 220, 244))
            draw.text((24, 80), status[:110], font=small, fill=status_color)
            draw.rectangle((0, height - 42, width, height), fill=(12, 20, 29))
            draw.text((24, height - 33), "Recorded states + mj_forward only  |  No new physics validation",
                      font=small, fill=(186, 205, 216))
            last_frame = canvas.tobytes()
            process.stdin.write(last_frame)
            encoded += 1
        if last_frame is None:
            raise ValueError("No frame selected from trace")
        for _ in range(hold_frames):
            process.stdin.write(last_frame)
            encoded += 1
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg exited {code}: {stderr[-2000:]}")
    finally:
        renderer.close()
        if process is not None and process.poll() is None:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            process.kill()
            process.wait()
    return {
        "status": "RENDERED_SOURCE_FAILED" if source["source_status"] == "FAILED" else "RENDERED_SOURCE_COMPLETED",
        "source_outcome": source,
        "video": mp4.name,
        "video_sha256": sha256(mp4),
        "renderer_script_sha256": sha256(Path(__file__)),
        "ffmpeg_command": command,
        "width": width, "height": height, "fps": fps,
        "camera_mode": camera_mode, "distance": distance,
        "azimuth": azimuth, "elevation": elevation,
        "source_frames": len(indices), "end_hold_frames": hold_frames,
        "frame_native_indices": indices,
        "encoded_frames": encoded,
        "mj_forward_calls": forwards,
        "mujoco_version": mujoco.__version__,
        "new_physics_steps": 0,
        "meaning": "Recorded C++ states rendered offline; no policy inference or new dynamics validation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path,
                        help="C++ run directory containing report.json and trace.jsonl")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path,
                        help="new directory for runtime_trace.mp4 and render_report.json")
    parser.add_argument("--check-only", action="store_true",
                        help="validate report/trace/model hashes without GL, ffmpeg or output files")
    parser.add_argument("--evaluation-report", type=Path,
                        help="optional offline score receipt; verify its source identities and label qualification")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera", choices=("fixed", "follow"), default="fixed")
    parser.add_argument("--distance", type=float, default=2.7)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-22.0)
    parser.add_argument("--end-hold-s", type=float, default=1.0)
    args = parser.parse_args()
    if not 1 <= args.fps <= 120:
        parser.error("--fps must be in 1..120")
    if (args.width < 320 or args.height < 240 or args.width > 3840 or args.height > 2160 or
            args.width % 2 or args.height % 2):
        parser.error("--width and --height must be even and within 320x240..3840x2160")
    if (not all(math.isfinite(x) for x in (args.distance, args.azimuth, args.elevation,
                                          args.end_hold_s)) or args.distance <= 0 or
            not 0 <= args.end_hold_s <= 5):
        parser.error("Invalid camera or end-hold value")
    source_run = args.source_run.resolve()
    model_path = args.model.resolve()
    source = inspect_source(source_run, model_path)
    if args.evaluation_report is not None:
        score_path = args.evaluation_report.resolve()
        score = json.loads(score_path.read_text(encoding="utf-8"))
        identities = score.get("input_sha256", {})
        if (score.get("status") not in ("PASS", "FAIL") or score.get("evidence_valid") is not True or
                identities.get(str(source_run / "report.json")) != source["source_report_sha256"] or
                identities.get(str(source_run / "trace.jsonl")) != source["source_trace_sha256"]):
            raise ValueError("Evaluation receipt does not validly score this exact runtime trace")
        source["qualification_status"] = score["status"]
        source["qualification_report_sha256"] = sha256(score_path)
    if args.check_only:
        print(json_text({"status": "STATIC_TRACE_VALIDATED", **source,
                         "planned_source_frames": len(frame_indices(
                             source["trace_rows"], source["native_dt_s"], args.fps)),
                         "new_physics_steps": 0}))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required when rendering")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        parser.error(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        receipt = render(source_run, model_path, output_dir, source,
                         args.width, args.height, args.fps, args.camera,
                         args.distance, args.azimuth, args.elevation, args.end_hold_s)
    except BaseException as exc:
        (output_dir / "render_report.json").write_text(json_text({
            "status": "RENDER_FAILED", "error": f"{type(exc).__name__}: {exc}",
            "source_outcome": source, "new_physics_steps": 0,
            "renderer_script_sha256": sha256(Path(__file__)),
        }), encoding="utf-8")
        raise
    (output_dir / "render_report.json").write_text(json_text(receipt), encoding="utf-8")
    print(json_text({"output": str(output_dir), "status": receipt["status"],
                     "source_status": source["source_status"],
                     "encoded_frames": receipt["encoded_frames"],
                     "new_physics_steps": 0}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
