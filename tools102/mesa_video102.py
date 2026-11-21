"""Render a fresh movie from the sealed replay using the Mesa software EGL device."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from tools91.launch91 import PRESSURE, THREADS, require
from tools92.launch92 import scope_for
from tools94.common94 import ROOT, digest, read, write
from tools95.run95 import files_for_seal
from tools102.common102 import BASE, PLAN_SHA, protocol, sources
from tools102.launch102 import completed, folder


UNIT = "d1-occlusion102-video-02"
OUTPUT = BASE / "video_02"
GUARD = BASE / "guard_video_02"
VENDOR = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
GL_ENV = dict(__EGL_VENDOR_LIBRARY_FILENAMES=str(VENDOR), LIBGL_ALWAYS_SOFTWARE="1",
              GALLIUM_DRIVER="llvmpipe", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless")


def video_sources():
    name = "tools102/mesa_video102.py"
    return {**sources("video"), name: digest(ROOT / name)}


def first_attempt():
    files = {"guard": BASE / "guard_video_01/report.json", "partial": folder("video") / "partial.json",
             "movie_receipt": folder("video") / "video.json"}
    receipt = read(files["movie_receipt"])
    require(read(files["guard"])["status"] == "INCOMPLETE" and read(files["partial"])["status"] == "PARTIAL"
            and receipt["status"] == "PARTIAL" and receipt["frames"] == 0
            and "expected a CPU rasterizer" in receipt["error"], "unexpected first movie attempt")
    return {k: digest(path) for k, path in files.items()}


def probe_cpu_egl(output):
    """Select the advertised software device before MuJoCo initializes the process EGL display."""
    import ctypes
    from mujoco import GLContext
    from mujoco.egl import egl_ext as egl
    from OpenGL import GL

    require(all(os.environ.get(key) == value for key, value in GL_ENV.items())
            and read(VENDOR)["ICD"]["library_path"] == "libEGL_mesa.so.0", "Mesa vendor not pinned")
    query = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)(egl.eglGetProcAddress("eglQueryDeviceStringEXT"))
    indices = [i for i, device in enumerate(egl.eglQueryDevicesEXT())
               if b"EGL_MESA_device_software" in (query(device, int(egl.EGL_EXTENSIONS)) or b"").split()]
    require(indices, "Mesa reported no EGL software device")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(indices[0])
    context = GLContext(16, 16)
    try:
        context.make_current()
        driver = (GL.glGetString(GL.GL_RENDERER) or b"").decode()
        vendor = (GL.glGetString(GL.GL_VENDOR) or b"").decode()
        require("llvmpipe" in driver.lower(), f"expected Mesa llvmpipe, got {driver!r}")
    finally:
        context.free()
    receipt = dict(renderer=driver, vendor=vendor, egl_device_index=indices[0],
                   vendor_library_file=str(VENDOR), vendor_file_sha256=digest(VENDOR), frames=0, physics_substeps=0)
    write(output / "driver.json", receipt)
    return receipt


def resource_summary(limits):
    receipt = read(GUARD / "report.json")
    scope = scope_for(UNIT)
    require(receipt["status"] == "COMPLETE" and receipt["returncode"] == 0 and receipt["child_started"]
            and receipt["owned_cgroup_children_empty"] and receipt["scope"] == scope
            and len(receipt["cpu_affinity"]) == limits["cpu_affinity_count"]
            and receipt["supervisor_sha256"] == limits["supervisor_sha256"], "incomplete movie supervisor")
    samples = [json.loads(line) for line in (GUARD / "resources.jsonl").read_text().splitlines()]
    startup = [s["effective_available_bytes"] for s in samples if s["stage"] == "startup"]
    runtime = [s["effective_available_bytes"] for s in samples if s["stage"] == "runtime"]
    require(startup and runtime and startup[-1] >= limits["admission_min_bytes"]
            and min(runtime) >= limits["runtime_min_bytes"], "movie memory reserve")
    for sample in samples:
        owned = sample["owned"]
        require(owned["path"] == scope and owned["high_bytes"] == limits["memory_high_bytes"]
                and owned["max_bytes"] == limits["memory_max_bytes"] and owned["swap_max_bytes"] == 0
                and owned["swap_current_bytes"] == 0, "incorrect render-unit cgroup")
    return dict(scope=scope, receipt_sha256=digest(GUARD / "report.json"), log_sha256=digest(GUARD / "resources.jsonl"),
                startup_bytes=startup[-1], minimum_running_bytes=min(runtime), cpu_affinity=receipt["cpu_affinity"],
                peak_group_bytes=receipt["peak_group_bytes"], elapsed_s=receipt["elapsed_s"],
                final_memory_events=receipt["final_memory_events"])


def completed_video(plan):
    replay, _ = completed("replay", plan)
    seal = read(OUTPUT / "seal.json")
    driver, movie = read(OUTPUT / "driver.json"), read(OUTPUT / "video.json")
    require(seal["status"] == "COMPLETE" and seal["protocol_sha256"] == PLAN_SHA and seal["stage"] == "video"
            and seal["unit"] == UNIT and seal["sources"] == video_sources() and seal["files"] == files_for_seal(OUTPUT)
            and seal["replay_seal_sha256"] == digest(replay / "seal.json") and seal["first_attempt"] == first_attempt()
            and movie["status"] == "COMPLETE" and movie["renderer"] == driver["renderer"] == seal["renderer"]
            and "llvmpipe" in driver["renderer"].lower() and driver["vendor_library_file"] == str(VENDOR),
            "fresh movie or source verification failed")
    return OUTPUT, resource_summary(plan["resources"])


def worker(name):
    from tools27.resource_guard27 import read_memory

    plan = protocol()
    limits, scope = plan["resources"], scope_for(UNIT)
    require(name == UNIT and Path("/proc/self/cgroup").read_text().strip() == "0::" + scope
            and os.getpriority(os.PRIO_PROCESS, 0) == limits["nice"] and len(os.sched_getaffinity(0)) == 1
            and all(os.environ.get(k) == "1" for k in THREADS) and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and read_memory(scope)["effective_available_bytes"] >= limits["admission_min_bytes"],
            "Mesa render worker requires the admitted single-core unit")
    replay, _ = completed("replay", plan)
    start = dict(stage="video", unit=UNIT, protocol_sha256=PLAN_SHA, sources=video_sources(),
                 replay_seal_sha256=digest(replay / "seal.json"), first_attempt=first_attempt())
    OUTPUT.mkdir()
    write(OUTPUT / "start.json", start)
    try:
        from tools102.mask102 import ROBOT_PATH, verify_nominal_robot
        from tools102.video102 import render
        driver = probe_cpu_egl(OUTPUT)
        verify_nominal_robot(ROOT / ROBOT_PATH)
        movie = render(replay, OUTPUT, plan)
        require(movie["renderer"] == driver["renderer"] and start["sources"] == video_sources()
                and start["replay_seal_sha256"] == digest(replay / "seal.json")
                and start["first_attempt"] == first_attempt(), "renderer or source changed during playback")
        write(OUTPUT / "seal.json", dict(status="COMPLETE", renderer=movie["renderer"], files=files_for_seal(OUTPUT), **start))
    except BaseException as error:
        write(OUTPUT / "partial.json", dict(status="PARTIAL", error=repr(error), **start))
        raise


def launch():
    plan = protocol()
    limits = plan["resources"]
    require(digest(PRESSURE) == limits["supervisor_sha256"] and VENDOR.is_file(), "missing supervisor or Mesa ICD")
    with (ROOT / "evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if GUARD.exists() or OUTPUT.exists():
            raise FileExistsError("preserving the existing second movie attempt")
        first_attempt()
        completed("replay", plan)
        env = dict.fromkeys(THREADS, "1")
        env.update(GL_ENV, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
                   PYTHONPATH=os.pathsep.join(map(str, (ROOT / ".python-deps", ROOT / "training", ROOT))),
                   MPLBACKEND="Agg")
        cmd = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", f"--unit={UNIT}",
               f"--property=Nice={limits['nice']}", f"--property=MemoryHigh={limits['memory_high_bytes']}",
               f"--property=MemoryMax={limits['memory_max_bytes']}",
               f"--property=MemorySwapMax={limits['swap_max_bytes']}", "--property=CPUWeight=25",
               "--property=OOMPolicy=kill", f"--working-directory={ROOT}"]
        cmd.extend(f"--setenv={k}={v}" for k, v in env.items())
        cmd.extend([sys.executable, "-B", str(PRESSURE), "--scope", scope_for(UNIT), "--output", str(GUARD),
                    "--timeout", "1200", "--", sys.executable, "-B", "-m", "tools102.mesa_video102", "--worker-unit", UNIT])
        subprocess.run(cmd, check=True)
        completed_video(plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-unit")
    args = parser.parse_args()
    if args.worker_unit:
        worker(args.worker_unit)
    else:
        launch()
