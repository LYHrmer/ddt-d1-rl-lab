"""Render complete recorded states to UI-free PNG frames and a bounded GIF.

MuJoCo only runs mj_forward here. The recorded model, states, and original
controller are immutable inputs; rendering executes no policy or physics step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools35.record_cpp_showcase35 import (
    ResourceGuard,
    require,
    sha,
    validate_complete_states,
    write_json,
)

GIF_LIMIT_BYTES = 5_000_000


def load_recording(folder):
    """Reject partial, modified, or unrelated state captures before rendering."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    report = json.loads((folder / "report.json").read_text())
    parent = json.loads((folder.parent / "report.json").read_text())
    require(manifest["format"] == "mujoco-state-showcase-v1"
            and manifest["status"] == report["status"] == parent["status"] == "COMPLETE"
            and report["cleanup_complete"] and parent["cleanup_complete"]
            and report["error"] is None and parent["error"] is None,
            "only a complete closed capture can be rendered")
    require(manifest["states"] == 601 and manifest["duration_s"] == 12.0
            and not manifest["new_qualification_claimed"], "unexpected recording scope")
    for name, digest in manifest["artifacts_sha256"].items():
        require(Path(name).name == name and sha(folder / name) == digest,
                "recording artifact changed: " + name)
    require({"model.mjb", "states.npz"} <= manifest["artifacts_sha256"].keys(),
            "model or synchronized states missing")
    return manifest


def frame_indices(time_s):
    """Fixed nearest-state sampling at 20 fps; maximum timing offset 10 ms."""
    require(len(time_s) == 601, "full recording required")
    return [round(index * 2.5) for index in range(241)]


def encode_gif(frame_paths, destination, poll=lambda: None):
    import numpy as np
    from PIL import Image
    # One palette for the whole clip prevents frame-to-frame color flicker.
    sample_paths = frame_paths[::20]
    atlas = Image.new("RGB", (160 * len(sample_paths), 90))
    for i, path in enumerate(sample_paths):
        with Image.open(path) as frame:
            atlas.paste(frame.resize((160, 90)), (160 * i, 0))
    palette = atlas.quantize(colors=96)
    bayer = (np.asarray([[0, 8, 2, 10], [12, 4, 14, 6],
                         [3, 11, 1, 9], [15, 7, 13, 5]]) / 16 - .5) * 8
    frames = []
    try:
        for path in frame_paths:
            poll()
            with Image.open(path) as frame:
                rgb = np.asarray(frame.convert("RGB"), dtype=np.float32)
                offset = np.tile(bayer, ((frame.height + 3) // 4, (frame.width + 3) // 4))
                ordered = np.clip(rgb + offset[:frame.height, :frame.width, None], 0, 255).astype(np.uint8)
                frames.append(Image.fromarray(ordered).quantize(palette=palette, dither=Image.Dither.NONE))
        with Path(destination).open("xb") as stream:
            frames[0].save(stream, format="GIF", save_all=True, append_images=frames[1:],
                           duration=50, loop=0, optimize=True, disposal=1)
    finally:
        for frame in frames:
            frame.close()
        palette.close()
        atlas.close()
    require(destination.stat().st_size <= GIF_LIMIT_BYTES, "GIF exceeds the fixed 5 MB limit")
    with Image.open(destination) as saved:
        durations = []
        for index in range(saved.n_frames):
            saved.seek(index)
            durations.append(saved.info.get("duration", 0))
        require(sum(durations) == 50 * len(frame_paths), "GIF duration changed during encoding")
        return {"gif_frames": saved.n_frames, "duration_ms": sum(durations),
                "bytes": destination.stat().st_size, "sha256": sha(destination)}


def add_world_grid(mujoco, scene, center):
    """Render-only world reference; no contact geometry or simulated force."""
    import numpy as np
    origin = np.floor(np.asarray(center[:2]) / .5) * .5
    for offset in range(-6, 7):
        for axis in (0, 1):
            require(scene.ngeom < scene.maxgeom, "render scene geometry capacity exhausted")
            start, end = np.array([*origin, .002]), np.array([*origin, .002])
            start[axis] += offset * .5
            end[axis] += offset * .5
            start[1 - axis] -= 3
            end[1 - axis] += 3
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                               np.eye(3).reshape(-1), np.asarray([.35, .40, .44, 1.]))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1.0, start, end)
            scene.ngeom += 1


def render(recording, output, owned_scope):
    require(not output.exists(), "output exists: no overwrite or resume")
    output.mkdir(parents=True)
    result = {"status": "INCOMPLETE", "error": None, "physics_steps": 0, "policy_calls": 0,
              "forward_calls": 0, "frames_rendered": 0, "new_performance_claims": False,
              "recording": str(recording), "owned_scope": owned_scope}
    write_json(output / "report.json", result)
    started = time.monotonic()
    renderer = None
    from tools16.run_comparison16 import termination_guard
    with termination_guard() as control:
        try:
            guard = ResourceGuard(output, owned_scope)
            guard.poll(True, admission=True)
            require(os.environ.get("MUJOCO_GL") == "egl"
                    and os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1"
                    and os.environ.get("EGL_PLATFORM") == "surfaceless",
                    "software offscreen EGL environment required")
            for path in (ROOT / ".python-deps", ROOT / "training"):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            import numpy as np
            from PIL import Image

            import mujoco

            manifest = load_recording(recording)
            result["manifest_sha256"] = sha(recording / "manifest.json")
            result["renderer_sha256"] = sha(__file__)
            require(mujoco.__version__ == "3.3.0", "recording requires MuJoCo 3.3.0")
            with np.load(recording / "states.npz", allow_pickle=False) as saved:
                qpos, qvel, time_s = saved["qpos"], saved["qvel"], saved["time_s"]
            validate_complete_states(qpos, qvel, time_s, manifest["nq"], manifest["nv"])
            model = mujoco.MjModel.from_binary_path(str(recording / "model.mjb"))
            require(model.nq == manifest["nq"] and model.nv == manifest["nv"], "model shape differs")
            model.vis.global_.offwidth, model.vis.global_.offheight = 512, 288
            # These changes affect only the render copy of the archived model.
            for geom_id in range(model.ngeom):
                if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE:
                    model.geom_rgba[geom_id] = [.70, .73, .76, 1.]
                    material = model.geom_matid[geom_id]
                    if material >= 0:
                        model.mat_rgba[material] = [.70, .73, .76, 1.]
                        model.mat_reflectance[material] = 0
                        model.mat_specular[material] = 0
                        model.mat_texid[material] = -1
            data = mujoco.MjData(model)
            indices = frame_indices(time_s)
            # A fixed camera makes translation and turning visible against the ground.
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            positions = qpos[:, manifest["free_qpos"]:manifest["free_qpos"] + 3]
            low, high = positions.min(axis=0), positions.max(axis=0)
            camera.lookat[:] = ((low[0] + high[0]) / 2, (low[1] + high[1]) / 2, .25)
            camera.distance = float(max(1.8, 1.35 + 1.05 * np.linalg.norm(high[:2] - low[:2])))
            camera.azimuth, camera.elevation = 135, -20
            result["camera"] = {"lookat": camera.lookat.tolist(), "distance": camera.distance,
                                "azimuth": camera.azimuth, "elevation": camera.elevation}
            renderer = mujoco.Renderer(model, height=288, width=512)
            (output / "frames").mkdir()
            paths = []
            for number, index in enumerate(indices):
                guard.poll()
                data.qpos[:] = qpos[index]
                data.qvel[:] = qvel[index]
                data.time = time_s[index]
                mujoco.mj_forward(model, data)
                result["forward_calls"] += 1
                renderer.update_scene(data, camera=camera)
                add_world_grid(mujoco, renderer.scene, camera.lookat)
                path = output / "frames" / f"{number:04d}.png"
                Image.fromarray(renderer.render()).save(path)
                paths.append(path)
                result["frames_rendered"] += 1
            result["gif"] = encode_gif(paths, output / "showcase.gif", guard.poll)
            result["source_duration_s"] = float(time_s[-1] - time_s[0])
            result["source_frame_indices"] = indices
            result["encoding"] = {"width": 512, "height": 288, "fps": 20,
                                  "global_palette_colors": 96, "dither": "bayer4_amplitude8",
                                  "sampling": "fixed_nearest_state", "maximum_sampling_offset_s": .01}
            result["render_only_changes"] = {"floor": "light_gray_no_texture_no_reflection",
                "world_grid_spacing_m": .5, "grid_geometry": "mjvScene_only_no_contacts",
                "backend": "EGL_surfaceless_LIBGL_ALWAYS_SOFTWARE=1"}
            result["png_sequence_sha256"] = {p.name: sha(p) for p in paths}
            require(load_recording(recording) == manifest and sha(__file__) == result["renderer_sha256"],
                    "rendering inputs changed")
            guard.poll(True)
            result["status"] = "COMPLETE"
        except BaseException as exc:  # noqa: BLE001 -- Keep partial frames and a failed rendering journal.
            result["error"] = f"{type(exc).__name__}: {exc}"
            (output / "error.txt").write_text(traceback.format_exc())
        finally:
            control["defer"] = True
            if control["received"]:
                result.update(status="INCOMPLETE", error=control["received"])
            if renderer is not None:
                try:
                    renderer.close()
                except BaseException as exc:  # noqa: BLE001 -- Cleanup failures cannot be marked complete.
                    result.update(status="INCOMPLETE", error=f"renderer cleanup {type(exc).__name__}: {exc}")
            result["elapsed_s"] = time.monotonic() - started
            write_json(output / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owned-scope", required=True)
    args = parser.parse_args(argv)
    result = render(args.recording.resolve(), args.output.resolve(), args.owned_scope)
    print(json.dumps({k: v for k, v in result.items() if k not in ("png_sequence_sha256", "source_frame_indices")}, indent=2))
    return int(result["status"] != "COMPLETE")


if __name__ == "__main__":
    raise SystemExit(main())
