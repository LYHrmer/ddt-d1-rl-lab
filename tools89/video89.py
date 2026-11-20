"""Render the preregistered curriculum pair from saved real-time trajectories."""
import json
import os
import subprocess

import numpy as np

from tools87 import video87 as view
from tools87.readback87 import check
from tools89.common89 import BASE, PLAN, check_worker, complete_stage, read, sha

ARMS = ("original", "mixed")
LABELS = ("感知 PPO · 训练只从坡前起步", "感知 PPO · 训练交替从坡前和坡上起步")


def paired_sources(root, header, plan):
    seed, reset = plan["training"]["seeds"][0], plan["evaluation"]["reset_seeds"][0]
    check(header["status"] == "COMPLETE" and header["protocol_sha256"] == sha(PLAN), "evaluation identity")
    cases = {r["case"]: r for r in header["cases"]}
    check(len(header["cases"]) == len(cases), "duplicate evaluation case")
    traces, contacts = [], []
    toe = plan["curriculum"]["toe_x_m"]
    for arm in ARMS:
        name = f"{arm}_{seed}_up_{reset}"
        folder = root / name
        trace = view.read_trace(folder, arm)
        check(trace.meta == cases[name] and trace.meta["training_seed"] == seed and trace.meta["seed"] == reset
              and trace.meta["terrain"] == "up" and trace.meta["protocol_sha256"] == header["protocol_sha256"],
              name + " selected matched source")
        with np.load(folder / "trace.npz", allow_pickle=False) as z:
            progress = z["front_contact_x_m"] - toe
        check(progress.shape == trace.time.shape and not np.isinf(progress).any(), name + " loaded front trace")
        finite = progress[np.isfinite(progress)]
        check((float(finite.max()) if finite.size else None) == trace.meta["max_loaded_front_contact_past_toe_m"],
              name + " maximum progress")
        traces.append(trace)
        contacts.append(progress)
    left, right = traces
    common = min(len(left.time), len(right.time))
    check(np.array_equal(left.qpos[0], right.qpos[0]) and left.qpos[0, 0] == 0
          and np.array_equal(left.qvel[0], right.qvel[0]) and np.array_equal(left.command[:common], right.command[:common]),
          "both evaluations must start at the same pose, velocity and use the same command")
    return traces, contacts


def mark_line(mujoco, scene, x, y, ground_height):
    check(scene.ngeom < scene.maxgeom, "no decoration slot for the 0.40 m line")
    geometry = scene.geoms[scene.ngeom]
    kind = mujoco.mjtGeom.mjGEOM_CAPSULE
    mujoco.mjv_initGeom(geometry, kind, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                        np.array([.97, .83, .25, 1.], dtype=np.float32))
    z = float(ground_height("up", x)) + .005
    mujoco.mjv_connector(geometry, kind, .006, np.array([x, y - .68, z]), np.array([x, y + .68, z]))
    geometry.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    scene.ngeom += 1


def encode(output, traces, contacts, receipt, plan):
    from PIL import Image, ImageDraw
    import mujoco
    from tools87.terrain87 import ground_height, load_model

    check(os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1",
          "use the software offscreen video unit")
    model = load_model("up")
    data, cam, options = mujoco.MjData(model), mujoco.MjvCamera(), mujoco.MjvOption()
    model.vis.global_.offwidth, model.vis.global_.offheight = view.PANE_W, view.PANE_H
    floor = model.geom("floor").id
    model.geom_matid[floor] = -1
    model.geom_rgba[floor] = (.71, .74, .75, 1.)
    model.vis.headlight.ambient[:] = .48
    options.geomgroup[3] = 0
    positions = np.concatenate([r.qpos[:, :2] for r in traces])
    low, high = positions.min(axis=0), positions.max(axis=0)
    cx, cy = (low + high) / 2
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation = 100., -13.
    cam.distance = max(1.58, .75 * (1.18 + high[0] - low[0]) + .29 * (high[1] - low[1]))
    cam.lookat[:] = (cx, cy, float(ground_height("up", cx)) + .21)
    fonts = {"title": view.chinese_font(27, True), "label": view.chinese_font(21, True),
             "data": view.chinese_font(18), "small": view.chinese_font(15)}
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-n", "-threads", "1", "-f", "rawvideo", "-pix_fmt",
           "rgb24", "-s", f"{view.WIDTH}x{view.HEIGHT}", "-r", str(view.FPS), "-i", "-", "-an",
           "-filter_threads", "1", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-maxrate", "2600k",
           "-bufsize", "1500k", "-threads", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    toe = plan["curriculum"]["toe_x_m"]
    mark = plan["evaluation"]["course_mark_m_past_toe"]
    reference = max(traces, key=lambda r: len(r.time))
    with mujoco.Renderer(model, width=view.PANE_W, height=view.PANE_H) as renderer, subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL) as encoder:
        try:
            for frame in range(view.FRAMES):
                tick = view.source_tick(frame)
                canvas = Image.new("RGB", (view.WIDTH, view.HEIGHT), view.BACKGROUND)
                draw = ImageDraw.Draw(canvas)
                draw.text((18, 8), "D1 · 同一起点的 4° 上坡", font=fonts["title"], fill=view.TEXT)
                cmd_vx = reference.command[min(tick, len(reference.time) - 1)]
                draw.text((view.WIDTH-18, 17), f"指令 {cmd_vx:.2f} m/s　 训练 {receipt['training_seed']}　"
                          f"测试 {receipt['reset_seed']}　 1× 原速　 {tick * view.DT:.2f} s",
                          font=fonts["data"], fill=view.MUTED, anchor="ra")
                for side, trace in enumerate(traces):
                    i, x = min(tick, len(trace.time) - 1), side * (view.PANE_W + view.GAP)
                    draw.rectangle((x, 54, x + 5, view.SCENE_TOP - 2), fill=view.ACCENTS[side])
                    draw.text((x + 17, 53), LABELS[side], font=fonts["label"], fill=view.ACCENTS[side])
                    draw.text((x + view.PANE_W - 12, 60), f"有效测距 {int(trace.valid_beams[i])}/8",
                              font=fonts["small"], fill=view.MUTED, anchor="ra")
                    data.qpos[:], data.qvel[:], data.time = trace.qpos[i], trace.qvel[i], trace.time[i]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=cam, scene_option=options)
                    view.add_marks(mujoco, renderer.scene, "up", toe, ground_height, low[0] - 1.4, high[0] + 1.4, cy)
                    mark_line(mujoco, renderer.scene, toe + mark, cy, ground_height)
                    canvas.paste(Image.fromarray(renderer.render()), (x, view.SCENE_TOP))
                    progress = contacts[side][i]
                    location = f"{progress:+.3f} m" if np.isfinite(progress) else "未记录"
                    draw.text((x+17, 403), f"机身 {trace.speed[i]:+.3f} m/s　 承重前轮距坡脚 {location}",
                              font=fonts["data"], fill=view.TEXT)
                    if trace.meta["terminated"] and tick >= len(trace.time) - 1:
                        draw.text((x+view.PANE_W-10, 404), "终止·定格", fill="#ffc793", font=fonts["small"], anchor="ra")
                draw.text((18, 438), "机身前向速度 (m/s) · 白线为指令 · 两侧轮速均直接取自策略，手工增量为 0",
                          font=fonts["small"], fill=view.MUTED)
                draw.text((view.WIDTH-18, 438), f"橙线为坡脚　黄线为坡脚后 {mark:.2f} m",
                          font=fonts["small"], fill=view.MUTED, anchor="ra")
                view.draw_speed(draw, traces, tick, fonts)
                encoder.stdin.write(canvas.tobytes())
                receipt["frames_written"] += 1
        finally:
            encoder.stdin.close()
        check(encoder.wait() == 0, "ffmpeg encoding")


def render():
    check_worker("video")
    plan, header = read(PLAN), complete_stage("evaluate")
    root = BASE / "evaluate_01"
    traces, contacts = paired_sources(root, header, plan)
    output = BASE / "uphill89.mp4"
    if output.exists():
        raise FileExistsError(output)
    receipt = {"status": "INCOMPLETE", "protocol_sha256": sha(PLAN), "source_sha256": sha(__file__),
               "training_seed": traces[0].meta["training_seed"], "reset_seed": traces[0].meta["seed"],
               "terrain": "up", "unmodified_policy_output": True, "physics_steps": 0, "policy_calls": 0,
               "frames_written": 0, "source_cases": {r.meta["arm"]: {"case": r.meta["case"], **r.hashes} for r in traces}}
    with output.with_suffix(".json").open("x") as journal:
        try:
            encode(output, traces, contacts, receipt, plan)
            probe = subprocess.run(["ffprobe", "-v", "error", "-threads", "1", "-count_frames", "-select_streams", "v:0",
                                    "-show_entries", "stream=width,height,nb_read_frames,duration,codec_name,r_frame_rate",
                                    "-of", "json", str(output)], check=True, capture_output=True, text=True, timeout=60)
            stream, = json.loads(probe.stdout)["streams"]
            check(receipt["frames_written"] == int(stream["nb_read_frames"]) == view.FRAMES
                  and (stream["width"], stream["height"]) == (view.WIDTH, view.HEIGHT)
                  and stream["r_frame_rate"] == "25/1" and stream["codec_name"] == "h264"
                  and abs(float(stream["duration"]) - view.DURATION) <= 1e-4
                  and 0 < output.stat().st_size <= 5_000_000, "encoded clip integrity")
            for r in traces:
                check(all(sha(root / r.meta["case"] / file) == digest for file, digest in r.hashes.items()),
                      "source changed during rendering")
            receipt.update(status="COMPLETE", bytes=output.stat().st_size, sha256=sha(output), stream=stream, fps=view.FPS)
        except BaseException as exc:
            receipt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            json.dump(receipt, journal, indent=2, allow_nan=False)
            journal.write("\n")
    return receipt


if __name__ == "__main__":
    print(json.dumps(render()))
