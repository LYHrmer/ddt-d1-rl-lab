"""Publish a GIF of the selected trial and the same policies on the opposite slope."""
import argparse
import json
from pathlib import Path
import subprocess

from tools35.render_showcase35 import encode_gif
from tools87.video87 import digest, render


def preview(root):
    receipt = json.loads((root / "showcase_01.json").read_text())
    video = root / "showcase_01.mp4"
    if receipt["status"] != "COMPLETE" or digest(video) != receipt["sha256"]:
        raise ValueError("complete original representative video required")
    target = root / "previews_01"
    target.mkdir()
    frames = target / "png"
    frames.mkdir()
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-n", "-threads", "1",
                    "-i", str(video), "-an", "-filter_threads", "1", "-vf", "fps=20,scale=760:-2",
                    "-threads", "1", str(frames / "frame_%03d.png")], check=True)
    paths = sorted(frames.glob("frame_*.png"))
    if len(paths) != 120:
        raise ValueError("the six-second preview must contain 120 frames")
    result = encode_gif(paths, target / "showcase.gif")
    evaluation = root / "evaluate_01"
    selected = json.loads((evaluation / "report.json").read_text())["video_selection"]
    other = "down" if selected["terrain"] == "up" else "up"
    seed, train = selected["seed"], selected["training_seed"]
    alternate = render(evaluation / f"baseline_0_{other}_{seed}", evaluation / f"range_{train}_{other}_{seed}",
                       target / f"{other}.mp4")
    with (target / "report.json").open("x") as stream:
        json.dump({"status": "COMPLETE", "source_sha256": receipt["sha256"], "gif": result,
                   "opposite_slope": other, "opposite_slope_sha256": alternate["sha256"],
                   "png_frames": len(paths), "new_physics_steps": 0}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    preview(parser.parse_args().root)
