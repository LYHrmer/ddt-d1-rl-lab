"""Show the measured round-96 robot beside the three late model-response estimates."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from tools87.video87 import chinese_font
from tools94.common94 import ROOT, digest, read
from tools96 import video96 as parent_video
from tools97.common97 import BASE, PARENT, PLAN_SHA, protocol


CUT, END, FPS = 16, 32, 25
TAGS = ("p1", "zero")
METHODS = ("native", "closed", "fixed", "anchored")
LABELS = ("原生实测", "原闭环预测", "固定动作预测", "固定动作＋真状态预测")
COLORS = ("#FFE0A1", "#C8C2FF", "#7CC7FF", "#62DDAC")
WIDTH, HEIGHT = 1344, 592
CROP_W, CROP_H = parent_video.PANE_W, parent_video.PANE_H
CROP_X, CROP_Y = 4, parent_video.SCENE_Y


def require(condition, message):
    if not condition:
        raise ValueError(message)


def f32sha(array):
    return sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()


@dataclass
class Branch:
    x: np.ndarray             # scoring endpoints only, starting at global tick 17
    info: dict


def suffix_branch(folder, seal, used, stage, seed, reset, tag, native, forecast, tape, checkpoint):
    rel = Path(f"seed_{seed}/reset_{reset}/paired/{tag}")
    receipt, states_file = (folder / rel).with_suffix(".json"), (folder / rel).with_suffix(".npz")
    receipt_sha = parent_video.sealed_input(folder, receipt, seal, used)
    source_sha = parent_video.sealed_input(folder, states_file, seal, used)
    info = {"receipt": str(receipt.relative_to(folder)), "source": str(states_file.relative_to(folder)),
            "receipt_sha256": receipt_sha, "source_sha256": source_sha}
    if not receipt_sha and not source_sha:
        return Branch(np.empty(0), {**info, "status": "MISSING", "scored_ticks": 0})
    row = read(receipt)
    require(receipt_sha and source_sha and source_sha == row["state_file_sha256"], f"suffix file identity: {rel}")
    require((row["stage"], row["seed"], row["reset"], row["model"], row["tag"], row["alpha"], row["cut_tick"], row["end_tick"])
            == (stage, seed, reset, "paired", tag, 1. if tag == "p1" else 0., CUT, END), f"suffix arm identity: {rel}")
    base = forecast if stage == "fixed" else native
    seed_state = np.asarray(base.trace.states[CUT], dtype=np.float32)
    sources = dict(native_trace_sha256=native.info["source_sha256"], forecast96_sha256=forecast.info["source_sha256"],
                   checkpoint_sha256=checkpoint, start_state_sha256=f32sha(seed_state), applied_tape_sha256=f32sha(tape))
    require(all(row[key] == value for key, value in sources.items()), f"suffix start, actions or world differ: {rel}")
    with np.load(states_file, allow_pickle=False) as saved:
        states, applied = saved["state45"], saved["applied16"]
        n, attempts = row["ticks"], row["attempted_model_steps"]
        require(0 <= n <= attempts <= END-CUT and n == row["retained_model_steps"] and
                states.shape == (n+1,45) and states.dtype == np.float32 and np.isfinite(states).all()
                and np.array_equal(states[0], seed_state) and np.array_equal(applied, tape[:attempts]),
                f"invalid retained states or wrong applied actions: {rel}")
        require((n == END-CUT) == (row["status"] == "COMPLETE") and
                ((row.get("error") is None) == (n == END-CUT)), f"incorrect suffix status: {rel}")
        x = states[1:,0].astype(np.float64)
    info.update(**sources, scored_ticks=n, status=row["status"], error=row.get("error"))
    return Branch(x, info)


def recorded_suffix(arm):
    if not arm.trace:
        return Branch(np.empty(0), {**arm.info, "scored_ticks": 0})
    x = arm.trace.states[CUT+1:END+1,0].astype(np.float64)
    return Branch(x, {**arm.info, "scored_ticks": len(x)})


def gains(positive, zero, origin):
    n = min(len(positive.x), len(zero.x))
    if not n:
        return np.empty(0)
    reduction = np.abs(zero.x[:n]-origin) - np.abs(positive.x[:n]-origin)
    return 1000*np.cumsum(reduction, dtype=np.float64) / np.arange(1, n+1)


def pinned_media(path, revision, used):
    # The comparison uses only the measured left pane of the previously published movie.
    rel = str(path.relative_to(ROOT))
    raw = path.read_bytes()
    expected = subprocess.check_output(["git", "--no-replace-objects", "-C", str(ROOT), "rev-parse", revision+":"+rel])
    current = subprocess.check_output(["git", "--no-replace-objects", "-C", str(ROOT), "hash-object", "--stdin"], input=raw)
    require(expected.strip() == current.strip(), f"published media differs from the registered parent commit: {rel}")
    used[path] = sha256(raw).hexdigest()


def timeline(source, order):
    require(source["status"] == "COMPLETE" and source["fps"] == FPS and source["cover_frames"] == 0 and
            source["displayed_alpha"] == 1 and source["reference_alpha"] == 0, "unexpected source movie semantics")
    require([(c["seed"],c["reset"]) for c in source["cells"]] == order, "source footage is not in the registered order")
    cursor = 0
    for c in source["cells"]:
        last = max(p["p1"].get("ticks",0) for p in c["panels"].values())
        require(c["first_frame"] == cursor and c["endpoint_frame"] == cursor + last and
                c["last_frame"] == c["endpoint_frame"] + source["endpoint_hold_frames"], "invalid clip mapping")
        cursor = c["last_frame"] + 1
    require(cursor == source["frames"] == int(source["stream"]["nb_read_frames"]), "source frame count differs")


def load_data():
    plan = protocol(private=True)
    order = [(s,r) for s in plan["training_seeds"] for r in plan["resets"]]
    used = {}
    source_movie = ROOT / "docs/assets/paired_response96.mp4"
    source_receipt = ROOT / "reports/paired_response96_video.json"
    for path in (source_movie, source_receipt):
        pinned_media(path, plan["parent_commit"], used)
    source = read(source_receipt)
    require(used[source_movie] == source["video_sha256"] and source["stream"]["width"] == parent_video.WIDTH and
            source["stream"]["height"] == parent_video.HEIGHT, "incorrect source movie or crop geometry")
    timeline(source, order)
    public_path = ROOT / "reports/paired_response96.json"
    used[public_path] = digest(public_path)
    require(used[public_path] == plan["parents_sha256"][str(public_path.relative_to(ROOT))], "round-96 public report changed")
    public = read(public_path)
    checkpoints = {f["seed"]: f["arms"]["paired"]["checkpoint"]["sha256"] for f in public["fits"]}
    dirs = {key: parent / f"{key}_01" for key, parent in (("native", PARENT), ("predictions", PARENT),
                                                          ("fixed", BASE), ("anchored", BASE))}
    seals, seal_hash = {}, {}
    for key, folder in dirs.items():
        path = folder/"seal.json"
        seals[key], seal_hash[key] = read(path), digest(path)
        used[path] = seal_hash[key]
        if key in ("native","predictions"):
            require(seal_hash[key] == plan["local_parent_seals_sha256"][key] == source[key+"_seal_sha256"],
                    f"incorrect archived {key} stage")
        else:
            previous = seal_hash["native" if key == "fixed" else "fixed"]
            require(seals[key]["stage"] == key and seals[key]["protocol_sha256"] == PLAN_SHA and
                    seals[key]["previous_seal_sha256"] == previous, f"incorrect round-97 {key} stage")
    cells = []
    for item in source["cells"]:
        seed, reset = item["seed"], item["reset"]
        rel = Path(f"seed_{seed}/reset_{reset}")
        lines = {kind: {} for kind in METHODS}
        physical_start = None
        for tag in TAGS:
            n = parent_video.native_arm(dirs["native"], dirs["native"]/rel/tag, reset, seals["native"], used)
            f = parent_video.model_arm(dirs["predictions"], dirs["predictions"]/rel/"paired", tag,
                                       seals["predictions"], used, seed, reset)
            require(n.trace and f.trace and min(n.trace.ticks,f.trace.ticks) >= CUT, "missing original recorded branch")
            for arm, slot in ((n,"native"),(f,"paired")):
                record = item["panels"][slot][tag]
                require(all(arm.info.get(k) == record.get(k) for k in
                            ("source_sha256","receipt_sha256","ticks","safe","stop_reason")), f"clip mapping changed: {rel}/{tag}")
            origin = n.trace.states[0]
            require(np.array_equal(f.trace.states[0], origin.astype(np.float32)) and
                    (physical_start is None or np.array_equal(origin, physical_start)), "different physical reset")
            physical_start = origin
            lines["native"][tag], lines["closed"][tag] = recorded_suffix(n), recorded_suffix(f)
            with np.load(n.trace.path, allow_pickle=False) as archive:
                tape = np.ascontiguousarray(archive["applied16"][CUT:END], dtype=np.float32)
            for stage in ("fixed","anchored"):
                lines[stage][tag] = suffix_branch(dirs[stage], seals[stage], used, stage, seed, reset, tag,
                                                   n, f, tape, checkpoints[seed])
        origin_x = float(physical_start[0])
        curves = {m: gains(lines[m]["p1"], lines[m]["zero"], origin_x) for m in METHODS}
        summary = {m: {**{tag: lines[m][tag].info for tag in TAGS}, "common_scored_ticks": len(curves[m]),
                       "suffix_gain_mm": float(curves[m][-1]) if len(curves[m]) == END-CUT else None} for m in METHODS}
        record = dict(seed=seed, reset=reset, physical_reset_x_m=origin_x, series=summary,
                      first_frame=item["first_frame"], endpoint_frame=item["endpoint_frame"], last_frame=item["last_frame"])
        cells.append((record,curves))
    return cells, source, source_movie, used, seal_hash


def graph_limits(cells):
    all_values = np.concatenate([np.zeros(1), *(series for _, curves in cells for series in curves.values())])
    lo, hi = float(all_values.min()), float(all_values.max())
    interval = max((hi-lo)/4, .01)
    unit = 10**math.floor(math.log10(interval))
    step = next(m*unit for m in (1,2,5,10) if m*unit >= interval)
    return min(0,math.floor(lo/step))*step, max(1,math.ceil(hi/step))*step, step


def frame(cell, curves, index, tick, measured, fonts, limits):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (WIDTH,HEIGHT), parent_video.BG)
    draw = ImageDraw.Draw(image)
    title, label, small = fonts
    draw.text((16,4), "D1 驻坡 · 第 16 拍以后，模型的预测怎样变化？", font=title, fill=parent_video.INK)
    draw.text((WIDTH-16,9), f"{index}/9    策略 {cell['seed']}    起点 {cell['reset']}    {tick/50:.2f} s · 0.5×",
              font=small, fill=parent_video.MUTED, anchor="ra")
    draw.text((16,58), "原生 MuJoCo 实测　·　策略参数 +1 步", font=label, fill=COLORS[0])
    image.paste(measured,(16,98))
    draw.text((16,389), "橙线为驻车起点", font=small, fill=parent_video.MUTED)
    for i, tag in enumerate(TAGS):
        arm = cell["series"]["native"][tag]
        reason = arm.get("stop_reason")
        if reason and tick >= arm["ticks"]:
            shown = "+1 步" if tag == "p1" else "零步对照"
            draw.text((16,421+i*24), f"{shown}　{reason} · {arm['ticks']/50:.2f} s 定格", font=small, fill="#FFD5A1")
    draw.text((552,58), "相对零步对照，平均少偏离驻车点多少？（毫米）", font=label, fill=parent_video.INK)
    x0,x1,y0,y1 = 610,1300,105,335
    low,high,step = limits
    xt = lambda t: x0+(t-CUT)*(x1-x0)/(END-CUT)
    yt = lambda v: y1-(v-low)*(y1-y0)/(high-low)
    for i in range(round((high-low)/step)+1):
        value = low+i*step
        y = yt(value)
        draw.line((x0,y,x1,y), fill="#75858A" if abs(value) < 1e-10 else "#374855", width=1)
        draw.text((x0-10,y), f"{value:g}", font=small, fill=parent_video.MUTED, anchor="rm")
    for t in (16,20,24,28,32):
        x = xt(t)
        draw.line((x,y1,x,y1+4),fill=parent_video.MUTED)
        draw.text((x,y1+9), f"{t} 拍",font=small,fill=parent_video.MUTED,anchor="ma")
    count = max(0,tick-CUT)
    if count == 0:
        draw.text(((x0+x1)/2,(y0+y1)/2),"第 16 拍分叉，从第 17 拍开始计分",font=small,fill=parent_video.MUTED,anchor="mm")
    for i,m in enumerate(METHODS):
        g = curves[m]
        visible = min(count,len(g))
        points = [(xt(CUT+j+1),yt(float(g[j]))) for j in range(visible)]
        if len(points)>1:
            draw.line(points,fill=COLORS[i],width=4)
        if points:
            x,y=points[-1]
            draw.ellipse((x-3,y-3,x+3,y+3),fill=COLORS[i])
        x,y = 556+(i%2)*391,390+(i//2)*40
        draw.line((x,y+12,x+25,y+12),fill=COLORS[i],width=4)
        text = f"{g[count-1]:+.3f} mm" if 0 < count <= len(g) else ("缺测" if count else "待计分")
        draw.text((x+35,y),f"{LABELS[i]}  {text}",font=small,fill=COLORS[i])
    draw.line((16,486,WIDTH-16,486),fill="#435360")
    draw.text((16,499), "原闭环由策略根据预测状态给动作。固定动作从模型的第 16 拍状态出发，改用逐拍记录的真实动作。",font=small,fill=parent_video.INK)
    draw.text((16,536), "最后一种沿用同一串真实动作，仅在第 16 拍换成原生状态，随后逐拍由模型预测。收益仅计第 17～32 拍。",
              font=small,fill=parent_video.INK)
    return image


def encode(source_movie, output, cells, report):
    from PIL import Image

    common = ["-nostdin","-hide_banner","-loglevel","error","-threads","1","-filter_threads","1"]
    decode = ["ffmpeg",*common,"-i",str(source_movie),"-map","0:v:0","-an","-sn","-vf",
              f"crop={CROP_W}:{CROP_H}:{CROP_X}:{CROP_Y}","-pix_fmt","rgb24","-c:v","rawvideo","-threads","1","-f","rawvideo","-"]
    encode_cmd = ["ffmpeg",*common,"-n","-f","rawvideo","-pix_fmt","rgb24","-s",f"{WIDTH}x{HEIGHT}","-r",str(FPS),
                  "-i","-","-an","-c:v","libx264","-preset","veryfast","-crf","20","-maxrate","1050k",
                  "-bufsize","1050k","-threads","1","-x264-params","threads=1:lookahead-threads=1","-pix_fmt","yuv420p",
                  "-movflags","+faststart",str(output)]
    fonts = chinese_font(28,True), chinese_font(22,True), chinese_font(18)
    chunk = CROP_W*CROP_H*3
    with subprocess.Popen(decode,stdout=subprocess.PIPE) as decoder, subprocess.Popen(encode_cmd,stdin=subprocess.PIPE) as encoder:
        try:
            for i,(cell,curves) in enumerate(cells,1):
                pixels = None
                for j in range(cell["first_frame"],cell["last_frame"]+1):
                    crop = decoder.stdout.read(chunk)
                    require(len(crop) == chunk, f"missing measured crop for source frame {j}")
                    report["decoded_native_crops"] += 1
                    if j <= cell["endpoint_frame"]:
                        measured = Image.frombytes("RGB",(CROP_W,CROP_H),crop)
                        pixels = frame(cell,curves,i,j-cell["first_frame"],measured,fonts,report["graph_limits_mm"]).tobytes()
                    encoder.stdin.write(pixels)
                    report["frames"] += 1
            require(not decoder.stdout.read(1),"unexpected extra frames in measured movie")
        finally:
            decoder.stdout.close()
            encoder.stdin.close()
        require(decoder.wait() == 0 and encoder.wait() == 0,"video decoder or encoder failed")


def render(output, receipt):
    if output.exists() or receipt.exists() or output == receipt:
        raise FileExistsError("use unused video and receipt paths")
    report = dict(status="PARTIAL", file=output.name, frames=0, fps=FPS, cover_frames=0, decoded_native_crops=0,
                  source_dt_s=.02, playback_speed=.5, cut_tick=CUT, scored_ticks=[CUT+1,END], model="paired",
                  displayed_alpha=1., reference_alpha=0., source_crop_xywh=[CROP_X,CROP_Y,CROP_W,CROP_H],
                  new_physics_steps=0, forward_kinematics_calls=0, model_steps=0, policy_calls=0, training_updates=0)
    with receipt.open("x",encoding="utf-8") as journal:
        try:
            cells, source, movie, used, seal_hash = load_data()
            report.update(protocol_sha256=PLAN_SHA, source_video_sha256=used[movie],
                          source_video_receipt_sha256=used[ROOT/"reports/paired_response96_video.json"],
                          stage_seals_sha256=seal_hash, endpoint_hold_frames=source["endpoint_hold_frames"],
                          cells=[c for c,_ in cells], graph_limits_mm=graph_limits(cells))
            encode(movie,output,cells,report)
            probe = subprocess.run(["ffprobe","-v","error","-threads","1","-select_streams","v:0","-count_frames",
                                    "-show_entries","stream=width,height,nb_read_frames,duration,codec_name,r_frame_rate",
                                    "-of","json",str(output)],check=True,capture_output=True,text=True,timeout=100)
            stream, = json.loads(probe.stdout)["streams"]
            require(report["frames"] == report["decoded_native_crops"] == source["frames"] == int(stream["nb_read_frames"])
                    and (stream["width"],stream["height"],stream["codec_name"],stream["r_frame_rate"])==(WIDTH,HEIGHT,"h264","25/1")
                    and abs(float(stream["duration"])-report["frames"]/FPS)<.0001 and 0<output.stat().st_size<=5_000_000,
                    "encoded movie size, frame count or duration differs")
            require(all(digest(path)==expected for path,expected in used.items()),"input changed while rendering")
            report.update(status="COMPLETE", video_sha256=digest(output), bytes=output.stat().st_size,
                          duration_s=report["frames"]/FPS, stream=stream)
        except BaseException as error:
            report["reason"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            json.dump(report,journal,ensure_ascii=False,allow_nan=False,indent=2)
            journal.write("\n")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    result=render(**vars(parser.parse_args()))
    print(json.dumps({k:result[k] for k in ("status","frames","duration_s","video_sha256")}))
