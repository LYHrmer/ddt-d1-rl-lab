"""Draw the D1's archived CAD meshes and the two registered simulated mount positions."""
import argparse
import os
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from tools94.common94 import ROOT, digest, read
from training.d1_rl.interface import DEFAULT_Q


WIDTH, HEIGHT = 980, 680
CAMERA = np.array([1.14, -1.44, .72])
TARGET = np.array([.02, 0., -.15])
FOV = 25.
INK, MUTED, OLD, NEW, BRACKET = "#203746", "#4B616D", "#713C80", "#AC4800", "#117371"


def _numbers(values):
    return " ".join(f"{v:.9g}" for v in values)


def _render(assembly):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("LP_NUM_THREADS", "1")
    import mujoco as mj

    config = read(ROOT / "training/configs/pilot03d.json")
    path = (ROOT / config["scene"]).with_name("robot.xml")
    if digest(path) != config["robot_xml_sha256"]:
        raise ValueError("D1 source mesh placements changed")
    tree = ET.parse(path)
    compiler = tree.find("compiler")
    meshdir = (path.parent / compiler.get("meshdir")).resolve()
    mounting = read(ROOT / "docs/experiments107/mount.json")
    if digest(meshdir / "funtion_Link.STL") != mounting["front_panel_mesh_sha256"]:
        raise ValueError("D1 front cover mesh changed")
    compiler.set("meshdir", str(meshdir))
    compiler.set("usethread", "false")
    world = tree.find("worldbody")
    base = world.find("body[@name='base_link']")
    for name, (lo, hi) in assembly.items():
        rgba = ".89 .42 .10 1" if "housing" in name else ".10 .49 .47 1"
        ET.SubElement(base, "geom", type="box", pos=_numbers((lo+hi)/2), size=_numbers((hi-lo)/2),
                      rgba=rgba, group="2", contype="0", conaffinity="0", density="0")
    visual = ET.SubElement(tree.getroot(), "visual")
    ET.SubElement(visual, "global", offwidth=str(WIDTH), offheight=str(HEIGHT))
    ET.SubElement(visual, "headlight", ambient=".43 .43 .43", diffuse=".56 .56 .56", specular=".12 .12 .12")
    ET.SubElement(tree.find("asset"), "texture", name="white", type="skybox", builtin="flat",
                  rgb1="1 1 1", rgb2="1 1 1", width="32", height="192")
    forward = TARGET-CAMERA
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    ET.SubElement(world, "camera", name="mount_view", pos=_numbers(CAMERA),
                  xyaxes=_numbers(np.r_[right, up]), fovy=str(FOV))
    model = mj.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))
    data = mj.MjData(model)
    data.qpos[:] = np.r_[0., 0., 0., 1., 0., 0., 0., DEFAULT_Q]
    mj.mj_fwdPosition(model, data)
    opt = mj.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[2] = 1
    opt.sitegroup[:] = 0
    with mj.Renderer(model, width=WIDTH, height=HEIGHT) as renderer:
        renderer.update_scene(data, camera="mount_view", scene_option=opt)
        rgb = renderer.render()

    def project(xyz):
        delta = np.atleast_2d(xyz) - CAMERA
        focal = HEIGHT/(2*np.tan(np.deg2rad(FOV/2)))
        xy = np.column_stack((delta @ right, -delta @ up))
        return np.array([WIDTH/2, HEIGHT/2]) + focal*xy/(delta @ forward)[:, None]

    return rgb, model, data, project


def _shell_section(model, data, height):
    """Intersect actual trunk visual triangles with a body-fixed horizontal plane."""
    import mujoco as mj
    ids = np.flatnonzero((model.geom_bodyid == model.body("base_link").id) &
                         (model.geom_type == mj.mjtGeom.mjGEOM_MESH))
    cuts = []
    for geom in ids:
        mesh = model.geom_dataid[geom]
        va, fa = model.mesh_vertadr[mesh], model.mesh_faceadr[mesh]
        verts = model.mesh_vert[va:va+model.mesh_vertnum[mesh]] @ data.geom_xmat[geom].reshape(3, 3).T
        verts += data.geom_xpos[geom]
        triangles = verts[model.mesh_face[fa:fa+model.mesh_facenum[mesh]]]
        mask = triangles[:, :, 2] >= height
        triangles = triangles[mask.any(axis=1) & ~mask.all(axis=1)]
        next_vertex = np.roll(triangles, -1, axis=1)
        edges = (triangles[:, :, 2] >= height) != (next_vertex[:, :, 2] >= height)
        start, end = triangles[edges], next_vertex[edges]
        fraction = (height-start[:, 2]) / (end[:, 2]-start[:, 2])
        cuts.append((start[:, :2] + fraction[:, None]*(end[:, :2] - start[:, :2])).reshape(-1, 2, 2))
    return np.concatenate(cuts)*1000


def _dimension(ax, x1, x2, y, label, text_above=True):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="<->", shrinkA=0, shrinkB=0, lw=1.2, color=INK))
    for x in (x1, x2):
        ax.plot([x, x], [y-2, y+2], color=INK, lw=1)
    ax.annotate(label, ((x1+x2)/2, y), xytext=(0, 3 if text_above else -4),
                textcoords="offset points", ha="center", va="bottom" if text_above else "top",
                fontsize=9.6, weight="bold", color=INK)


def draw(output):
    from tools107.mount107 import CONFIG, origins, parts

    old, new = (np.asarray(origins(arm)) for arm in ("original", "forward"))
    base_parts, assembly = parts("original"), parts("forward")
    supports = [k for k in assembly if "housing" not in k]
    if not supports or base_parts.keys() != assembly.keys():
        raise ValueError("unexpected support roster")
    for k in supports:
        np.testing.assert_array_equal(assembly[k], base_parts[k])
    np.testing.assert_allclose(new-old, np.broadcast_to([.070, 0, 0], old.shape), rtol=0, atol=1e-9)
    rgb, model, data, project = _render(assembly)
    shell = _shell_section(model, data, new[0, 2])
    front_ref = 1000*np.asarray(read(CONFIG)["front_panel_reference_m"][0])
    np.testing.assert_allclose(front_ref[2], 1000*new[0, 2])
    a, b = shell[:, 0], shell[:, 1]
    crosses = (a[:, 1] >= front_ref[1]) != (b[:, 1] >= front_ref[1])
    a, b = a[crosses], b[crosses]
    nose = np.max(a[:, 0]+(front_ref[1]-a[:, 1])*(b[:, 0]-a[:, 0])/(b[:, 1]-a[:, 1]))
    np.testing.assert_allclose(nose, front_ref[0], rtol=0, atol=.015)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP",
                         "DejaVu Sans"], "axes.unicode_minus": False, "text.color": INK,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig = plt.figure(figsize=(11.8, 6), facecolor="white")
    fig.text(.050, .932, "D1 前部双测距器的仿真安装位置", fontsize=21, weight="bold")
    fig.text(.051, .870, "旧位与候选位共用一对支架，两侧测距发射点（光心）沿车身正前方各移动 70 mm。", fontsize=11.4, color=MUTED)
    ax = fig.add_axes([.024, .148, .552, .652])
    ax.imshow(rgb)
    ax.set_axis_off()
    ax.set_title("厂方 D1 网格与候选安装件", loc="left", pad=1, fontsize=12.7, weight="bold")
    for xyz, marker, color in ((old, "o", OLD), (new, "D", NEW)):
        pix = project(xyz)
        ax.scatter(*pix.T, s=55, marker=marker, edgecolors=color if marker == "o" else "white",
                   facecolors="none" if marker == "o" else color, lw=1.6, zorder=3)
    start, end = project([[.20, -.12, -.39], [.38, -.12, -.39]])
    ax.annotate("车头", xy=end, xytext=start, fontsize=10.5, weight="bold", ha="right",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, shrinkA=4))

    view = fig.add_axes([.635, .208, .30, .570])
    view.set_aspect("equal")
    view.add_collection(LineCollection(shell, colors="#596771", linewidths=.85))
    for name, (minimum, maximum) in assembly.items():
        lo, hi = 1000*minimum, 1000*maximum
        if hi[1] < 0:
            continue
        sensor = "housing" in name
        view.add_patch(Rectangle(lo[:2], *(hi-lo)[:2], facecolor="#F4C99A" if sensor else "#A4D2CA",
                                 edgecolor=NEW if sensor else BRACKET, linewidth=1, zorder=4 if sensor else 3))
    oldx, newx = 1000*old[0, 0], 1000*new[0, 0]
    y = 1000*new[0, 1]
    view.plot([nose, nose], [45, y+24], color="#64737A", lw=.9, ls=(0, (4, 3)), zorder=2)
    view.scatter(oldx, y, s=64, facecolors="white", edgecolors=OLD, lw=1.65, zorder=6)
    view.scatter(newx, y, s=62, marker="D", facecolors=NEW, edgecolors="white", lw=1, zorder=6)
    _dimension(view, oldx, newx, y+14, "+70 mm")
    _dimension(view, nose, newx, 95, f"{newx-nose:.0f} mm")
    view.text(nose-2, 82, f"前盖 {nose:.0f}", ha="right", va="top", fontsize=9.3, color=MUTED)
    view.set(xlim=(300, 448), ylim=(42, 220), xticks=[310, 335, 355, 375, 400, 425],
             yticks=[60, 100, 150, 190], xlabel="距基座原点的 x（mm）　向前 →", ylabel="y（mm）　向左")
    view.set_title("左侧支架俯视　│　右侧镜像", loc="left", fontsize=12.7, weight="bold", pad=9)
    view.spines[["top", "right"]].set_visible(False)
    view.spines[["left", "bottom"]].set_color("#B4BFC3")
    view.tick_params(length=3, labelsize=8.6, color="#93A2AA")
    entries = [Patch(facecolor="#A4D2CA", edgecolor=BRACKET, label="两侧共用的支架设计"),
               Line2D([], [], ls="", marker="o", mfc="none", mec=OLD, mew=1.6, label=f"旧光心 x = {oldx:.0f} mm"),
               Line2D([], [], ls="", marker="D", mfc=NEW, mec=NEW, label=f"新光心 x = {newx:.0f} mm")]
    fig.legend(handles=entries, loc="center left", bbox_to_anchor=(.048,.141), ncol=3, frameon=False,
               fontsize=9.6, handlelength=1.2, columnspacing=1.35)
    fig.text(.05, .052, f"两侧光心 y = ±{y:.0f} mm、z = +{1000*new[0,2]:.0f} mm。灰线是该高度的机壳网格截面。"
             "腿轮取默认姿态。支架与测距器外壳为仿真几何，实物固定孔、强度与振动仍待测。", fontsize=9.5, color=MUTED)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Base-frame x (mm): original {oldx:.3f}, nose at y={front_ref[1]:g}/z={front_ref[2]:g} "
          f"{nose:.3f}, forward {newx:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets/d1_sensor_mount107.png")
    draw(parser.parse_args().output)
