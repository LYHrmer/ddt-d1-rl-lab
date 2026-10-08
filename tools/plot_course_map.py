#!/usr/bin/env python3
"""Draw an annotated map from the generated terrain manifest; zero physics."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Circle

ROOT = Path(__file__).resolve().parents[1]

def label(zone):
    n, p = zone["name"], zone["parameters"]
    if n.startswith("single_step"):
        return f"单台阶 {p['height_m']*1000:g} mm"
    if n.startswith("ramp_"):
        return f"坡道 {p['angle_deg']}°"
    if n.startswith("stairs_"):
        return f"阶梯 每级 {p['riser_m']*1000:g} mm"
    if n.startswith("rough_"):
        return f"起伏 0–{p['max_height_m']*1000:g} mm"
    if n.startswith("gate_"):
        return f"门净宽 {p['clear_width_m']:g} m"
    return {"fixed_broken_plates":"固定碎板", "slalom":"绕桩"}.get(n,n)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"evidence/terrain_course_03/map_zh.png")
    args=parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing map")
    scene=ROOT/"src/ddt_ros2_control/urdfs/d1_description/mujoco"
    manifest=json.loads((scene/"training_course_03_assets/layout_manifest.json").read_text())
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family":"Noto Sans CJK JP","axes.unicode_minus":False,
        "text.color":"#eaf1f7","axes.labelcolor":"#c4d3e0","xtick.color":"#a6bbcc","ytick.color":"#a6bbcc"})
    fig,ax=plt.subplots(figsize=(15,11),facecolor="#0d1722")
    ax.set_facecolor("#142331")
    colors={"easy":"#59b895","moderate":"#d7a64b","hard":"#df7975"}
    for z in manifest["zones"]:
        if z["name"]=="flat_connectors": continue
        x0,y0,x1,y1=z["bounds_xy_m"]
        color=colors[z["difficulty"]]
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor=color,alpha=.15,edgecolor=color,lw=1.5))
        if z["name"]=="straight_acceleration_braking":
            ax.text(8,0.55,"20 m 平地加速 / 制动道",ha="center",fontsize=14,color="#d4e8f0")
            ax.annotate("",(17,0),(1,0),arrowprops={"arrowstyle":"->","color":"#c2d7e7","lw":1.5})
        else:
            ax.text((x0+x1)/2,(y0+y1)/2,label(z),ha="center",va="center",fontsize=10.5,color=color)
            ax.annotate("",(x1-.12,y0+.16),(x0+.12,y0+.16),arrowprops={"arrowstyle":"->","color":color,"lw":1})
    ax.add_patch(Circle((0,0),.22,facecolor="#f7fafc",edgecolor="#6dcdec",lw=2))
    ax.text(-.3,-.62,"起点 (0, 0)",ha="center",fontsize=10)
    ax.set_xlim(-3,19); ax.set_ylim(-12,11)
    ax.set_xticks(range(-2,20,2)); ax.set_yticks(range(-12,12,2))
    ax.grid(color="#607d91",alpha=.17,lw=.7); ax.set_axisbelow(True); ax.set_aspect("equal")
    ax.set_xlabel("世界 X / m →",fontsize=12); ax.set_ylabel("世界 Y / m →",fontsize=12)
    for spine in ax.spines.values(): spine.set_color("#324a5d")
    fig.suptitle("D1 参数化地形训练场 · 03",fontsize=23,fontweight="bold",y=.97)
    fig.text(.5,.923,"地图显示分区范围；箭头为建议接近方向，区间由平地连接",ha="center",color="#a8bdce",fontsize=12)
    fig.text(.5,.043,"绿 / 黄 / 红：设计难度分档，尚未验证策略通过率   ·   静态地图，0 个物理步",ha="center",fontsize=11,color="#becbd7")
    fig.subplots_adjust(top=.885,bottom=.10,left=.12,right=.90)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("xb") as stream: fig.savefig(stream,format="png",dpi=120,facecolor=fig.get_facecolor())
    plt.close(fig); print(args.output)

if __name__=="__main__": main()
