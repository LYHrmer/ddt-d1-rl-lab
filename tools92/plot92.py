"""Plot the effect of the same wheel intervention under two model action sources."""
import argparse
import hashlib
import json
import math
from pathlib import Path


MODELS = (91011, 91023, 91037)
COLORS = ("#0072A8", "#BA4D11", "#007D69")
MARKERS = ("o", "s", "^")
MODES = ("closed", "fixed")
PLAN = Path(__file__).resolve().parents[1] / "docs/experiments92/fixed_actions.json"
PANELS = {
    "move": ((88101, 88102, 88103), 1, .010, "Driving uphill",
             "Gain in mean forward body speed, 4–6 s (m/s)"),
    "stand": ((90101, 90102, 90103), 1000, .003, "Standing on the incline",
              "Rollback prevented in the first second (mm)"),
}


def read_effects(report_path):
    report = json.loads(report_path.read_bytes())
    protocol = PLAN.read_bytes()
    if hashlib.sha256(protocol).hexdigest() != report["protocol_sha256"]:
        raise ValueError("report does not match the registered fixed-action protocol")
    rules = json.loads(protocol)["tasks"]
    for task, panel in PANELS.items():
        if tuple(rules[task]["reset_seeds"]) != panel[0] or rules[task]["effect_tolerance"] != panel[2]:
            raise ValueError("figure labels differ from registered resets or effect tolerance")
    measured, predicted = {}, {}
    keys = {(task, reset) for task, (resets, *_) in PANELS.items() for reset in resets}
    if sorted(m["seed"] for m in report["models"]) != list(MODELS):
        raise ValueError("figure requires each of the three registered model seeds")
    for model in report["models"]:
        pairs = model["pairs"]
        if len(pairs) != len(keys) or {(p["task"], p["reset_seed"]) for p in pairs} != keys:
            raise ValueError("figure requires one pair per registered task and reset")
        for pair in pairs:
            task, reset = pair["task"], pair["reset_seed"]
            for mode in MODES:
                grade = pair[mode]
                truth, effect = grade["true_effect"], grade["predicted_effect"]
                if not math.isfinite(truth) or measured.setdefault((task, reset), truth) != truth:
                    raise ValueError("mismatched physical reference")
                if grade["checks"]["both_complete"] != (effect is not None):
                    raise ValueError("completion flag disagrees with effect availability")
                if effect is not None and not math.isfinite(effect):
                    raise ValueError("invalid model effect")
                within = effect is not None and abs(truth - effect) <= rules[task]["effect_tolerance"]
                if within != grade["checks"]["effect_tolerance"]:
                    raise ValueError("plotted effect differs from the registered tolerance decision")
                predicted[task, reset, mode, model["seed"]] = effect
    return measured, predicted


def make_figure(measured, predicted):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator

    ink, muted = "#1C3041", "#486074"
    fig = Figure(figsize=(12.5, 6.2), dpi=150, facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=.082, right=.979, top=.69, bottom=.25, wspace=.23)
    fig.text(.082, .956, "How the world model predicts uphill wheel response",
             color=ink, fontsize=19, weight="bold", va="top")
    fig.text(.082, .875, "Open symbols on the left use policy feedback. Filled symbols on the right replay recorded actions.",
             fontsize=11, color=muted)
    handles = [Line2D([], [], color=ink, marker="D", lw=1.0, ms=5, label="Archived physics")]
    for seed, color, marker in zip(MODELS, COLORS, MARKERS, strict=True):
        handles.append(Line2D([], [], ls="", marker=marker, color=color, ms=7, label=f"Model {seed}"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.081, .86), frameon=False,
               ncol=4, fontsize=10, borderaxespad=0, labelcolor=ink, handletextpad=.6, columnspacing=2.3)
    for ax, (task, (resets, factor, tol, title, quantity)) in zip(axes, PANELS.items(), strict=True):
        for pos, reset in enumerate(resets):
            truth = measured[task, reset] * factor
            ax.add_patch(Rectangle((pos-.425, truth-tol*factor), .85, 2*tol*factor,
                                   facecolor="#E7EFF2", edgecolor="#B7C9D0", lw=.65, zorder=1))
            ax.hlines(truth, pos-.425, pos+.425, colors=ink, linewidth=.8, zorder=2)
            ax.plot(pos, truth, ls="", marker="D", ms=5.5, color=ink, zorder=4)
            for mode, shift in (("closed", -.225), ("fixed", .225)):
                for i, (seed, color, marker) in enumerate(zip(MODELS, COLORS, MARKERS, strict=True)):
                    x = pos + shift + (i-1)*.092
                    effect = predicted[task, reset, mode, seed]
                    if effect is None:
                        ax.plot(x, .025, ls="", marker="x", color=color, ms=6,
                                transform=ax.get_xaxis_transform(), zorder=4)
                    else:
                        ax.plot(x, factor*effect, ls="", marker=marker, ms=7.3, mew=1.5,
                                mec=color, mfc="white" if mode == "closed" else color, zorder=4)
        ax.axhline(0, lw=1, ls=(0, (3, 3)), color="#6B7D89", zorder=2)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E1E6EA", lw=.7)
        ax.set_title(title, loc="left", color=ink, fontsize=14, weight="bold", pad=12)
        ax.set_ylabel(quantity, color=ink, fontsize=10)
        ax.set_xticks(range(len(resets)), [str(r) for r in resets])
        ax.set_xlabel("Physical reset seed", color=muted, fontsize=10)
        ax.set_xlim(-.53, len(resets)-.47)
        ax.margins(y=.11)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.tick_params(colors=muted, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#9CAEB8")
    fig.text(.082, .125, "Shading spans ±0.010 m/s around the measured driving effect and ±3 mm around the measured standing effect.",
             color=muted, fontsize=10)
    fig.text(.082, .085, "Acceptance also requires two complete arms, a positive effect, lower error than predicting zero, and both arm errors within tolerance.",
             color=muted, fontsize=9.3)
    if any(v is None for v in predicted.values()):
        fig.text(.082, .045, "A cross at the lower edge marks an incomplete model pair.", color=muted, fontsize=9.3)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="public fixed_actions92.json")
    parser.add_argument("output_png", type=Path, help="new PNG file")
    args = parser.parse_args()
    if args.output_png.suffix.lower() != ".png":
        parser.error("output must be a new .png file")
    fig = make_figure(*read_effects(args.report))
    with args.output_png.open("xb") as out:
        fig.savefig(out, format="png", metadata={"Title": "D1 fixed-action replay"})


if __name__ == "__main__":
    main()
