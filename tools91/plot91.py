"""Plot the six intervention effects from the public 91 score report."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1] / "reports/terrain_world91.json"
PROTOCOL_SHA = "aedb3c121d026f18b7d8197f2bfda91f001197a875704717f7e4192f6d0aa929"
ARCHIVED = (64001, 64002, 64003)
CANDIDATE = (91011, 91023, 91037)
COLORS = ("#0072B2", "#C44D1C", "#178268")
MARKERS = ("o", "s", "^")
TASKS = {
    "move": dict(resets=(88101, 88102, 88103), factor=1., tolerance=.010,
                 title="Driving uphill", label="Mean body-speed gain, 4–6 s (m/s)",
                 intervention="All four wheel targets +0.50 rad/s from 3.60 s"),
    "stand": dict(resets=(90101, 90102, 90103), factor=1000., tolerance=.003,
                  title="Standing on the incline", label="Rollback prevented in the first second (mm)",
                  intervention="Common target offset when the mean wheel target is negative"),
}


def read_effects():
    report = json.loads(REPORT.read_text())
    if report["protocol_sha256"] != PROTOCOL_SHA or report["physical_pairs"] != 6:
        raise ValueError("public scores must match the six registered 91 pairs and tolerances")
    physical, predictions = {}, {}
    for family, seeds in (("archived", ARCHIVED), ("candidate", CANDIDATE)):
        models = report["families"][family]["models"]
        if len(models) != 3 or {m["seed"] for m in models} != set(seeds):
            raise ValueError(f"incomplete {family} seed set")
        for model in models:
            seed = model["seed"]
            for pair in model["pairs"]:
                task, reset = pair["task"], pair["reset_seed"]
                style = TASKS[task]
                key = task, reset
                measured, predicted = pair["true_effect"], pair["predicted_effect"]
                if (reset not in style["resets"] or (seed, *key) in predictions
                        or not pair["checks"]["both_complete"] or predicted is None
                        or not math.isfinite(measured) or not math.isfinite(predicted)):
                    raise ValueError(f"incomplete or duplicate effect for {seed}, {key}")
                # The score report contains effect-gate booleans. Verify the
                # protocol-bound tolerances used to draw their physical limits.
                within = abs(predicted - measured) <= style["tolerance"]
                if (within != pair["checks"]["effect_tolerance"]
                        or physical.setdefault(key, measured) != measured):
                    raise ValueError(f"physical effect or tolerance differs for {seed}, {key}")
                predictions[(seed, *key)] = predicted
    if len(predictions) != 36 or len(physical) != 6:
        raise ValueError("all six physical effects and 36 model effects are required")
    return physical, predictions


def make_figure(physical, predictions):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import MaxNLocator

    ink, muted, archived = "#182B3F", "#475D6E", "#73828E"
    fill, edge = "#EAF0F3", "#B9C9D3"
    fig = Figure(figsize=(12, 6.4), dpi=150, facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=.078, right=.979, top=.66, bottom=.225, wspace=.21)
    fig.text(.078, .96, "Predicting the effect of wheel targets on a 4° incline",
             fontsize=20, fontweight="bold", va="top", color=ink)

    def point(marker, color, size):
        return Line2D([], [], ls="", marker=marker, markerfacecolor=color, markersize=size,
                      markeredgecolor="white", markeredgewidth=.65)

    handles = [point("D", ink, 7), point(MARKERS[0], COLORS[0], 7),
               Patch(facecolor=fill, edgecolor=edge), point(MARKERS[1], COLORS[1], 7),
               point("o", archived, 6), point(MARKERS[2], COLORS[2], 7)]
    leg = ["Physical simulation", "Terrain model 91011", "Allowed effect error (±0.010 m/s or ±3 mm)",
           "Terrain model 91023", "Archived 64001, 64002, 64003 (left to right)", "Terrain model 91037"]
    fig.legend(handles, leg, ncol=3, loc="upper left", mode="expand",
               bbox_to_anchor=(.078, .808, .901, .082), borderaxespad=0, frameon=False,
               handlelength=1.2, handletextpad=.5, labelspacing=.55, labelcolor=ink, fontsize=9.7)

    for ax, (task, style) in zip(axes, TASKS.items(), strict=True):
        factor = style["factor"]
        tolerance = style["tolerance"] * factor
        for pos, reset in enumerate(style["resets"]):
            measured = physical[task, reset] * factor
            ax.add_patch(Rectangle((pos - .40, measured - tolerance), .80, 2 * tolerance,
                                   facecolor=fill, edgecolor=edge, linewidth=.65, zorder=1))
            for i, seed in enumerate(ARCHIVED):
                ax.plot(pos - .29 + .09 * i, predictions[seed, task, reset] * factor,
                        ls="", marker="o", ms=5.5, mfc=archived, mec="white", mew=.55, zorder=3)
            ax.plot(pos, measured, ls="", marker="D", ms=6.8, mfc=ink, mec="white", mew=.7, zorder=4)
            for i, seed in enumerate(CANDIDATE):
                ax.plot(pos + .11 + .09 * i, predictions[seed, task, reset] * factor,
                        ls="", marker=MARKERS[i], ms=7., mfc=COLORS[i], mec="white", mew=.65, zorder=4)

        ax.axhline(0., color=muted, lw=1.1, ls=(0, (3, 3)), zorder=2)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DFE4E9", lw=.65)
        ax.set_title(style["title"], loc="left", fontsize=14, color=ink, fontweight="bold", pad=22)
        ax.text(0, 1.025, style["intervention"], transform=ax.transAxes, fontsize=9.5, color=muted)
        ax.set_ylabel(style["label"], fontsize=10.5, color=ink, labelpad=8)
        ax.set_xticks([0, 1, 2], [str(seed) for seed in style["resets"]])
        ax.set_xlabel("Reset seed", fontsize=10, color=muted, labelpad=5)
        ax.set_xlim(-.49, 2.49)
        ax.margins(y=.09)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.tick_params(axis="both", colors=muted, labelsize=9, length=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#9FAEB9")

    fig.text(.078, .092, "Pair acceptance also checks each arm's error (≤0.040 m/s or ≤30 mm), complete horizons, "
             "a positive effect and a smaller error than predicting zero.", color=muted, fontsize=9.6)
    fig.text(.078, .053, "A model must pass all six pairs. A family must qualify all three models.",
             color=muted, fontsize=9.6)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_png", type=Path, help="new PNG file")
    path = parser.parse_args().output_png
    if path.suffix.lower() != ".png":
        parser.error("output must be a new .png file")
    fig = make_figure(*read_effects())
    with path.open("xb") as stream:
        fig.savefig(stream, format="png")


if __name__ == "__main__":
    main()
