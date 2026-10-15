"""
Results figure for the paper, generated from `results.jsonl`.

Deliberately a SCRIPT, not a hand-drawn figure: the grid is long-running and
resumable, so the figure must be regenerable at any moment and must state how
much of the grid it is based on. A figure that cannot say "n of N runs" invites
a reader to treat a partial grid as a finished result.

    python src/experiments/plot_results.py --results experiments/loso/results.jsonl

Panels
------
(A) five small multiples -- accuracy, macro-F1, edit, F1@50, stage-onset MAE --
    one bar per arm, error bars = std across folds x seeds. Every panel carries
    the IMAGE-FREE CONTROL as a dashed reference line: a predictor that splits
    each sequence into equal monotone runs without seeing a pixel. A bar that
    does not clear its own control is not evidence of anything, and the edit
    panel is marked disqualified for exactly that reason.
(B) order-violation rate, kept separate because lower is better and `hard` and
    `gated` are ~0 by construction. The panel is annotated to say so, because a
    reader who reads it as a win has misread the paper.
(C) the decisive comparison: `hard` vs `gated` paired per held-out site, on
    F1@50 -- the metric that survives both controls. A bar chart hides that the
    mean can move while most sites disagree; a paired plot cannot hide it.
(D) the hyper-parameter ablation, drawn as init -> learned dumbbells so that the
    finding is the ABSENCE of length.
(E) the time base. Panel (A5) is in frames, and a frame is not a unit anyone
    schedules in. This panel measures what a frame is actually worth in days on
    the one sequence carrying dates burnt into the image, against the constant a
    frames-to-days conversion assumes. It belongs in the results figure rather
    than the data section because it is what stops (A5) from being quoted in
    days -- a caveat placed three sections away from the number it qualifies is a
    caveat nobody reads.

The gated arm is NOT labelled "ours". This paper's result is that the rigid
constraint wins on this corpus; a figure that badges the learnable arm as the
contribution would contradict its own conclusion.

Colour follows the entity, never its rank: each arm keeps its hue whatever the
figure contains. The original four-arm palette passed all six checks of the
dataviz validator (lightness band, chroma floor, CVD separation, normal-vision
floor, contrast). Three hues have been added since -- teal for the LSTM, gold for
MS-TCN, plum for ASFormer -- and the seven-way palette has NOT been re-validated
against those checks. Seven categorical colours is past the point where CVD
separation can be assumed, so the figure never asks colour to carry a claim on
its own: every arm is also labelled on its axis, and the one comparison that
matters is drawn as a paired plot rather than by hue.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.lines import Line2D           # noqa: E402
from matplotlib.patches import Patch          # noqa: E402

# fixed hue per arm -- validated, never cycled, never reassigned by rank
ARM_COLOR = {
    "none":     "#3A6EA5",
    "free":     "#5F8F2E",
    "lstm":     "#3F8F8F",   # 5th hue, added with arm (b')
    "mstcn":    "#C9A227",   # 6th and 7th added with the TAS baselines
    "asformer": "#8E4585",
    "hard":     "#7B5EA7",
    "gated":    "#B3541E",
}
ARM_LABEL = {
    "none":     "per-frame (no sequence model)",
    "free":     "free CRF (no prior)",
    "lstm":     "LSTM encoder (no prior)",
    "mstcn":    "MS-TCN (no prior)",
    "asformer": "ASFormer (no prior)",
    "hard":     "hard monotonic",
    "gated":    "gated (learnable prior)",
}
#: unprimed arms first, in increasing order of how much they help, then the two
#: that carry the construction prior. The ordering is the argument: the reader
#: should be able to read left to right and see four architectures fail to reach
#: the control line before the primed arm clears it.
ARM_ORDER = ["none", "free", "lstm", "mstcn", "asformer", "hard", "gated"]

#: arms carrying no construction knowledge. Used only for the annotation that
#: states how many of them clear the control -- never for styling, since colour
#: here follows the entity and not its category.
UNPRIMED = ("none", "free", "lstm", "mstcn", "asformer")

#: short tick labels. At seven arms the full keys collide into each other on a
#: panel this width -- "lstmmstcnasformerhard" was the actual first render. The
#: legend carries the full names, so the axis only has to disambiguate.
ARM_TICK = {"none": "none", "free": "free", "lstm": "LSTM", "mstcn": "MS-TCN",
            "asformer": "ASFm", "hard": "hard", "gated": "gated"}

# Image-free control: uniform monotone split, no pixels, scored like the models --
# each sequence subsampled to 64 frames, per sequence, averaged over sites.
# src/experiments/monotone_floor.py
FLOOR = {"acc": 0.370, "macro_f1": 0.310, "edit": 0.800, "f1@50": 0.209,
         "onset_mae_frames": 7.46}

# Second-annotator disagreement on stage onsets, full corpus, in the SAME units
# and granularity as the bars in (A5): on the 64-frame evaluation subsample, one
# figure per sequence, then averaged. (In raw frames it is 3.69; the bars are not
# in raw frames.) src/data_collection/annotator_b_agreement.py
ANNOTATION_NOISE_FRAMES = 1.11

INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d8d8d8"
FLOOR_C, WARN = "#8f3a37", "#7a3a12"


def load(results_path: Path) -> list[dict]:
    rows = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    # arm key folds the temporal dimension in: the LSTM baseline is
    # mode="none" + temporal="lstm" and must not be collapsed onto per-frame
    for r in rows:
        t = r.get("temporal", "none")
        r["arm"] = t if t != "none" else r["mode"]
    return [r for r in rows if r["arm"] in ARM_ORDER]


def agg(vals):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return float("nan"), 0.0
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def style_axis(ax, ylabel=None, ymax=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if ymax:
        ax.set_ylim(0, ymax)


def bar_panel(ax, by_arm, key, title, higher_better=True, floor=None,
              floor_note=None, disqualified=False, extra_line=None):
    off_scale = False
    arms = [a for a in ARM_ORDER if a in by_arm]
    means, stds = [], []
    for a in arms:
        m, s = agg([r.get(key) for r in by_arm[a]])
        means.append(m); stds.append(s)
    xs = range(len(arms))
    ax.bar(xs, means, yerr=stds, width=0.62,
           color=[ARM_COLOR[a] for a in arms], capsize=3,
           error_kw=dict(ecolor=MUTED, lw=1.1))
    top = max([m + s for m, s in zip(means, stds) if m == m] + [0.1])
    if floor is not None and floor <= top * 1.15:
        top = max(top, floor)
    for x, m, s in zip(xs, means, stds):
        if m == m:
            # sit the value ABOVE the error bar, never on it. Rotated upright
            # because at seven arms the horizontal labels overlap each other.
            ax.text(x, m + s + top * 0.05, f"{m:.3f}" if top < 5 else f"{m:.1f}",
                    ha="center", va="bottom", fontsize=8, color=INK,
                    rotation=90 if len(arms) > 5 else 0)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([ARM_TICK.get(a, a) for a in arms],
                       fontsize=8.5, color=MUTED,
                       rotation=40 if len(arms) > 5 else 0,
                       ha="right" if len(arms) > 5 else "center")
    ax.set_title(title, fontsize=11, color=WARN if disqualified else INK,
                 pad=6, loc="left")
    # rotated value labels stand upright and need roughly twice the headroom
    headroom = (1.40 if floor is not None else 1.32) * (1.30 if len(arms) > 5 else 1.0)
    style_axis(ax, ymax=top * headroom)

    # the image-free control
    if floor is not None:
        if floor <= ax.get_ylim()[1]:
            # The line is identified once, in the figure legend. Repeating
            # "control 0.374" inside every panel duplicated the legend and, with
            # seven bars, there was nowhere to put it that did not sit on top of
            # one. The values are in the caption.
            ax.axhline(floor, color=FLOOR_C, lw=1.4, ls="--", zorder=4)
        else:
            # off scale: note goes top-LEFT and pushes the direction marker down,
            # since both bottom corners are occupied by tall bars
            off_scale = True
            ax.text(0.02, 0.96, f"control {floor:g} (off scale)",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=8, color=FLOOR_C, style="italic")
    if extra_line is not None:
        y, lab = extra_line
        # likewise identified in the legend, not labelled in the panel
        ax.axhline(y, color=WARN, lw=1.4, ls=":", zorder=4)

    ax.text(0.98, 0.87 if off_scale else 0.96,
            "higher is better" if higher_better else "lower is better",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=MUTED, style="italic")
    if disqualified:
        # a marker, not a shout: the reason this metric cannot support a claim
        # is argued in the caption and in Section 7.1.1
        # above the control line, where this panel has empty space; lower down it
        # sat on the bars, and at mid-height on the line itself
        ax.text(0.5, 0.86, "control exceeds every arm", transform=ax.transAxes,
                ha="center", va="center", fontsize=8.5, color=FLOOR_C,
                style="italic", zorder=5)


def load_ablation(path: Path):
    """Rows from the hyper-parameter ablation, grouped by setting.

    Returns [] when the file is absent so the figure still builds on a tree that
    has not run the ablation.
    """
    if not path.exists():
        return []
    groups: dict[tuple, list] = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = (round(r.get("reg_weight", 0.1), 4),
             round(r.get("backward_bias", 8.0), 4),
             round(r.get("gate_init", 2.0), 4))
        groups.setdefault(k, []).append(r)
    return sorted(groups.items())


def ablation_panel(ax, groups):
    """init -> learned gate, one dumbbell per setting. The point is the length."""
    import math

    LAB = {
        (0.1, 8.0, 2.0):   "default",
        # raw strings: "\l" is not a Python escape, so an unraw literal raises a
        # SyntaxWarning and would become a hard error in a later Python
        (0.0, 8.0, 2.0):   r"$\lambda$ = 0 (no reg.)",
        (1.0, 8.0, 2.0):   r"$\lambda$ = 1",
        (0.1, 8.0, 0.0):   "gate init half open",
        (0.1, 8.0, -2.0):  "gate init OPEN",
        (0.1, 2.0, 2.0):   "B = 2 (weak prior)",
        (0.1, 20.0, 2.0):  "B = 20 (strong prior)",
    }
    ORDER = [(0.1, 8.0, 2.0), (0.0, 8.0, 2.0), (1.0, 8.0, 2.0),
             (0.1, 2.0, 2.0), (0.1, 20.0, 2.0),
             (0.1, 8.0, 0.0), (0.1, 8.0, -2.0)]
    d = dict(groups)
    rows = [(k, d[k]) for k in ORDER if k in d]
    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "ablation not run", ha="center", va="center",
                fontsize=11, color=MUTED, style="italic")
        return

    for i, (k, rs) in enumerate(rows):
        y = len(rows) - 1 - i
        init = 1.0 / (1.0 + math.exp(-k[2]))
        learned = statistics.mean(r["mean_backward_gate"] for r in rs
                                  if r.get("mean_backward_gate") is not None)
        f150 = statistics.mean(r["f1@50"] for r in rs
                               if r.get("f1@50") is not None)
        c = ARM_COLOR["gated"]
        ax.plot([init, learned], [y, y], "-", color=c, lw=3, zorder=2,
                solid_capstyle="round")
        ax.plot(init, y, "o", ms=8, mfc="white", mec=c, mew=1.8, zorder=3)
        ax.plot(learned, y, "o", ms=8, color=c, zorder=3)
        # fixed annotation columns so nothing collides with a dumbbell
        ax.text(1.10, y, f"{learned - init:+.3f}", va="center", ha="center",
                fontsize=9, color=INK)
        ax.text(1.30, y, f"{f150:.3f}", va="center", ha="center", fontsize=9,
                color=FLOOR_C if f150 < FLOOR["f1@50"] else INK)

    ax.axvline(1.0, color=GRID, lw=1.0)
    ax.text(1.10, len(rows) - 0.45, "moved", ha="center", fontsize=9,
            color=MUTED, style="italic")
    ax.text(1.30, len(rows) - 0.45, "F1@50", ha="center", fontsize=9,
            color=MUTED, style="italic")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LAB.get(k, str(k)) for k, _ in rows][::-1], fontsize=9)
    ax.set_xlim(0, 1.42)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("mean backward gate", color=INK, fontsize=10)
    ax.set_title("(D) Mean backward gate before and after training "
                 "(hollow: initialisation, filled: learned)",
                 fontsize=11, color=INK, loc="left", pad=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7); ax.set_axisbelow(True)
    # the reading of this panel belongs in the caption, not printed into the image


def load_timebase(seq: str = "seq0010"):
    """(segments, summary) for the one sequence with a readable time base.

    Read from the anchors, never hard-coded here: the numbers in the paper, the
    overview figure and this panel all have to come from the same file or they
    will drift apart. Returns None when the timeline is absent so the figure
    still builds on a tree without it.
    """
    try:
        from schedule_linkage import load_timeline, uniformity_report
        anchors = load_timeline(seq)
    except Exception:
        return None
    return uniformity_report(anchors)


def timebase_panel(ax, u, onset_mae=4.4):
    """What one frame is worth in days, segment by segment."""
    linear = u["linear_days_per_frame"]
    for (f0, f1), r in u["segments"]:
        stalled = (r == 0.0)
        # a stalled segment has zero height and would be invisible; draw a stub
        # so "no time passed here" reads as a mark rather than as missing data
        ax.bar((f0 + f1) / 2, max(r, 0.16), width=(f1 - f0) * 0.9,
               color=FLOOR_C if stalled else ARM_COLOR["hard"],
               edgecolor="none", zorder=2)

    ax.axhline(linear, color=WARN, lw=1.6, ls="--", zorder=3)
    ax.text(u["segments"][0][0][0] + 1, linear + 0.75,
            f"linear assumption: {linear:.2f} days per frame",
            fontsize=8.5, color=WARN, style="italic", va="bottom", zorder=4)

    ax.set_xlabel("frame index in seq0010", color=INK, fontsize=10)
    ax.set_title("(E) Elapsed days per frame, seq0010",
                 fontsize=11, color=INK, loc="left", pad=8)
    style_axis(ax, ylabel="elapsed days per frame",
               ymax=u["max_days_per_frame"] * 1.30)
    ax.text(0.985, 0.955, f"red: {u['stalled_segments']} segments with no "
            f"elapsed time",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color=FLOOR_C)
    # the interpretation -- what a 4.4-frame error is worth in days -- is in the
    # caption and in Section 7.6, not printed into the image


def paired_panel(ax, by_arm, key="f1@50", label="segmental F1@50"):
    """hard vs gated, one line per held-out site (mean over seeds)."""
    if not {"hard", "gated"} <= set(by_arm):
        ax.axis("off")
        ax.text(0.5, 0.5, "decisive panel needs both `hard` and `gated`\n(not yet in the grid)",
                ha="center", va="center", fontsize=11, color=MUTED, style="italic")
        return

    def per_site(arm):
        d = {}
        for r in by_arm[arm]:
            v = r.get(key)
            if v is not None and v == v:
                d.setdefault(r["held_out"], []).append(v)
        return {k: statistics.mean(v) for k, v in d.items()}

    h, g = per_site("hard"), per_site("gated")
    sites = sorted(set(h) & set(g))
    if not sites:
        ax.axis("off"); return
    wins = sum(1 for s in sites if g[s] > h[s])
    for s in sites:
        ax.plot([0, 1], [h[s], g[s]], "-", color=GRID, lw=1.2, zorder=1)
        ax.plot(0, h[s], "o", ms=7, color=ARM_COLOR["hard"], zorder=2)
        ax.plot(1, g[s], "o", ms=7, color=ARM_COLOR["gated"], zorder=2)
    ax.plot([0, 1], [statistics.mean(h[s] for s in sites),
                     statistics.mean(g[s] for s in sites)],
            "-", color=INK, lw=2.4, zorder=3, label="mean")
    ax.set_xlim(-0.42, 1.42); ax.set_xticks([0, 1])
    ax.set_xticklabels(["hard monotonic", "gated (learnable prior)"],
                       fontsize=10, color=INK)
    ax.set_title(f"(C) Rigid versus gated, paired by held-out site "
                 f"(gate ahead on {wins}/{len(sites)})",
                 fontsize=11, color=INK, pad=8, loc="left")
    style_axis(ax, ylabel=label)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/loso/results.jsonl")
    # the TAS baselines were run later and into their own file, so that the
    # published grid could not be modified by adding arms to it
    ap.add_argument("--tas", default="experiments/tas/results.jsonl",
                    help="MS-TCN / ASFormer rows; skipped if absent")
    ap.add_argument("--out", default="diagrams/paper-fig2-results")
    ap.add_argument("--expected", type=int, default=273,
                    help="runs across every file the figure draws from "
                         "(195 published + 78 TAS)")
    args = ap.parse_args()

    rows = load(Path(args.results))
    tas_path = Path(args.tas)
    if tas_path.exists():
        rows += load(tas_path)
    if not rows:
        raise SystemExit(f"no rows in {args.results}")
    by_arm: dict[str, list[dict]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)

    has_decisive = {"hard", "gated"} <= set(by_arm)

    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    ablation = load_ablation(Path(args.results).parent.parent / "ablation" / "results.jsonl")
    timebase = load_timebase()

    # Layout. Every panel is a data panel: the explanatory prose that used to
    # occupy the whole (B) row now lives in the LaTeX caption, where a journal
    # can copyedit it and where it does not become unreadable at print size.
    #   row 0  A1..A5   small multiples
    #   row 1  B (2 cols) + C (3 cols)
    #   row 2  D         ablation, full width
    #   row 3  E         time base, full width
    ratios = [1.0]
    if has_decisive:
        ratios.append(1.05)                           # (B) and (C) share a row
        if ablation:
            ratios.append(1.05)                       # (D)
        if timebase:
            ratios.append(0.95)                       # (E)
    else:
        ratios.append(1.0)                            # (B) alone
    rows_needed = len(ratios)
    HEIGHT = {2: 6.2, 3: 9.0, 4: 11.6}
    fig = plt.figure(figsize=(15.0, HEIGHT[rows_needed]))
    # left margin is set by panel (D), whose category labels ("gate init half
    # open") sit OUTSIDE its axes; at 0.055 they were clipped by the figure edge
    gs = fig.add_gridspec(rows_needed, 5, height_ratios=ratios,
                          hspace=0.75, wspace=0.36,
                          left=0.105, right=0.98, top=0.955, bottom=0.10)

    # No figure title and no run-count banner: the caption carries both, and a
    # title printed inside the image duplicates it in every rendering. The one
    # exception is a PARTIAL grid -- that is a warning about the data rather
    # than a description of it, and it must be impossible to miss.
    complete = len(rows)
    if complete < args.expected:
        fig.text(0.055, 0.995,
                 f"PARTIAL — {complete} of {args.expected} runs; arms present: "
                 f"{', '.join(a for a in ARM_ORDER if a in by_arm)}",
                 fontsize=11, color=WARN, style="italic", va="top")

    panels = [
        ("acc",              "(A1) Accuracy",        True,  FLOOR["acc"],      False, None),
        ("macro_f1",         "(A2) Macro-F1",        True,  FLOOR["macro_f1"], False, None),
        ("edit",             "(A3) Edit score",      True,  FLOOR["edit"],     True,  None),
        ("f1@50",            "(A4) F1@50",           True,  FLOOR["f1@50"],    False, None),
        ("onset_mae_frames", "(A5) Stage-onset MAE", False, FLOOR["onset_mae_frames"], False,
         (ANNOTATION_NOISE_FRAMES, "annotator disagreement")),
    ]
    for i, (key, title, hb, floor, dq, extra) in enumerate(panels):
        bar_panel(fig.add_subplot(gs[0, i]), by_arm, key, title,
                  higher_better=hb, floor=floor, disqualified=dq, extra_line=extra)

    bar_panel(fig.add_subplot(gs[1, 0:2]), by_arm, "violation",
              "(B) Order-violation rate", higher_better=False)

    if has_decisive:
        paired_panel(fig.add_subplot(gs[1, 2:5]), by_arm)
        row = 1
        if ablation:
            row += 1
            ablation_panel(fig.add_subplot(gs[row, 0:5]), ablation)
        if timebase:
            row += 1
            onset = agg([r.get("onset_mae_frames") for r in by_arm["hard"]])[0]
            timebase_panel(fig.add_subplot(gs[row, 0:5]), timebase,
                           onset_mae=round(onset, 1))

    handles = [Patch(facecolor=ARM_COLOR[a], label=ARM_LABEL[a])
               for a in ARM_ORDER if a in by_arm]
    handles += [Line2D([0], [0], color=FLOOR_C, ls="--", lw=1.4, label="image-free control"),
                Line2D([0], [0], color=WARN, ls=":", lw=1.4, label="annotation noise")]
    # nine entries in one row runs off both edges of the figure; wrap instead
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 5), frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.002), labelcolor=INK)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + ext), dpi=200, facecolor="white")
    state = ("COMPLETE" if complete >= args.expected
             else f"PARTIAL {complete}/{args.expected}")
    print(f"wrote {out}.png and {out}.pdf   [{state} — {complete} runs]")


if __name__ == "__main__":
    main()
