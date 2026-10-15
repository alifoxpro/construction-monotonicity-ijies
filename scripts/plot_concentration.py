"""Figure for the concentrated-vs-scattered re-work experiment (paper 6.4).

Two panels, because the experiment answers two separate questions:

  (A) does concentration change WHERE the break-even sits? -- gated minus hard
      accuracy against the measured ground-truth violation rate, one line per
      regime. Concentration should shift the crossing left.

  (B) is the gate SELECTIVE? -- the gate on the exercised transition against the
      mean of every unexercised backward gate. If the mechanism works as designed
      the first falls and the second does not. This is the panel that tests the
      claim in the method section, which nothing else in the paper does.

    python src/experiments/plot_concentration.py
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

REGIME_COLOR = {"scattered": "#3A6EA5", "concentrated": "#B3541E"}
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d8d8d8"

# First violation rate at which the gate's advantage is SIGNIFICANT under each
# regime (paired tests, paper 6.4). A curve crossing zero is not a significant
# difference: the concentrated line crosses at ~0.012 but does not become
# significant until 0.0155, and reading the crossing off the plot instead of the
# test is how the discarded "~2%" break-even claim was produced.
SIG_FROM = {"scattered": 0.0485, "concentrated": 0.0155}


def agg(rows, regime, ne, mode, key):
    v = [r[key] for r in rows
         if r["regime"] == regime and r["n_events"] == ne and r["mode"] == mode]
    if not v:
        return float("nan"), 0.0
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/rework_concentration/results.json")
    ap.add_argument("--out", default="diagrams/paper-fig4-concentration")
    args = ap.parse_args()

    d = json.load(open(args.results))
    rows, fixed_from = d["rows"], d["fixed_from"]
    events = sorted({r["n_events"] for r in rows})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))

    def style(ax, xlabel, ylabel, title):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=3)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7); ax.set_axisbelow(True)
        ax.set_xlabel(xlabel, color=INK, fontsize=10)
        ax.set_ylabel(ylabel, color=INK, fontsize=10)
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)

    # ---- (A) break-even against the MEASURED violation rate -----------------
    for regime in ("scattered", "concentrated"):
        xs, ys = [], []
        for ne in events:
            gv, _ = agg(rows, regime, ne, "gated", "gold_violation")
            g, _ = agg(rows, regime, ne, "gated", "acc")
            h, _ = agg(rows, regime, ne, "hard", "acc")
            xs.append(gv); ys.append(g - h)
        axes[0].plot(xs, ys, "-o", ms=6, lw=2, color=REGIME_COLOR[regime],
                     label=regime)
    axes[0].axhline(0, color=MUTED, lw=1.2, ls=":")
    axes[0].text(0.985, 0.40, "gated better", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=9, color=MUTED, style="italic")
    # zero crossing is not significance -- mark where the tests actually turn
    for regime, x in SIG_FROM.items():
        c = REGIME_COLOR[regime]
        axes[0].axvline(x, color=c, lw=1.3, ls="--", alpha=0.75, zorder=1)
        ha = "left" if regime == "concentrated" else "right"
        axes[0].annotate(f"significant\nfrom {x:.4f}", xy=(x, 0.02),
                         xycoords=("data", "axes fraction"), ha=ha, va="bottom",
                         fontsize=8, color=c, style="italic")
    axes[0].axvline(0.0, color="#8f3a37", lw=1.6, ls="--", alpha=0.9, zorder=1)
    axes[0].annotate("real corpus", xy=(0.0006, 0.30),
                     xycoords=("data", "axes fraction"), fontsize=8.5,
                     color="#8f3a37", va="center")
    style(axes[0], "measured ground-truth order-violation rate",
          "accuracy: gated $-$ hard", "(A) Gated minus rigid against violation rate")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")

    # ---- (B) selectivity ----------------------------------------------------
    for regime in ("scattered", "concentrated"):
        ex, ot = [], []
        for ne in events:
            e, _ = agg(rows, regime, ne, "gated", "gate_exercised")
            o, _ = agg(rows, regime, ne, "gated", "gate_others")
            ex.append(e); ot.append(o)
        c = REGIME_COLOR[regime]
        axes[1].plot(events, ex, "-o", ms=6, lw=2.2, color=c,
                     label=f"{regime}: exercised transition")
        axes[1].plot(events, ot, "--s", ms=5, lw=1.8, color=c, alpha=0.75,
                     label=f"{regime}: all other backward gates")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(events)
    style(axes[1], "re-work events per sequence", "learned gate value",
          f"(B) Learned gate value, exercised transition "
          f"{fixed_from}$\\rightarrow${fixed_from-1} versus the rest")
    axes[1].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")

    # Title and explanatory paragraph moved to the LaTeX caption; see the note in
    # plot_results.py. Everything the reader needs to decode the panel -- axis
    # labels, legend, the dashed significance rules -- is still drawn.
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + ext), dpi=200, facecolor="white")
    print(f"figure -> {out}.png / .pdf")


if __name__ == "__main__":
    main()
