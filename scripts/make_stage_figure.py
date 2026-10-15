"""
Stage-taxonomy figure — real example frames for every construction stage, with
the measured ambiguity of each boundary drawn between the rows.

The paper defines a five-stage taxonomy in a table but never shows it. A reader
outside construction has no mental image of what "envelope" or "structure" looks
like from a fixed camera, and cannot judge whether the classes are separable or
whether the boundaries are as ambiguous as the limitations section claims.

This builds that figure from the corpus itself: one row per stage, several
examples per row, each drawn from a DIFFERENT site so the row shows intra-class
variety rather than three frames of the same building. Frames are chosen from the
middle of each stage's run within a sequence, avoiding transition frames whose
label is least certain.

Between the rows it then answers the question the images raise. The second
annotator pass (src/data_collection/annotator_b_agreement.py) labelled the whole
corpus independently, and every disagreement it produced was between ADJACENT
stages -- never a jump of two. So each disagreement count belongs to exactly one
boundary and can be drawn in the gap between the two rows it separates. The
figure then states, rather than merely illustrating, which boundaries the
taxonomy fails to pin down.

    python src/experiments/make_stage_figure.py
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.sequence_dataset import list_sequence_dirs        # noqa: E402

STAGE_NAMES = {
    0: "0  Site work",
    1: "1  Foundation",
    2: "2  Structure",
    3: "3  Envelope / walls",
    4: "4  Finishing",
}

RAW = Path("seq_data_raw")


def read_labels(path: Path) -> list[int]:
    rows = csv.reader(open(path, newline="", encoding="utf-8"))
    return [int(r[1]) for r in rows if r]


def boundary_disagreements() -> tuple[dict[int, int], dict[int, int], int, int]:
    """(counts per boundary s|s+1, frames per stage, compared frames, total disagreements).

    Returns empty counts if the second-annotator labels are not present, so the
    figure still builds on a corpus that has not been double-labelled.
    """
    pair: dict[int, int] = defaultdict(int)
    per_stage: dict[int, int] = defaultdict(int)
    compared = disagreements = 0
    for seq in sorted(RAW.iterdir()) if RAW.is_dir() else []:
        a_path, b_path = seq / "labels.csv", seq / "labels.B.csv"
        if not (a_path.exists() and b_path.exists()):
            continue
        a, b = read_labels(a_path), read_labels(b_path)
        for x in a:                       # corpus counts: annotator A, all frames
            if x >= 0:
                per_stage[x] += 1
        for x, y in zip(a, b):
            if x < 0 or y < 0:
                continue
            compared += 1
            if x != y:
                disagreements += 1
                if abs(x - y) == 1:          # every observed disagreement is adjacent
                    pair[min(x, y)] += 1
    return dict(pair), dict(per_stage), compared, disagreements


def load_runs(seq_dir: Path):
    """Contiguous runs of one stage: {stage: [(path, ...), ...]} for this site."""
    csv_path = seq_dir / "labels.csv"
    if not csv_path.exists():
        return {}
    frames = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().lower() in {"filename", "file", ""}:
                continue
            stage = int(row[1])
            if stage < 0:
                continue
            frames.append((seq_dir / row[0].strip(), stage))
    runs = defaultdict(list)
    i = 0
    while i < len(frames):
        j = i
        while j + 1 < len(frames) and frames[j + 1][1] == frames[i][1]:
            j += 1
        # take the MIDDLE of the run: transition frames are the least certain
        if j - i >= 2:
            runs[frames[i][1]].append(frames[(i + j) // 2][0])
        i = j + 1
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="seq_data")
    ap.add_argument("--cols", type=int, default=4, help="examples per stage")
    ap.add_argument("--out", default="diagrams/paper-fig0-stages")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    per_site = {}
    for seq_dir in list_sequence_dirs(args.data):
        runs = load_runs(seq_dir)
        if runs:
            per_site[seq_dir.name] = runs

    chosen = {}
    for stage in sorted(STAGE_NAMES):
        sites = sorted(s for s, r in per_site.items() if r.get(stage))
        random.shuffle(sites)
        picks = []
        for site in sites:                      # one frame per site, for variety
            cands = per_site[site][stage]
            picks.append((site, cands[len(cands) // 2]))
            if len(picks) == args.cols:
                break
        chosen[stage] = picks
        print(f"stage {stage}: {len(picks)} example(s) from {[s for s, _ in picks]}")

    pair, per_stage, compared, total = boundary_disagreements()
    if pair:
        print(f"\nsecond-annotator disagreements: {total} of {compared} compared frames")
        for s in sorted(pair):
            print(f"  boundary {s}|{s+1}: {pair[s]:3d}")
    else:
        print("\nno labels.B.csv found — boundary annotations omitted")

    render(chosen, args.cols, Path(args.out), pair, per_stage, compared, total)


def render(chosen, cols, out_path, pair, per_stage, compared, total):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from PIL import Image

    INK, MUTED, WARN = "#1a1a1a", "#5a5a5a", "#8f3a37"
    plt.rcParams.update({"font.family": "serif", "figure.facecolor": "white"})
    rows = len(chosen)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.42 * rows))
    if rows == 1:
        axes = [axes]

    for r, stage in enumerate(sorted(chosen)):
        picks = chosen[stage]
        for c in range(cols):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#cccccc")
            if c < len(picks):
                site, path = picks[c]
                try:
                    im = Image.open(path).convert("RGB")
                    ax.imshow(im)
                    ax.set_xlabel(site, fontsize=7.5, color=MUTED, labelpad=2)
                except Exception as exc:            # missing frame -> say so
                    ax.text(0.5, 0.5, f"unavailable\n{exc.__class__.__name__}",
                            ha="center", va="center", fontsize=7, color=MUTED,
                            transform=ax.transAxes)
            else:
                ax.axis("off")
            if c == 0:
                lab = STAGE_NAMES[stage]
                if per_stage.get(stage):
                    lab += f"\n{per_stage[stage]:,} frames"
                ax.set_ylabel(lab, fontsize=11.5, color=INK,
                              rotation=0, ha="right", va="center", labelpad=14)

    # No suptitle and no explanatory note: the LaTeX caption carries both, where
    # a journal can copyedit them. The note also asserted that the second
    # annotator worked "blind", which contradicts Section 5.4 and the limitations
    # -- annotator B had read the manuscript and knew the expected result. An
    # error printed into a raster is one nobody can correct at proof stage.
    fig.tight_layout(rect=(0.11, 0.028, 1, 0.99))
    # widen the gap between rows so the boundary annotations have somewhere to sit
    fig.subplots_adjust(hspace=0.34)

    # ---- the measured ambiguity of each boundary, drawn where it belongs -----
    if pair:
        stages = sorted(chosen)
        for r in range(len(stages) - 1):
            s = stages[r]
            n = pair.get(s, 0)
            top = axes[r][0].get_position()
            bot = axes[r + 1][0].get_position()
            y = (top.y0 + bot.y1) / 2
            x0 = axes[r][0].get_position().x0
            x1 = axes[r][cols - 1].get_position().x1
            heavy = n >= 50
            fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure,
                                  color=WARN if heavy else "#bbbbbb",
                                  lw=1.6 if heavy else 1.0,
                                  ls="--", alpha=0.95 if heavy else 0.8, zorder=5))
            fig.text(x1, y, f"  {n} frames disagree  ",
                     ha="right", va="center", fontsize=9.5,
                     color=WARN if heavy else MUTED,
                     style="italic", fontweight="bold" if heavy else "normal",
                     bbox=dict(facecolor="white", edgecolor="none", pad=1.6), zorder=6)
            fig.text(x0, y, f"  boundary {s} | {s + 1}",
                     ha="left", va="center", fontsize=9, color=MUTED,
                     bbox=dict(facecolor="white", edgecolor="none", pad=1.6), zorder=6)

        print(f"  caption inputs: {total} disagreements of {compared:,} compared; "
              f"the two boundaries either side of Envelope carry "
              f"{pair.get(2, 0) + pair.get(3, 0)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_path.with_suffix("." + ext), dpi=170, facecolor="white",
                    bbox_inches="tight")
    print(f"figure -> {out_path}.png / .pdf")


if __name__ == "__main__":
    main()
