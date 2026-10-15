"""
Controlled re-work sweep — paper §6.3.

Why this experiment exists
--------------------------
The full LOSO grid (§6.1-6.2) refuted the gate claim on real data: `hard` beat
`gated`, and the learned gates never left their initialization. The measured
reason is that the corpus gold order-violation rate is exactly 0.000 — there is
no backward evidence for the likelihood term to act on.

This sweep is therefore the ONLY remaining evidence for the gate, and its job is
to answer one question precisely: **at what rate of genuine re-work does a
learnable constraint start to beat a rigid one?** Below that rate, `hard` is the
correct choice and the paper should say so.

Design
------
Synthetic sequences over K stages with a controlled fraction of genuine backward
segments, and emissions from a noisy oracle. Only the sequence layer varies; the
"backbone" is held fixed by construction, which is exactly the isolation the real
grid cannot give us.

TRAIN AND TEST ARE SEPARATE SETS. The self-test inside `sequence_constraint.py`
trains and scores on the same batch, which is fine for a smoke test and useless
for a paper number.

Note on `hard` under re-work: its gold likelihood is astronomically low (each gold
backward step costs HARD_BIAS), i.e. the model *cannot* represent the data it is
asked to fit. That is the point of the comparison, not a bug.

    python src/experiments/rework_sweep.py --seeds 5
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.sequence_constraint import (                      # noqa: E402
    ConstructionSequenceConstraint, order_violation_rate,
)
from metrics import edit_score                                # noqa: E402

MODES = ("free", "hard", "gated")
MODE_COLOR = {"argmax": "#3A6EA5", "free": "#5F8F2E", "hard": "#7B5EA7", "gated": "#B3541E"}
MODE_LABEL = {"argmax": "per-frame argmax", "free": "free CRF (no prior)",
              "hard": "hard monotonic", "gated": "gated (learnable prior)"}

# Significance brackets from the paired tests on this sweep (paper 6.3). These are
# the RESULT of the experiment, so the figure marks them rather than leaving the
# reader to eyeball where two curves cross -- a sign change is not a significant
# difference, and reading one off the plot is how the earlier "~2%" claim arose.
SIG = {"rigid_ahead_to": 0.0146, "tie_at": 0.0273, "gate_ahead_from": 0.0485,
       "gate_ahead_concentrated": 0.0155}
# Where the real corpora sit on this axis (paper 4.1 and 4.3).
CORPUS = {"evaluation corpus": 0.0, "after targeted renovation search": 0.00035}


def make_dataset(n, K, T, noise, strength, n_events, rework_len):
    """Stage labels with `n_events` genuine backward segments per sequence.

    Parameterized by EVENTS PER SEQUENCE rather than "fraction of sequences with
    one event": one 4-frame dip in a 64-frame sequence is a gold order-violation
    rate of ~0.003 even when every sequence has one, which is far too weak to
    separate the modes. The sweep reports the resulting measured gold violation
    rate, which is the quantity comparable to the real corpus (0.000).
    """
    labels = torch.zeros(n, T, dtype=torch.long)
    for i in range(n):
        cps = torch.sort(torch.randint(1, T, (K - 1,))).values
        lab, s = [], 0
        for t in range(T):
            while s < K - 1 and t >= cps[s]:
                s += 1
            lab.append(s)
        for _ in range(n_events):
            cands = [t for t in range(T - rework_len) if lab[t] > 0]
            if not cands:
                break
            t0 = cands[int(torch.randint(len(cands), ()))]
            base = lab[t0] - 1                # capture before mutating lab[t0]
            for t in range(t0, min(t0 + rework_len, T)):
                lab[t] = base
        labels[i] = torch.tensor(lab)
    onehot = torch.nn.functional.one_hot(labels, K).float() * strength
    return onehot + noise * torch.randn(n, T, K), labels


def evaluate(paths, gold):
    golds = gold.tolist()
    acc = st.mean(
        sum(p == g for p, g in zip(pp, gg)) / len(gg) for pp, gg in zip(paths, golds)
    )
    return {"acc": acc, "edit": edit_score(paths, golds),
            "violation": order_violation_rate(paths)}


def run_one(mode, rate, seed, cfg, device):
    torch.manual_seed(seed)
    trX, trY = make_dataset(cfg["n_train"], cfg["K"], cfg["T"], cfg["noise"],
                            cfg["strength"], rate, cfg["rework_len"])
    torch.manual_seed(seed + 10_000)          # held-out set, same distribution
    teX, teY = make_dataset(cfg["n_test"], cfg["K"], cfg["T"], cfg["noise"],
                            cfg["strength"], rate, cfg["rework_len"])
    trX, trY, teX, teY = trX.to(device), trY.to(device), teX.to(device), teY.to(device)

    torch.manual_seed(seed)
    model = ConstructionSequenceConstraint(cfg["K"], mode=mode).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    for _ in range(cfg["steps"]):
        opt.zero_grad()
        model.nll(trX, trY, reg_weight=cfg["reg_weight"]).backward()
        opt.step()

    out = evaluate(model.decode(teX), teY)
    gs = model.gate_stats()
    out.update(mode=mode, rework_rate=rate, seed=seed,
               mean_gate=gs["mean_backward_gate"], min_gate=gs["min_backward_gate"],
               gold_violation=order_violation_rate(teY.tolist()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="0,1,2,4,8", help="re-work events per sequence")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--out", default="experiments/rework_sweep")
    ap.add_argument("--fig", default="diagrams/paper-fig3-rework-sweep")
    ap.add_argument("--replot", action="store_true",
                    help="redraw the figure from a saved results.json without "
                         "re-running the sweep (100 training runs)")
    args = ap.parse_args()

    if args.replot:
        saved = json.load(open(Path(args.out) / "results.json"))
        rows = saved["rows"]
        rates = sorted({r["rework_rate"] for r in rows})
        plot(rows, rates, Path(args.fig))
        print(f"replotted from {Path(args.out)/'results.json'} ({len(rows)} rows, no runs)")
        return

    cfg = dict(K=5, T=64, n_train=192, n_test=192, noise=2.4, strength=2.0,
               rework_len=4, lr=0.05, steps=args.steps, reg_weight=0.1)
    rates = [int(r) for r in args.rates.split(",")]
    device = "cpu"        # 5x5 CRF on synthetic emissions; the GPU adds nothing

    rows = []
    for rate in rates:
        for seed in range(args.seeds):
            # per-frame argmax needs no training -- same data, reference line
            torch.manual_seed(seed + 10_000)
            teX, teY = make_dataset(cfg["n_test"], cfg["K"], cfg["T"], cfg["noise"],
                                    cfg["strength"], rate, cfg["rework_len"])
            am = evaluate(teX.argmax(-1).tolist(), teY)
            am.update(mode="argmax", rework_rate=rate, seed=seed,
                      mean_gate=None, min_gate=None,
                      gold_violation=order_violation_rate(teY.tolist()))
            rows.append(am)
            for mode in MODES:
                rows.append(run_one(mode, rate, seed, cfg, device))
        done = [r for r in rows if r["rework_rate"] == rate]
        gv = st.mean(r["gold_violation"] for r in done)
        print(f"events/seq {rate:>2} | gold violations {gv:.4f} | " + "  ".join(
            f"{m}: acc {st.mean(r['acc'] for r in done if r['mode']==m):.3f}"
            for m in ("argmax",) + MODES))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump({"config": cfg, "rows": rows}, f, indent=1)

    plot(rows, rates, Path(args.fig))
    print(f"\nrows -> {out/'results.json'}")


def plot(rows, rates, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d8d8d8"
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    def series(mode, key):
        m, lo, hi = [], [], []
        for r in rates:
            v = [x[key] for x in rows if x["mode"] == mode and x["rework_rate"] == r
                 and x[key] is not None]
            mu = st.mean(v); sd = st.stdev(v) if len(v) > 1 else 0.0
            m.append(mu); lo.append(mu - sd); hi.append(mu + sd)
        return m, lo, hi

    # gold order-violation rate per x tick -- the quantity comparable to the real
    # corpus, which sits at exactly 0.000 (see paper 6.2)
    gold = {r: st.mean(x["gold_violation"] for x in rows if x["rework_rate"] == r)
            for r in rates}

    def x_of(gold_rate, rates, gold):
        """Interpolate an events-per-sequence x for a given gold violation rate."""
        pts = sorted((gold[r], r) for r in rates)
        for (g0, r0), (g1, r1) in zip(pts, pts[1:]):
            if g0 <= gold_rate <= g1:
                f = 0.0 if g1 == g0 else (gold_rate - g0) / (g1 - g0)
                return r0 + f * (r1 - r0)
        return pts[-1][1] if gold_rate > pts[-1][0] else pts[0][1]

    def shade_significance(ax, rates, gold):
        lo = x_of(SIG["rigid_ahead_to"], rates, gold)
        hi = x_of(SIG["gate_ahead_from"], rates, gold)
        ax.axvspan(min(rates), lo, color="#7B5EA7", alpha=0.075, lw=0, zorder=0)
        ax.axvspan(lo, hi, color="#999999", alpha=0.085, lw=0, zorder=0)
        ax.axvspan(hi, max(rates), color="#B3541E", alpha=0.085, lw=0, zorder=0)
        # labels sit on the floor of the axes: the top is where the curves and the
        # legend live, and a band label that overlaps either is worse than none
        # the leftmost band is only ~1 unit wide, far too narrow for a centred
        # label: anchor the outer two to their band edges instead
        for x, txt, col, ha in (
                (min(rates) + 0.06, "rigid\nahead", "#4a3374", "left"),
                ((lo + hi) / 2, "indistinguishable", "#5a5a5a", "center"),
                (max(rates) - 0.06, "gate ahead", "#7a3a12", "right")):
            ax.annotate(txt, xy=(x, 0.012), xycoords=("data", "axes fraction"),
                        ha=ha, va="bottom", fontsize=8.5, color=col,
                        style="italic", fontweight="bold", linespacing=1.15)

    def style(ax, ylabel, title):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=3)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7); ax.set_axisbelow(True)
        ax.set_xlabel("re-work events per sequence", color=INK, fontsize=10)
        ax.set_ylabel(ylabel, color=INK, fontsize=10)
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=26)
        top = ax.secondary_xaxis("top")
        top.set_xticks(rates)
        top.set_xticklabels([f"{gold[r]:.3f}" for r in rates], fontsize=8, color=MUTED)
        top.set_xlabel("resulting gold order-violation rate", fontsize=9, color=MUTED,
                       labelpad=2)
        top.spines["top"].set_color(GRID); top.tick_params(colors=MUTED, length=2)

    for key, ax, ylab, title in ((
            "acc", axes[0], "per-frame accuracy", "(A) Accuracy versus re-work rate"), (
            "edit", axes[1], "segmental edit score", "(B) Edit score versus re-work rate")):
        for mode in ("argmax",) + MODES:
            m, lo, hi = series(mode, key)
            ax.plot(rates, m, "-o", ms=5, lw=2, color=MODE_COLOR[mode], label=MODE_LABEL[mode])
            ax.fill_between(rates, lo, hi, color=MODE_COLOR[mode], alpha=0.12, lw=0)
        style(ax, ylab, title)
        shade_significance(ax, rates, gold)
        ax.axvline(0, color="#8f3a37", lw=1.6, ls="--", alpha=0.9, zorder=2)
        ax.annotate("both real\ncorpora\nsit here",
                    xy=(0.16, 0.42), xycoords=("data", "axes fraction"),
                    fontsize=8.5, color="#8f3a37", va="center")

    # the panel that carries the mechanism: does the gate respond to re-work?
    for key, ls, lab in (("mean_gate", "-", "mean backward gate"),
                         ("min_gate", "--", "lowest backward gate")):
        m, lo, hi = series("gated", key)
        axes[2].plot(rates, m, ls, marker="o", ms=5, lw=2,
                     color=MODE_COLOR["gated"], label=lab)
        axes[2].fill_between(rates, lo, hi, color=MODE_COLOR["gated"], alpha=0.12, lw=0)
    axes[2].axhline(0.8808, color=MUTED, lw=1.2, ls=":", label="initialisation sigma(2.0)")
    style(axes[2], "gate value", "(C) Learned gate value, gated arm")
    axes[2].set_ylim(0, 1.12)
    # the reading of the dotted line -- that on the real corpus the gates never
    # leave it -- is in the caption, not annotated onto the panel

    for ax in axes[:2]:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="center right")
    axes[2].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    # No figure title and no explanatory paragraph: both duplicated the LaTeX
    # caption, which is where a journal can copyedit them and where they stay
    # legible at print size. The shaded bands and axis labels carry the reading.
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fig_path.with_suffix("." + ext), dpi=200, facecolor="white")
    print(f"figure -> {fig_path}.png / .pdf")


if __name__ == "__main__":
    main()
