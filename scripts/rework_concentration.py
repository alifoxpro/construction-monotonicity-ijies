"""
Concentrated vs scattered re-work — does the gate learn SELECTIVELY?

Why this experiment
-------------------
Section 3 claims the gate lets data "re-open one specific transition without
relaxing the rest". Nothing has tested that. The main sweep
(`rework_sweep.py`) scatters re-work across random stage pairs, so it measures
only the aggregate effect of a rising violation rate.

The distinction matters for data collection. A demolition-then-rebuild sequence
contributes ONE large, always-identical backward step (finished -> site work).
Across a corpus that is a very low overall violation rate --- roughly 0.7% within
such a sequence and far less corpus-wide --- which the main sweep says is well
below the ~2% break-even. But that reading assumes the evidence is spread thin
across many transitions. If instead the gate can learn from evidence CONCENTRATED
on a single transition, a handful of demo-rebuild sequences may suffice even at a
low aggregate rate.

Two regimes, matched on events per sequence:

  scattered    : each re-work event drops from a random stage (as in the main sweep)
  concentrated : every re-work event is the SAME transition (FIXED_FROM -> FIXED_FROM-1)

Reported per regime: accuracy and edit, plus the measurement that answers the
selectivity question --- the gate value on the *exercised* transition against the
mean of all *unexercised* backward gates. If the mechanism works as designed the
first should fall and the second should not.

    python src/experiments/rework_concentration.py --seeds 5
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.sequence_constraint import (                       # noqa: E402
    ConstructionSequenceConstraint, order_violation_rate,
)
from metrics import edit_score                                 # noqa: E402

FIXED_FROM = 3          # concentrated regime always exercises 3 -> 2
MODES = ("hard", "gated")


def make_dataset(n, K, T, noise, strength, n_events, rework_len, concentrated):
    labels = torch.zeros(n, T, dtype=torch.long)
    for i in range(n):
        cps = torch.sort(torch.randint(1, T, (K - 1,))).values
        lab, s = [], 0
        for t in range(T):
            while s < K - 1 and t >= cps[s]:
                s += 1
            lab.append(s)
        for _ in range(n_events):
            if concentrated:
                # only where the sequence is AT the fixed stage -> always 3->2
                cands = [t for t in range(T - rework_len) if lab[t] == FIXED_FROM]
            else:
                cands = [t for t in range(T - rework_len) if lab[t] > 0]
            if not cands:
                break
            t0 = cands[int(torch.randint(len(cands), ()))]
            base = lab[t0] - 1
            for t in range(t0, min(t0 + rework_len, T)):
                lab[t] = base
        labels[i] = torch.tensor(lab)
    onehot = torch.nn.functional.one_hot(labels, K).float() * strength
    return onehot + noise * torch.randn(n, T, K), labels


def gate_split(model, K):
    """(gate on the exercised transition, mean gate on the others)."""
    import torch as th
    g = th.sigmoid(model.gate_logit)
    idx = th.arange(K)
    backward = (idx.view(K, 1) > idx.view(1, K))
    target = th.zeros_like(backward)
    target[FIXED_FROM, FIXED_FROM - 1] = True
    others = backward & ~target
    return float(g[target].mean()), float(g[others].mean())


def run_one(mode, n_events, seed, concentrated, cfg):
    torch.manual_seed(seed)
    trX, trY = make_dataset(cfg["n_train"], cfg["K"], cfg["T"], cfg["noise"],
                            cfg["strength"], n_events, cfg["rework_len"], concentrated)
    torch.manual_seed(seed + 10_000)
    teX, teY = make_dataset(cfg["n_test"], cfg["K"], cfg["T"], cfg["noise"],
                            cfg["strength"], n_events, cfg["rework_len"], concentrated)
    torch.manual_seed(seed)
    model = ConstructionSequenceConstraint(cfg["K"], mode=mode)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    for _ in range(cfg["steps"]):
        opt.zero_grad()
        model.nll(trX, trY, reg_weight=cfg["reg_weight"]).backward()
        opt.step()

    paths = model.decode(teX)
    golds = teY.tolist()
    acc = st.mean(sum(p == g for p, g in zip(pp, gg)) / len(gg)
                  for pp, gg in zip(paths, golds))
    tgt, oth = gate_split(model, cfg["K"])
    return {"mode": mode, "regime": "concentrated" if concentrated else "scattered",
            "n_events": n_events, "seed": seed, "acc": acc,
            "edit": edit_score(paths, golds),
            "gold_violation": order_violation_rate(golds),
            "gate_exercised": tgt, "gate_others": oth}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="0,1,2,4")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--out", default="experiments/rework_concentration")
    args = ap.parse_args()

    cfg = dict(K=5, T=64, n_train=192, n_test=192, noise=2.4, strength=2.0,
               rework_len=4, lr=0.05, steps=args.steps, reg_weight=0.1)
    events = [int(e) for e in args.events.split(",")]

    rows = []
    for conc in (False, True):
        for ne in events:
            for seed in range(args.seeds):
                for mode in MODES:
                    rows.append(run_one(mode, ne, seed, conc, cfg))
            done = [r for r in rows if r["regime"] == ("concentrated" if conc else "scattered")
                    and r["n_events"] == ne]
            gv = st.mean(r["gold_violation"] for r in done)
            g = [r for r in done if r["mode"] == "gated"]
            h = [r for r in done if r["mode"] == "hard"]
            print(f"{'concentrated' if conc else 'scattered  '} ev={ne} "
                  f"viol={gv:.4f} | hard acc {st.mean(r['acc'] for r in h):.3f} "
                  f"| gated acc {st.mean(r['acc'] for r in g):.3f} "
                  f"| gate exercised {st.mean(r['gate_exercised'] for r in g):.3f} "
                  f"others {st.mean(r['gate_others'] for r in g):.3f}", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump({"config": cfg, "fixed_from": FIXED_FROM, "rows": rows}, f, indent=1)
    print(f"\nrows -> {out/'results.json'}")


if __name__ == "__main__":
    main()
