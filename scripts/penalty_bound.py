"""What score bound makes a FINITE backward penalty equivalent to a mask?
— IJIES reviewer 1, point 6.

THE OBJECTION. The paper calls the rigid arm's output "monotone by construction"
while the arm does not mask backward transitions at all: it subtracts a finite
penalty of HARD_BIAS = 10,000 from every backward entry of the transition matrix.
The reviewer asks us either to mask exactly or to state the score bound under
which a finite penalty guarantees monotonicity. It is the second: masking would
change the published numbers, and the bound is both checkable and informative.

THE ARGUMENT. Viterbi maximises

    S(y) = start[y_0] + em[0,y_0] + sum over t>0 of ( trans_eff[y_t-1,y_t] + em[t,y_t] )
           + end[y_T],

and in the rigid arm trans_eff = trans_free - P on backward entries (P = 10,000).
Let

    V_free  = max over ALL paths of S computed with trans_free (no penalty),
    V_mono  = max over MONOTONE paths of the same S.

Every path containing at least one backward step pays the penalty at least once,
so its penalised score is at most V_free - P. The decoded path is therefore
monotone whenever

    P  >  V_free - V_mono                                                  (*)

and the right-hand side is exactly computable by two Viterbi passes: one over the
free matrix, one with backward transitions masked out. This script reports the
margin (*) for every sequence in the corpus, so the guarantee rests on a measured
quantity rather than on the word "prohibitive".

A reader who wants the bound without re-running anything can use the closed form
it implies: a single backward step can buy at most one per-frame emission spread
per frame plus one transition range per step, so

    V_free - V_mono  <=  sum over t of ( max_k em[t,k] - min_k em[t,k] )
                         + (T-1) * ( max_ij trans_free - min_ij trans_free ),

which is also printed, together with the per-frame spread that drives it.

Emissions come from a trained fold of the rigid arm itself -- the arm whose
monotonicity is claimed -- and the bound is evaluated on all 13 sequences, not
only the held-out one.

    python src/experiments/penalty_bound.py                       # default fold
    python src/experiments/penalty_bound.py --held-out seq0010
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from data.sequence_dataset import SequenceFrameDataset, list_sequence_dirs, pad_collate  # noqa: E402
from models.sequence_constraint import HARD_BIAS                                          # noqa: E402
from train_sequence import build_model, resolve_device, run_fold                          # noqa: E402

NEG = -1e9  # stands in for -inf; keeps the argmax finite and comparable


def viterbi(em: torch.Tensor, trans: torch.Tensor,
            start: torch.Tensor, end: torch.Tensor) -> float:
    """Best path score for one sequence. em: (T,K), trans: (K,K)."""
    T, K = em.shape
    delta = start + em[0]
    for t in range(1, T):
        delta = (delta.view(K, 1) + trans + em[t].view(1, K)).max(dim=0).values
    return float((delta + end).max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="seq_data")
    ap.add_argument("--held-out", default="seq0003")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--json", default="experiments/penalty_bound.json")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.cpu)
    dirs = list_sequence_dirs(args.data)
    train_dirs = [d for d in dirs if d.name != args.held_out]
    eval_dirs = [d for d in dirs if d.name == args.held_out]
    if not eval_dirs:
        sys.exit(f"held-out site {args.held_out!r} not in {args.data}")

    cfg = {"mode": "hard", "temporal": "none", "num_stages": 5,
           "epochs": args.epochs, "batch_size": 1, "img_size": 224, "max_len": 64,
           "backbone_lr": 1e-4, "head_lr": 1e-3, "reg_weight": 0.1,
           "seed": args.seed}

    print(f"training the RIGID arm, fold {args.held_out} seed {args.seed}, on {device}\n")
    res = run_fold(train_dirs, eval_dirs, cfg, device, verbose=False)
    model = build_model(cfg, device)
    model.load_state_dict(res["state_dict"])
    model.eval()
    crf = model.crf

    # The free matrix is what Viterbi would see with the penalty removed; the
    # rigid arm's own effective matrix is trans_free - HARD_BIAS on backward.
    trans_free = crf.trans_free.detach().cpu()
    start = crf.start.detach().cpu()
    end = crf.end.detach().cpu()
    bw = crf.backward_mask.bool().cpu()
    trans_mono = trans_free.clone()
    trans_mono[bw] = NEG

    from torch.utils.data import DataLoader
    ds = SequenceFrameDataset(sequence_dirs=dirs, img_size=224, max_len=64)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=pad_collate)

    tr_range = float(trans_free.max() - trans_free.min())
    print(f"backward penalty in the rigid arm P = {HARD_BIAS:.0f}")
    print(f"free transition range = {tr_range:.3f}\n")
    print(f"{'#':>3s} {'T':>4s} {'V_free':>10s} {'V_mono':>10s} {'margin':>10s} "
          f"{'closed form':>12s} {'max spread':>11s}")

    rows = []
    for n, (frames, labels, mask) in enumerate(dl):
        frames, mask = frames.to(device), mask.to(device)
        with torch.no_grad():
            em = model.emissions(frames, mask)[0].cpu()
        T = int(mask[0].sum().item())
        em = em[:T]
        v_free = viterbi(em, trans_free, start, end)
        v_mono = viterbi(em, trans_mono, start, end)
        spread = (em.max(dim=1).values - em.min(dim=1).values)
        closed = float(spread.sum()) + (T - 1) * tr_range
        rows.append({"index": n, "T": T, "v_free": v_free, "v_mono": v_mono,
                     "margin": v_free - v_mono, "closed_form_bound": closed,
                     "max_frame_spread": float(spread.max()),
                     "em_min": float(em.min()), "em_max": float(em.max())})
        print(f"{n:3d} {T:4d} {v_free:10.3f} {v_mono:10.3f} {v_free - v_mono:10.3f} "
              f"{closed:12.1f} {float(spread.max()):11.3f}")

    worst = max(r["margin"] for r in rows)
    worst_closed = max(r["closed_form_bound"] for r in rows)
    max_spread = max(r["max_frame_spread"] for r in rows)
    max_T = max(r["T"] for r in rows)
    em_lo = min(r["em_min"] for r in rows)
    em_hi = max(r["em_max"] for r in rows)

    print(f"\n{len(rows)} sequences, longest T = {max_T}")
    print(f"emission range over the corpus        : [{em_lo:.2f}, {em_hi:.2f}]")
    print(f"largest per-frame emission spread     : {max_spread:.2f}")
    print(f"WORST required penalty (V_free-V_mono): {worst:.3f}")
    print(f"closed-form upper bound on that       : {worst_closed:.1f}")
    print(f"rigid arm's penalty                   : {HARD_BIAS:.0f}")
    ok = HARD_BIAS > worst
    ok_closed = HARD_BIAS > worst_closed
    print(f"\nguaranteed monotone on this corpus (measured margin): {ok}")
    print(f"guaranteed monotone by the closed form alone        : {ok_closed}")
    print("\nThe second is the stronger statement: it holds for any emissions")
    print("within the measured spread, not only the ones scored here.")

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"held_out": args.held_out, "seed": args.seed, "penalty": HARD_BIAS,
         "transition_range": tr_range, "max_T": max_T,
         "emission_min": em_lo, "emission_max": em_hi,
         "max_frame_spread": max_spread,
         "worst_required_penalty": worst,
         "worst_closed_form_bound": worst_closed,
         "guaranteed_measured": ok, "guaranteed_closed_form": ok_closed,
         "sequences": rows}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
