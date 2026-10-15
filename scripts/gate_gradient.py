"""Where does the gate's gradient come from? — IJIES reviewer 1, point 1.

THE OBJECTION. Section 7.2 explained the gate's immobility by saying the
regulariser holds it at the prior because the corpus contains no backward
transitions for the likelihood to act on. The reviewer points out that this is
wrong as stated: the CRF log-partition sums over ALL paths, including paths that
do step backwards, so the likelihood term contributes a nonzero gradient to a
backward gate even when the observed path never steps back.

THE DERIVATION. With

    psi_eff(i,j) = psi_free(i,j) + sigmoid(theta_ij) * b(i,j),
    b(i,j) = -B for j < i (backward), 0 otherwise,

the negative log-likelihood of the observed path y* is L = log Z - S(y*), and
the standard CRF identity gives the gradient with respect to the transition
potential as expected minus observed counts:

    dL / d psi_eff(i,j) = E_p[n_ij] - n_ij(y*).

The gate sits behind a sigmoid, so the chain rule adds its derivative:

    dL / d theta_ij = ( E_p[n_ij] - n_ij(y*) ) * sigmoid'(theta_ij) * b(i,j)
                    = ( E_p[n_ij] - n_ij(y*) ) * s(1-s) * (-B),   s = sigmoid(theta_ij).

For a BACKWARD transition on this corpus n_ij(y*) = 0, so the bracket reduces to
E_p[n_ij], which is strictly positive: the model always assigns some mass to
backward paths. With B > 0 the product is NEGATIVE, and gradient descent
therefore pushes theta UP.

The regulariser R = mean over backward of (1 - s) has

    d(lambda R) / d theta_ij = -lambda * s(1-s) / n_backward,

also negative, also pushing theta up.

SO THE CORRECTION IS NOT THAT THE PAPER OVERSTATED THE REGULARISER. It is that
BOTH terms push the same way. The gate does not stay high because the likelihood
is silent; it stays high because nothing in the objective pushes it down. That is
a stronger statement than the one in the manuscript, and it is the one the
measurement below tests.

WHAT IS MEASURED. A trained fold is reloaded and two gradients are taken
separately on the training data: the likelihood alone (reg_weight = 0) and the
regulariser alone. Reported per backward transition, with the sigmoid factor, so
the claim rests on numbers rather than on the absence of visible gate movement.

    python src/experiments/gate_gradient.py --held-out seq0003 --seed 0
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
from train_sequence import build_model, resolve_device, run_fold                          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="seq_data")
    ap.add_argument("--held-out", default="seq0003")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--reg-weight", type=float, default=0.1)
    ap.add_argument("--json", default="experiments/gate_gradient.json")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.cpu)
    dirs = list_sequence_dirs(args.data)
    train_dirs = [d for d in dirs if d.name != args.held_out]
    eval_dirs = [d for d in dirs if d.name == args.held_out]
    if not eval_dirs:
        sys.exit(f"held-out site {args.held_out!r} not in {args.data}")

    cfg = {"mode": "gated", "temporal": "none", "num_stages": 5,
           "epochs": args.epochs, "batch_size": 1, "img_size": 224, "max_len": 64,
           "backbone_lr": 1e-4, "head_lr": 1e-3, "reg_weight": args.reg_weight,
           "seed": args.seed}

    print(f"training {args.held_out} seed {args.seed} on {device} to measure its gates\n")
    res = run_fold(train_dirs, eval_dirs, cfg, device, verbose=False)
    model = build_model(cfg, device)
    model.load_state_dict(res["state_dict"])
    model.eval()
    crf = model.crf

    from torch.utils.data import DataLoader
    ds = SequenceFrameDataset(sequence_dirs=train_dirs, img_size=224, max_len=64)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=pad_collate)

    bw = crf.backward_mask.bool()
    g_ll = torch.zeros_like(crf.gate_logit)
    g_reg = torch.zeros_like(crf.gate_logit)

    for frames, labels, mask in dl:
        frames, labels, mask = frames.to(device), labels.to(device), mask.to(device)
        with torch.no_grad():
            em = model.emissions(frames, mask)          # gates are not in the emissions
        em = em.detach().requires_grad_(False)

        crf.zero_grad(set_to_none=True)
        crf.nll(em, labels, mask, reg_weight=0.0).backward()   # LIKELIHOOD ONLY
        g_ll += crf.gate_logit.grad.detach().clone()

        crf.zero_grad(set_to_none=True)
        (args.reg_weight * crf.prior_regularizer()).backward()  # REGULARISER ONLY
        g_reg += crf.gate_logit.grad.detach().clone()

    n = len(ds)
    g_ll /= n
    g_reg /= n
    s = torch.sigmoid(crf.gate_logit)

    print(f"{len(ds)} training sequences; gradients averaged over them")
    print(f"backward penalty B = {crf.backward_bias:.3f}, "
          f"reg weight = {args.reg_weight}\n")
    print(f"{'i->j':>6s} {'gate s':>8s} {'s(1-s)':>8s} {'dL_lik/dθ':>12s} "
          f"{'dL_reg/dθ':>12s} {'total':>12s} {'lik share':>10s}")
    rows = []
    K = crf.num_stages
    for i in range(K):
        for j in range(K):
            if not bw[i, j]:
                continue
            a, b = g_ll[i, j].item(), g_reg[i, j].item()
            tot = a + b
            share = abs(a) / (abs(a) + abs(b)) if (abs(a) + abs(b)) else float("nan")
            print(f"{i}->{j:<3d} {s[i,j].item():8.4f} {(s[i,j]*(1-s[i,j])).item():8.4f} "
                  f"{a:12.3e} {b:12.3e} {tot:12.3e} {share:10.3f}")
            rows.append({"from": i, "to": j, "gate": s[i, j].item(),
                         "grad_likelihood": a, "grad_regulariser": b,
                         "grad_total": tot, "likelihood_share": share})

    same = sum(1 for r in rows if r["grad_likelihood"] * r["grad_regulariser"] > 0)
    print(f"\nbackward gates: {len(rows)}")
    print(f"  both terms pushing the SAME way: {same}/{len(rows)}")
    print(f"  mean likelihood share of |gradient|: "
          f"{sum(r['likelihood_share'] for r in rows)/len(rows):.3f}")
    up = sum(1 for r in rows if r["grad_total"] < 0)
    print(f"  net gradient pushing the gate UP (descent raises theta): {up}/{len(rows)}")
    print("\nA negative gradient raises theta under descent, i.e. applies MORE prior.")
    print("If both columns are negative the likelihood is not silent -- it agrees")
    print("with the regulariser, and nothing in the objective opens the gate.")

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"held_out": args.held_out, "seed": args.seed,
         "backward_bias": crf.backward_bias, "reg_weight": args.reg_weight,
         "n_train_sequences": n, "gates": rows}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
