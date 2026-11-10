"""Is an individual gate identifiable alongside the free transition matrix?

IJIES editor, second round, item 4. The fixed-soft control separated prior
STRENGTH from gate LEARNABILITY, but it did not establish that a single gate
value means anything on its own while a free transition matrix sits underneath
it. This script answers that, and the answer is no.

THE ALGEBRA. The layer computes, for every pair (i, j),

    psi_eff(i, j) = psi_free(i, j) + sigmoid(theta_ij) * prior_bias(i, j)
    prior_bias(i, j) = -B  if j < i  (backward),  0 otherwise

On a backward cell BOTH psi_free(i, j) and theta_ij are free parameters of the
same scalar. For any target gate value g' in (0, 1) we can set

    psi_free'(i, j) = psi_free(i, j) + B * (sigmoid(theta_ij) - g')

and psi_eff is unchanged. The likelihood sees only psi_eff, so it cannot tell
the two parameterisations apart: there is an exact one-dimensional flat
direction per backward cell. On a FORWARD cell prior_bias is 0, so the gate
does not enter the model at all and is unidentifiable outright.

This does not contradict Section 7.2's nonzero dL/dtheta. Moving theta ALONE
changes psi_eff and so changes the likelihood. What is flat is the COMPENSATED
move (d theta, d psi_free) that holds psi_eff fixed. Both statements hold at
once, and this script checks both.

WHAT IS MEASURED, on a real trained fold with real emissions, not a toy:

  1. equivalence   -- psi_eff, the NLL and the Viterbi paths under the trained
                      parameters and under a re-gated copy, for gates swept
                      across (0, 1)
  2. null direction-- the directional derivative of the NLL along the
                      compensating direction, against the raw |dL/dtheta|
  3. forward cells -- the gate's effect where the prior bias is zero

If the NLL and every decoded path are identical while the gate moves from near
0 to near 1, the learned gate value carries no information the data put there,
and the paper may interpret psi_eff and the decoder's behaviour but not the
gate value as such.

    python src/experiments/gate_identifiability.py
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.sequence_dataset import SequenceFrameDataset, list_sequence_dirs, pad_collate  # noqa: E402
from train_sequence import build_model, resolve_device, run_fold                          # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="seq_data")
    ap.add_argument("--held-out", default="seq0003")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--reg-weight", type=float, default=0.1)
    ap.add_argument("--json", default="experiments/gate_identifiability.json")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    dirs = list_sequence_dirs(args.data)
    train_dirs = [d for d in dirs if d.name != args.held_out]
    eval_dirs = [d for d in dirs if d.name == args.held_out]
    device = resolve_device(args.cpu)

    # Same config dict gate_gradient.py trains with, so the two analyses are
    # measured on the same arm under the same schedule.
    cfg = {"mode": "gated", "temporal": "none", "num_stages": 5,
           "epochs": args.epochs, "batch_size": 1, "img_size": 224,
           "max_len": 64, "backbone_lr": 1e-4, "head_lr": 1e-3,
           "reg_weight": args.reg_weight, "seed": args.seed}

    print(f"training {args.held_out} seed {args.seed} on {device}, "
          f"gated arm, to obtain real parameters\n")
    res = run_fold(train_dirs, eval_dirs, cfg, device, verbose=False)
    model = build_model(cfg, device)
    model.load_state_dict(res["state_dict"])
    model.eval()
    crf = model.crf
    B = crf.backward_bias
    K = crf.num_stages

    # ---- real emissions from the training sequences --------------------
    from torch.utils.data import DataLoader
    ds = SequenceFrameDataset(sequence_dirs=train_dirs, img_size=224, max_len=64)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=pad_collate)
    batches = []
    with torch.no_grad():
        for frames, labels, mask in loader:
            frames = frames.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            batches.append((model.emissions(frames, mask), labels, mask))
    print(f"{len(batches)} training sequences; emissions computed once and "
          f"held fixed\n")

    theta0 = crf.gate_logit.detach().clone()
    free0 = crf.trans_free.detach().clone()
    eff0 = crf.effective_transitions().detach().clone()
    bw = crf.backward_mask.bool()

    def nll_total() -> float:
        """Likelihood only -- reg_weight 0, since the regulariser is the term
        that DOES see the gate and would mask the invariance we are testing."""
        with torch.no_grad():
            return float(sum(crf.nll(em, lb, mk, reg_weight=0.0)
                             for em, lb, mk in batches))

    def paths() -> list[list[int]]:
        with torch.no_grad():
            out = []
            for em, _, mk in batches:
                out.extend(crf.decode(em, mk))
            return out

    base_nll = nll_total()
    base_paths = paths()
    print(f"trained backward gates : min {torch.sigmoid(theta0[bw]).min():.4f} "
          f"max {torch.sigmoid(theta0[bw]).max():.4f}")
    print(f"likelihood-only NLL    : {base_nll:.6f}\n")

    # ---- 1. equivalence under a compensated re-gating ------------------
    print("[1] re-gate every backward cell, compensate psi_free, "
          "hold psi_eff fixed")
    hdr = (f"{'target gate':>12s} {'max |d psi_eff|':>16s} {'d NLL':>14s} "
           f"{'paths identical':>16s}")
    print(hdr)
    print("-" * len(hdr))
    sweep = []
    for g_target in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        g_new = torch.full_like(theta0, float(g_target))
        theta_new = torch.log(g_new / (1.0 - g_new))
        # psi_eff = psi_free + sigmoid(theta) * prior_bias, so holding psi_eff
        # fixed while moving the gate to g' needs
        #     psi_free' = psi_free + (sigmoid(theta0) - g') * prior_bias
        # Taken straight from prior_bias rather than from B and a mask: the
        # buffer already carries the sign and is zero on forward cells, so
        # there is no sign to get backwards and no mask to forget.
        delta = (torch.sigmoid(theta0) - g_new) * crf.prior_bias
        with torch.no_grad():
            crf.gate_logit.copy_(torch.where(bw, theta_new, theta0))
            crf.trans_free.copy_(free0 + delta)
        eff_new = crf.effective_transitions().detach()
        d_eff = float((eff_new - eff0).abs().max())
        d_nll = nll_total() - base_nll
        same = paths() == base_paths
        print(f"{g_target:12.2f} {d_eff:16.3e} {d_nll:14.3e} "
              f"{str(same):>16s}")
        sweep.append({"target_gate": g_target, "max_abs_d_psi_eff": d_eff,
                      "d_nll": d_nll, "paths_identical": bool(same)})
    with torch.no_grad():
        crf.gate_logit.copy_(theta0)
        crf.trans_free.copy_(free0)

    # ---- 2. the null direction ----------------------------------------
    print("\n[2] gradient along the compensating direction, against "
          "d/d theta alone")
    crf.zero_grad(set_to_none=True)
    total = sum(crf.nll(em, lb, mk, reg_weight=0.0) for em, lb, mk in batches)
    total.backward()
    g_theta = crf.gate_logit.grad.detach().clone()
    g_free = crf.trans_free.grad.detach().clone()
    crf.zero_grad(set_to_none=True)

    # the compensating direction: d theta = v, d psi_free = B * s(1-s) * v
    s = torch.sigmoid(theta0)
    v = torch.zeros_like(theta0)
    v[bw] = 1.0
    comp_free = B * s * (1.0 - s) * v * crf.backward_mask
    dir_deriv = float((g_theta * v).sum() + (g_free * comp_free).sum())
    raw_theta = float(g_theta[bw].abs().max())
    scale = float(g_theta[bw].abs().sum() + g_free[bw].abs().sum())
    print(f"    directional derivative along the flat direction : "
          f"{dir_deriv:.3e}")
    print(f"    max |dL/dtheta| on a backward cell              : "
          f"{raw_theta:.3e}")
    print(f"    ratio (flat / raw), 0 means exactly flat        : "
          f"{abs(dir_deriv) / scale if scale else float('nan'):.3e}")

    # ---- 3. forward cells ---------------------------------------------
    print("\n[3] gates on forward and self transitions")
    fwd = ~bw
    with torch.no_grad():
        crf.gate_logit.copy_(torch.where(fwd, torch.full_like(theta0, -20.0),
                                         theta0))
    d_eff_fwd = float((crf.effective_transitions().detach() - eff0).abs().max())
    d_nll_fwd = nll_total() - base_nll
    with torch.no_grad():
        crf.gate_logit.copy_(theta0)
    n_fwd = int(fwd.sum())
    print(f"    driving all {n_fwd} forward/self gates to ~0 changes "
          f"psi_eff by {d_eff_fwd:.3e}")
    print(f"    and the NLL by {d_nll_fwd:.3e}: the prior bias is zero there, "
          f"so those gates\n    are not in the model at all")

    identified = all(r["paths_identical"] and abs(r["d_nll"]) < 1e-4
                     for r in sweep)
    print("\nconclusion")
    if identified:
        print("    A backward gate can be placed anywhere in (0, 1) with an")
        print("    exactly compensating free transition, leaving psi_eff, the")
        print("    likelihood and every decoded path unchanged. The gate is")
        print("    NOT identifiable from the data; its learned value is fixed")
        print("    by the regulariser. Interpret psi_eff and the decoder's")
        print("    behaviour, not the gate value.")
    else:
        print("    The sweep did NOT reproduce the invariance -- check the")
        print("    parameterisation before relying on this result.")

    out = {
        "held_out": args.held_out, "seed": args.seed, "backward_bias": B,
        "num_stages": K, "n_train_sequences": len(batches),
        "trained_gate_min": float(torch.sigmoid(theta0[bw]).min()),
        "trained_gate_max": float(torch.sigmoid(theta0[bw]).max()),
        "base_nll_likelihood_only": base_nll,
        "sweep": sweep,
        "directional_derivative_flat": dir_deriv,
        "max_abs_grad_theta_backward": raw_theta,
        "forward_gate_max_abs_d_psi_eff": d_eff_fwd,
        "forward_gate_d_nll": d_nll_fwd,
        "gate_identifiable_from_likelihood": not identified,
    }
    path = repo / args.json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
