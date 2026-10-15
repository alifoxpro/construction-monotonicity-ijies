"""Reproduce one LOSO fold and dump the held-out site's decoded stage track.

run_loso.py records metrics but not predictions, so stage onsets cannot be
recovered from results.jsonl after the fact. This re-runs a single fold and
writes the decoded track, which schedule_linkage.py converts to calendar dates.

IT CALLS run_fold, IT DOES NOT REIMPLEMENT IT. An earlier version of this script
had its own training loop, and two invisible differences -- the order in which
the RNG was consumed, and frozen gate parameters reaching the optimizer --
produced a DIFFERENT trained model from the one the paper reports: seeds scoring
0.42 and 0.53 in the grid collapsed to a single predicted stage here. The only
safe way to reproduce a published run is to execute the published code path.
Verify against experiments/loso/results.jsonl before trusting the output.

    python src/experiments/dump_predictions.py --held-out seq0010 --mode hard --seed 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from data.sequence_dataset import list_sequence_dirs                 # noqa: E402
from train_sequence import run_fold                                  # noqa: E402


def subsampled_indices(n_frames: int, max_len: int | None) -> list[int]:
    """Which ORIGINAL positions the dataset keeps.

    SequenceFrameDataset subsamples evenly when a sequence is longer than
    max_len, so a decoded track is indexed over the subsample. Onsets have to be
    mapped back through the same formula or they point at the wrong frames --
    and therefore the wrong dates.
    """
    if max_len is None or n_frames <= max_len:
        return list(range(n_frames))
    step = (n_frames - 1) / (max_len - 1)
    return [round(i * step) for i in range(max_len)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="seq_data")
    ap.add_argument("--held-out", default="seq0010")
    ap.add_argument("--mode", default="hard", choices=["none", "free", "hard", "gated"])
    ap.add_argument("--temporal", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-stages", type=int, default=5)
    ap.add_argument("--backbone-lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--reg-weight", type=float, default=0.1)
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--out", default="experiments/schedule")
    args = ap.parse_args()

    sites = list_sequence_dirs(REPO / args.data)
    held = [s for s in sites if s.name == args.held_out]
    if not held:
        ap.error(f"{args.held_out} not found under {args.data}")
    train_dirs = [s for s in sites if s.name != args.held_out]

    cfg = {
        "mode": args.mode, "temporal": args.temporal, "num_stages": args.num_stages,
        "epochs": args.epochs, "img_size": args.img_size, "max_len": args.max_len,
        "batch_size": args.batch_size, "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr, "reg_weight": args.reg_weight, "seed": args.seed,
    }
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"reproducing: {args.mode} | held out {args.held_out} | seed {args.seed} "
          f"| {device}")

    res = run_fold(train_dirs, held, cfg, device, verbose=True, return_paths=True)
    pred = res["eval_paths"][0]
    gold = res["eval_golds"][0]

    # the number of ASSESSABLE frames, which is what the dataset indexes over
    n_assessable = sum(res["eval_stage_counts"].values())
    keep = subsampled_indices(n_assessable, args.max_len)
    if len(keep) != len(pred):
        print(f"  WARNING: {len(keep)} subsample positions for {len(pred)} "
              f"predictions; index mapping may be wrong")

    out = Path(REPO / args.out); out.mkdir(parents=True, exist_ok=True)
    dest = out / f"pred_{args.held_out}_{args.mode}_seed{args.seed}.json"
    dest.write_text(json.dumps({
        "held_out": args.held_out, "mode": args.mode, "temporal": args.temporal,
        "seed": args.seed, "epochs": args.epochs, "max_len": args.max_len,
        "final_acc": res["final"]["acc"],
        "note": "pred/gold are indexed over the SUBSAMPLED assessable frames that "
                "the evaluation actually saw. assessable_index maps each position "
                "to its index among assessable frames; original_frame maps it to "
                "the frame number in the source sequence.",
        "assessable_index": keep,
        "pred": pred, "gold": gold,
    }, indent=1) + "\n", encoding="utf-8")

    print(f"\n  final acc {res['final']['acc']:.3f}  "
          f"F1@50 {res['final']['f1@50']:.3f}  "
          f"onset {res['final']['onset_mae_frames']}")
    print(f"  -> {dest}")
    print("  CHECK this accuracy against experiments/loso/results.jsonl for the "
          "same (mode, held_out, seed) before using the output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
