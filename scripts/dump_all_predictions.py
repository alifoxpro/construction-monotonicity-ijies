"""Per-site decoded tracks for the whole LOSO grid — IJIES reviewer 1, point 2.

The reviewer asks the reproducibility package to carry "per-site predictions and
metrics" and "the rerun discrepancies". run_loso.py records metrics but not
predictions, so the tracks have to be regenerated, and regenerating them is also
the rerun: each cell is retrained by the published code path and its accuracy is
compared with the published one.

WHAT IT DOES NOT DO. It does not reimplement training. It calls run_fold, the
same entry point run_loso.py calls, for the same reason dump_predictions.py
does: an earlier hand-rolled loop consumed the RNG in a different order and
trained a different model. Any difference from the published number is therefore
a property of the run, not of this script.

REPRODUCTION IS NOT EXACT, AND THAT IS THE POINT OF RECORDING IT. Training
reproduces to about four significant figures, but the Viterbi decode does not
always land on the same path when two paths score within float noise of each
other, so a minority of cells differ. The per-cell comparison is written to
`rerun.jsonl` and summarised at the end: how many accuracies match exactly, the
largest absolute difference, and whether the paired contrast the paper rests on
moves. Nothing here is averaged into a published number -- the published numbers
stay as they are, in experiments/loso/results.jsonl.

Resumable: a cell whose prediction file already exists is skipped, so the job can
be stopped and restarted. At roughly two minutes a cell, the full 5 x 13 x 3 grid
takes about six hours.

    python src/experiments/dump_all_predictions.py                    # whole grid
    python src/experiments/dump_all_predictions.py --arms hard gated  # subset
    python src/experiments/dump_all_predictions.py --dry-run          # plan only
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
# line_buffering=True matters here and is not cosmetic: re-wrapping stdout
# restores BLOCK buffering and silently defeats `python -u`, so an
# unattended three-hour job writes nothing to its log until it exits -- and
# if it is killed first, as happened on this machine, the log is empty and
# the failure looks like a crash with no output.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

from data.sequence_dataset import list_sequence_dirs  # noqa: E402
from train_sequence import run_fold                   # noqa: E402

#: arm name -> (transition mode, temporal encoder), matching run_loso's grid
ARMS = {
    "none":      ("none", "none"),
    "free":      ("free", "none"),
    "hard":      ("hard", "none"),
    "gated":     ("gated", "none"),
    "none+lstm": ("none", "lstm"),
}


def subsampled_indices(n_frames: int, max_len: int | None) -> list[int]:
    """Which ORIGINAL assessable positions the dataset keeps.

    Identical to dump_predictions.subsampled_indices and to the formula in
    SequenceFrameDataset.__getitem__; duplicated rather than imported so this
    file can be read on its own inside the release package.
    """
    if max_len is None or n_frames <= max_len:
        return list(range(n_frames))
    step = (n_frames - 1) / (max_len - 1)
    return [round(i * step) for i in range(max_len)]


def load_published(path: Path) -> dict[tuple[str, str, int], dict]:
    """(arm, site, seed) -> published record, from the grid's own results file."""
    out = {}
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        t = r.get("temporal", "none")
        arm = r["mode"] if t == "none" else f"{r['mode']}+{t}"
        out[(arm, r["held_out"], r["seed"])] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="seq_data")
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="1 is the published grid's value; 2 does NOT reproduce it")
    ap.add_argument("--published", default="experiments/loso/results.jsonl")
    ap.add_argument("--out", default="release/predictions")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    sites = list_sequence_dirs(REPO / args.data)
    published = load_published(REPO / args.published)
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rerun_path = out_dir / "rerun.jsonl"

    cells = [(arm, s.name, seed) for arm in args.arms
             for s in sites for seed in args.seeds]
    todo = [c for c in cells
            if not (out_dir / f"pred_{c[1]}_{c[0].replace('+', '-')}_seed{c[2]}.json").exists()]
    print(f"{len(cells)} cells, {len(cells) - len(todo)} already dumped, "
          f"{len(todo)} to run")
    missing_pub = [c for c in cells if c not in published]
    if missing_pub:
        print(f"WARNING: {len(missing_pub)} cells have no published record to "
              f"compare against: {missing_pub[:5]}")
    if args.dry_run:
        for c in todo[:20]:
            print("  would run", c)
        return 0

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}; appending comparisons to {rerun_path}\n")

    done = 0
    for arm, site, seed in todo:
        mode, temporal = ARMS[arm]
        held = [s for s in sites if s.name == site]
        train_dirs = [s for s in sites if s.name != site]
        cfg = {"mode": mode, "temporal": temporal, "num_stages": 5,
               "epochs": args.epochs, "img_size": 224, "max_len": args.max_len,
               "batch_size": args.batch_size, "backbone_lr": 1e-4,
               "head_lr": 1e-3, "reg_weight": 0.1, "seed": seed}
        t0 = time.time()
        res = run_fold(train_dirs, held, cfg, device, verbose=False,
                       return_paths=True)
        elapsed = time.time() - t0

        pred, gold = res["eval_paths"][0], res["eval_golds"][0]
        n_assessable = sum(res["eval_stage_counts"].values())
        keep = subsampled_indices(n_assessable, args.max_len)
        warn = None
        if len(keep) != len(pred):
            warn = (f"{len(keep)} subsample positions for {len(pred)} predictions")

        dest = out_dir / f"pred_{site}_{arm.replace('+', '-')}_seed{seed}.json"
        dest.write_text(json.dumps({
            "arm": arm, "mode": mode, "temporal": temporal,
            "held_out": site, "seed": seed,
            "epochs": args.epochs, "max_len": args.max_len,
            "batch_size": args.batch_size,
            "note": "pred/gold are indexed over the SUBSAMPLED assessable frames "
                    "the evaluation saw. assessable_index maps each position to "
                    "its index among assessable (stage >= 0) frames of the source "
                    "sequence; frames labelled -1 are excluded before indexing.",
            "index_warning": warn,
            "assessable_frames": n_assessable,
            "assessable_index": keep,
            "metrics": res["final"],
            "pred": pred, "gold": gold,
        }, indent=1) + "\n", encoding="utf-8")

        pub = published.get((arm, site, seed))
        row = {"arm": arm, "held_out": site, "seed": seed,
               "seconds": round(elapsed, 1),
               "rerun_acc": res["final"]["acc"],
               "rerun_f1@50": res["final"]["f1@50"],
               "published_acc": pub["acc"] if pub else None,
               "published_f1@50": pub["f1@50"] if pub else None}
        if pub:
            row["d_acc"] = res["final"]["acc"] - pub["acc"]
            row["d_f1@50"] = res["final"]["f1@50"] - pub["f1@50"]
            row["exact_acc"] = abs(row["d_acc"]) < 1e-9
        with rerun_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        done += 1
        # ASCII only: this log is written through a .bat whose console codepage
        # is cp1256, which mangles non-ASCII (an earlier run logged "Delta" as
        # mojibake) and can abort the write outright.
        mark = "" if not pub else ("=" if row.get("exact_acc") else
                                   f"d_acc {row['d_acc']:+.4f}")
        print(f"[{done}/{len(todo)}] {arm:10s} {site:9s} s{seed} "
              f"acc {res['final']['acc']:.4f} {mark}  ({elapsed:.0f}s)")

    # ---- summary of the rerun, which is itself a required artifact ----------
    rows = [json.loads(l) for l in rerun_path.open(encoding="utf-8") if l.strip()]
    cmp_rows = [r for r in rows if r.get("published_acc") is not None]
    if cmp_rows:
        exact = sum(1 for r in cmp_rows if r["exact_acc"])
        worst = max(cmp_rows, key=lambda r: abs(r["d_acc"]))
        print(f"\nRERUN: {exact}/{len(cmp_rows)} accuracies reproduce exactly")
        print(f"  largest |Δacc| {abs(worst['d_acc']):.4f} at "
              f"{worst['arm']} {worst['held_out']} seed {worst['seed']}")
        print(f"  mean Δacc {sum(r['d_acc'] for r in cmp_rows)/len(cmp_rows):+.5f}")
        print("  The published numbers are NOT replaced by these; this table is "
              "the disclosed reproduction margin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
