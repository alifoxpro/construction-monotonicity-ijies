"""
Milestone 3 training — ResNet18 backbone + learnable sequence-constraint layer.

Trains the backbone and the CRF end-to-end on ordered frame sequences, and reports
BOTH per-frame accuracy and the construction order-violation rate.

The four comparable configurations (`--mode`):

    none    vision-only baseline: per-frame cross-entropy + argmax decode
    free    classic linear-chain CRF: transitions learned, no domain prior
    hard    hard-monotonic CRF: backward transitions forbidden outright
    soft    fixed prior at the gated arm's INITIAL effective strength, frozen
    gated   OURS: prior applied through a learnable, regularized per-transition gate

`none` isolates "does any sequence model help", `free` isolates "does the domain
prior help beyond a learned CRF".

`soft` vs `gated` isolates **learnability**, and is the contrast the claim rests
on: the two arms carry the same backward penalty and differ only in whether it
can move. `hard` vs `gated` cannot do that job — it varies penalty magnitude
(1e4 against ~7) at the same time, so a difference between them is not
attributable to learnability. This was IJIES reviewer 2's central objection and
`soft` exists to answer it.

`--temporal {none,lstm,transformer}` is an ORTHOGONAL axis: an encoder placed
between the backbone and the decoder. `--temporal lstm --mode none` is
evaluation-plan baseline (b) — temporal context WITHOUT any domain constraint,
which separates "temporal modelling helps" from "the construction prior helps".

    python src/train_sequence.py --data seq_data --mode gated
    python src/train_sequence.py --data seq_data --mode hard
    python src/train_sequence.py --data seq_data --mode none
    python src/train_sequence.py --data seq_data --mode none --temporal lstm   # baseline (b)

Reported numbers are FINAL-epoch, not best-epoch: selecting the epoch on the same
split you report leaks the eval set into the result. The full per-epoch history is
saved to the run's JSON so an inner-fold selection can still be done honestly.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make models/ and data/ importable
from models.stage_sequence_model import StageSequenceModel, TEMPORAL_ENCODERS  # noqa: E402
from models.sequence_constraint import order_violation_rate       # noqa: E402
from metrics import compute_all                                   # noqa: E402

MODES = ("none", "free", "hard", "soft", "gated")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(prefer_cpu: bool = False) -> str:
    return "cpu" if prefer_cpu or not torch.cuda.is_available() else "cuda"


@torch.no_grad()
def evaluate(model, loader, device, return_paths: bool = False) -> dict:
    """All metrics in one pass: per-frame, domain, and segmental.

    `return_paths` additionally returns the decoded stage tracks and their gold
    counterparts under keys prefixed with an underscore. Anything that writes
    result rows must strip those: they are per-frame arrays, not metrics.
    """
    model.eval()
    correct = total = 0
    all_paths: list[list[int]] = []
    all_golds: list[list[int]] = []
    for frames, labels, mask in loader:
        frames, labels, mask = frames.to(device), labels.to(device), mask.to(device)
        em = model.emissions(frames, mask)
        paths = model.decode_emissions(em, mask)
        for b, path in enumerate(paths):
            L = len(path)
            gold = labels[b, :L]
            pred = torch.tensor(path, device=device)
            correct += (pred == gold).sum().item()
            total += L
            all_golds.append(gold.tolist())
        all_paths.extend(paths)
    metrics = {
        "acc": correct / max(1, total),
        "violation": order_violation_rate(all_paths),
    }
    metrics.update(compute_all(all_paths, all_golds))
    if return_paths:
        metrics["_paths"] = all_paths
        metrics["_golds"] = all_golds
    return metrics


def build_model(cfg, device):
    model = StageSequenceModel(
        cfg["num_stages"],
        use_constraint=cfg["mode"] != "none",
        transition_mode=cfg["mode"] if cfg["mode"] != "none" else "gated",
        temporal=cfg.get("temporal", "none"),
        # ablatable; defaults reproduce every run reported before 2026-08-11
        backward_bias=cfg.get("backward_bias", 8.0),
        gate_init=cfg.get("gate_init", 2.0),
        # backbone axis added 2026-08-20; the default keeps every earlier run
        backbone=cfg.get("backbone", "resnet18"),
        freeze_backbone=cfg.get("freeze_backbone", False),
    ).to(device)
    return model


def run_fold(train_dirs, eval_dirs, cfg, device, verbose: bool = True,
             return_paths: bool = False) -> dict:
    """Train on `train_dirs`, evaluate on `eval_dirs`, return a result record.

    Both arguments are explicit lists of sequence directories, so the caller
    decides the partition (fixed split, or one cross-validation fold).
    """
    from torch.utils.data import DataLoader
    from data.sequence_dataset import SequenceFrameDataset, pad_collate

    set_seed(cfg["seed"])

    train_ds = SequenceFrameDataset(
        sequence_dirs=train_dirs, img_size=cfg["img_size"], max_len=cfg["max_len"]
    )
    eval_ds = SequenceFrameDataset(
        sequence_dirs=eval_dirs, img_size=cfg["img_size"], max_len=cfg["max_len"]
    )
    train_dl = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=pad_collate
    )
    eval_dl = DataLoader(
        eval_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=pad_collate
    )

    model = build_model(cfg, device)
    # frozen gates (modes "free"/"hard") must not reach the optimizer
    groups = model.param_groups(cfg["backbone_lr"], cfg["head_lr"])
    for g in groups:
        g["params"] = [p for p in g["params"] if p.requires_grad]
    optimizer = torch.optim.Adam(groups)

    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running = 0.0
        for frames, labels, mask in train_dl:
            frames, labels, mask = frames.to(device), labels.to(device), mask.to(device)
            optimizer.zero_grad()
            em = model.emissions(frames, mask)
            loss = model.compute_loss(em, labels, mask, reg_weight=cfg["reg_weight"])
            loss.backward()
            optimizer.step()
            running += loss.item() * frames.size(0)
        m = evaluate(model, eval_dl, device)
        history.append({"epoch": epoch, "loss": running / len(train_ds), **m})
        if verbose:
            print(
                f"Epoch {epoch:02d} | loss {history[-1]['loss']:.4f} | "
                f"acc {m['acc']:.3f} | viol {m['violation']:.3f} | "
                f"macroF1 {m['macro_f1']:.3f} | edit {m['edit']:.3f} | "
                f"F1@50 {m['f1@50']:.3f} | onsetMAE {m['onset_mae_frames']:.1f}"
            )

    # Decoded tracks are fetched in a SEPARATE pass so they never enter
    # `history`, and therefore never enter `final`, which run_loso writes to
    # every result row. Evaluation is deterministic, so the extra pass changes
    # no reported number.
    paths = golds = None
    if return_paths:
        pg = evaluate(model, eval_dl, device, return_paths=True)
        paths, golds = pg["_paths"], pg["_golds"]

    return {
        "config": dict(cfg),
        "eval_paths": paths,
        "eval_golds": golds,
        "train_sites": list(train_ds.names),
        "eval_sites": list(eval_ds.names),
        "n_train_seq": len(train_ds),
        "n_eval_seq": len(eval_ds),
        "eval_stage_counts": eval_ds.stage_counts(),
        "gold_violation_train": train_ds.gold_order_violation_rate(),
        "gold_violation_eval": eval_ds.gold_order_violation_rate(),
        "history": history,
        # headline = FINAL epoch, never the best one (no selection on the eval split)
        "final": {k: v for k, v in history[-1].items() if k not in ("epoch", "loss")},
        "final_acc": history[-1]["acc"],
        "final_violation": history[-1]["violation"],
        "gate_stats": model.crf.gate_stats() if model.crf is not None else None,
        "state_dict": model.state_dict(),
    }


def cfg_from_args(args) -> dict:
    return {
        "mode": args.mode,
        "temporal": args.temporal,
        "num_stages": args.num_stages,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "max_len": args.max_len,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr,
        "reg_weight": args.reg_weight,
        "seed": args.seed,
    }


def train(args):
    from data.sequence_dataset import list_sequence_dirs

    device = resolve_device(args.cpu)
    cfg = cfg_from_args(args)
    print(f"Device: {device} | mode: {cfg['mode']} | temporal: {cfg['temporal']} | "
          f"seed: {cfg['seed']}")

    train_dirs = list_sequence_dirs(Path(args.data) / "train")
    eval_dirs = list_sequence_dirs(Path(args.data) / "val")
    print(f"train sequences: {len(train_dirs)} | val sequences: {len(eval_dirs)}")

    result = run_fold(train_dirs, eval_dirs, cfg, device)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": result.pop("state_dict"), "args": vars(args)}, out)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(result, f, indent=2)

    f = result["final"]
    print(f"\nFinal-epoch metrics  ->  {out}")
    print(f"  per-frame acc   {f['acc']:.3f}      macro-F1  {f['macro_f1']:.3f}")
    print(f"  order violation {f['violation']:.3f}      edit      {f['edit']:.3f}")
    print(f"  F1@10/25/50     {f['f1@10']:.3f} / {f['f1@25']:.3f} / {f['f1@50']:.3f}")
    print(f"  onset MAE       {f['onset_mae_frames']:.1f} frames "
          f"({f['onset_missing']} stage(s) never predicted)")
    if result["gate_stats"]:
        gs = result["gate_stats"]
        print(
            f"Backward gates: mean {gs['mean_backward_gate']:.3f} "
            f"min {gs['min_backward_gate']:.3f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="seq_data", help="root with train/ and val/ sequence folders")
    ap.add_argument("--mode", choices=MODES, default="gated", help="transition parameterization")
    ap.add_argument("--temporal", choices=TEMPORAL_ENCODERS, default="none",
                    help="temporal encoder between backbone and decoder; "
                         "'--temporal lstm --mode none' is evaluation-plan baseline (b)")
    ap.add_argument("--num-stages", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--max-len", type=int, default=None, help="cap frames per sequence")
    ap.add_argument("--backbone-lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--reg-weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-constraint", action="store_true",
                    help="deprecated alias for --mode none")
    ap.add_argument("--cpu", action="store_true", help="force CPU (GPU busy or unstable)")
    ap.add_argument("--out", default="checkpoints/sequence.pt")
    args = ap.parse_args()
    if args.no_constraint:
        args.mode = "none"
    train(args)


if __name__ == "__main__":
    main()
