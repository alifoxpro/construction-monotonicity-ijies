"""
Leave-one-site-out cross-validation over the whole sequence corpus.

Why this replaces the fixed train/val split
-------------------------------------------
The fixed split evaluates on 2 sequences, so every reported number is one draw
from a 2-site sample — far too noisy for a journal claim, and the previous
trainer additionally selected the best epoch *on that same split*, which leaks
the eval set into the headline number.

Here, each of the N labeled sites is held out in turn, the model trains on the
other N-1, and the held-out site's FINAL-epoch metrics are recorded. Averaging
over N folds x S seeds turns 13 sequences into 13*S independent measurements and
yields a mean +/- std that survives review. No epoch is selected on the held-out
site; the per-epoch history is stored so an inner-fold selection can be added
later without rerunning.

Usage
-----
    # the decisive comparison, all four modes, 3 seeds
    python src/experiments/run_loso.py --data seq_data --modes none,free,hard,gated --seeds 3

    # a quick smoke test first (2 folds, 1 seed, small images)
    python src/experiments/run_loso.py --modes gated,hard --seeds 1 --folds 2 \
        --epochs 2 --img-size 160 --max-len 16

Results stream to `<out>/results.jsonl` (one row per fold x seed x mode) and are
summarized into `<out>/summary.csv`. Reruns skip rows already present, so the
grid can be interrupted and resumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ importable
from data.sequence_dataset import list_sequence_dirs           # noqa: E402
from models.stage_sequence_model import TEMPORAL_ENCODERS      # noqa: E402
from train_sequence import MODES, resolve_device, run_fold     # noqa: E402


def _maybe_ignore_interrupts() -> None:
    """Optionally ignore console interrupts (LOSO_IGNORE_SIGINT=1).

    A detached grid on Windows keeps receiving spurious console control events
    when the shell that launched it goes away -- the run dies mid-fold with a
    bare '^C' in the log. This is opt-in via env var so an interactive run still
    stops normally with Ctrl+C; kill the detached job with `schtasks /end` or
    Task Manager instead.
    """
    if os.environ.get("LOSO_IGNORE_SIGINT") != "1":
        return
    import signal
    for name in ("SIGINT", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, signal.SIG_IGN)
            except (ValueError, OSError):
                pass


def arm_name(mode: str, temporal: str) -> str:
    """Display name for one (transition mode, temporal encoder) combination."""
    return mode if temporal == "none" else f"{mode}+{temporal}"


# Hyper-parameter defaults. A row written before these were ablatable has none
# of the fields, so .get() with these values makes old rows compare equal to a
# default-settings run -- which is what they were.
ABLATION_DEFAULTS = {"reg_weight": 0.1, "backward_bias": 8.0, "gate_init": 2.0,
                     "backbone": "resnet18", "freeze_backbone": False}


def run_key(row: dict) -> tuple:
    """Identity of a run, INCLUDING the ablatable hyper-parameters.

    Without these the resume logic treats two different settings as the same run
    and skips the second one, so an ablation would appear to complete instantly
    while producing nothing.

    `backbone` and `freeze_backbone` joined this tuple on 2026-08-20 for exactly
    that reason: without them a resnet50 grid resumed on top of the resnet18
    results.jsonl would report every run as already done. Rows written before
    that date carry no backbone field and default to resnet18, which is what they
    in fact used, so old and new keys agree.
    """
    return (
        row["mode"], row.get("temporal", "none"), row["held_out"], row["seed"],
        round(float(row.get("reg_weight", ABLATION_DEFAULTS["reg_weight"])), 6),
        round(float(row.get("backward_bias", ABLATION_DEFAULTS["backward_bias"])), 6),
        round(float(row.get("gate_init", ABLATION_DEFAULTS["gate_init"])), 6),
        str(row.get("backbone", ABLATION_DEFAULTS["backbone"])),
        bool(row.get("freeze_backbone", ABLATION_DEFAULTS["freeze_backbone"])),
    )


def load_done(results_path: Path) -> set[tuple]:
    """Runs already recorded — for resuming."""
    done: set[tuple] = set()
    if not results_path.exists():
        return done
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a half-written last line from an interrupted run
            # rows written before the temporal axis existed default to "none"
            done.add(run_key(row))
    return done


def summarize(results_path: Path, summary_path: Path) -> list[dict]:
    """Aggregate per-fold rows into mean +/- std per mode."""
    rows: list[dict] = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    by_arm: dict[str, list[dict]] = {}
    for r in rows:
        by_arm.setdefault(arm_name(r["mode"], r.get("temporal", "none")), []).append(r)

    def agg(values: list[float]) -> tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)

    summary = []
    for arm in sorted(by_arm, key=lambda a: (MODES.index(a.split("+")[0]), a)):
        rs = by_arm[arm]
        acc_m, acc_s = agg([r["acc"] for r in rs])
        vio_m, vio_s = agg([r["violation"] for r in rs])
        gates = [r["mean_backward_gate"] for r in rs if r.get("mean_backward_gate") is not None]
        gate_m, _ = agg(gates)
        row = {
            "arm": arm,
            "n_runs": len(rs),
            "acc_mean": round(acc_m, 4),
            "acc_std": round(acc_s, 4),
            "violation_mean": round(vio_m, 4),
            "violation_std": round(vio_s, 4),
        }
        # segmental metrics: mean +/- std across folds x seeds, same as accuracy
        for key in ("macro_f1", "edit", "f1@10", "f1@25", "f1@50"):
            vals = [r[key] for r in rs if r.get(key) is not None]
            m, s = agg(vals)
            row[f"{key}_mean"] = round(m, 4) if vals else ""
            row[f"{key}_std"] = round(s, 4) if vals else ""
        # onset MAE can be NaN for a fold where no stage was both gold and predicted
        onsets = [r["onset_mae_frames"] for r in rs
                  if r.get("onset_mae_frames") is not None
                  and r["onset_mae_frames"] == r["onset_mae_frames"]]
        om, os_ = agg(onsets)
        row["onset_mae_mean"] = round(om, 3) if onsets else ""
        row["onset_mae_std"] = round(os_, 3) if onsets else ""
        row["onset_missing_total"] = sum(r.get("onset_missing", 0) for r in rs)
        row["mean_backward_gate"] = round(gate_m, 4) if gates else ""
        summary.append(row)

    if summary:
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
    return summary


def print_summary(summary: list[dict]) -> None:
    if not summary:
        print("(no results yet)")
        return
    def cell(s, key, fmt="{:.3f}"):
        return fmt.format(s[f"{key}_mean"]) if s.get(f"{key}_mean", "") != "" else "-"

    print(f"\n{'arm':<16} {'runs':>4} {'acc':>7} {'mF1':>7} {'viol':>7} "
          f"{'edit':>7} {'F1@10':>7} {'F1@25':>7} {'F1@50':>7} {'onset':>7} {'gate':>7}")
    print("-" * 96)
    for s in summary:
        gate = f"{s['mean_backward_gate']}" if s["mean_backward_gate"] != "" else "-"
        onset = f"{s['onset_mae_mean']:.1f}" if s["onset_mae_mean"] != "" else "-"
        print(f"{s['arm']:<16} {s['n_runs']:>4} {s['acc_mean']:>7.3f} "
              f"{cell(s, 'macro_f1'):>7} {s['violation_mean']:>7.3f} {cell(s, 'edit'):>7} "
              f"{cell(s, 'f1@10'):>7} {cell(s, 'f1@25'):>7} {cell(s, 'f1@50'):>7} "
              f"{onset:>7} {gate:>7}")
    print("(std devs are in summary.csv; onset = mean absolute onset error in frames)")


def main():
    _maybe_ignore_interrupts()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="seq_data", help="corpus root (splits are pooled and re-partitioned)")
    ap.add_argument("--out", default="experiments/loso", help="output directory")
    ap.add_argument("--modes", default="none,free,hard,gated",
                    help=f"comma-separated subset of {','.join(MODES)}")
    ap.add_argument("--temporals", default="none",
                    help=f"comma-separated subset of {','.join(TEMPORAL_ENCODERS)}; "
                         "every (mode x temporal) pair becomes one arm. "
                         "'--modes none --temporals lstm' is baseline (b)")
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds per fold")
    ap.add_argument("--folds", type=int, default=None, help="cap the number of folds (smoke tests)")
    ap.add_argument("--num-stages", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    # NOTE: the backbone sees batch_size * max_len images in ONE forward pass.
    # 4 x 64 = 256 images at 224px peaks at ~6.0 GB, which leaves no headroom on
    # an 8 GB laptop GPU and fails as CUDNN_STATUS_EXECUTION_FAILED. 2 x 64 = 128
    # images peaks at ~3.1 GB.
    # 1, NOT 2. The published grid ran at 1 (experiments/loso/run_grid.bat) and
    # a batch-size-2 pass was taken and discarded
    # (experiments/loso/discarded_batchsize2_results.jsonl). The two are not
    # interchangeable: re-running the `hard` arm at 2 reproduced only 6 of 39
    # published cells and moved mean accuracy from .436 to .478, with single
    # cells swinging .893 -> .036. A default that silently contradicts the
    # published runs is a trap for anyone rebuilding the reproducibility
    # package, so the default now matches the paper.
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--backbone-lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--reg-weight", type=float, default=0.1,
                    help="lambda on the prior regulariser (ablatable)")
    ap.add_argument("--backward-bias", type=float, default=8.0,
                    help="B, magnitude of the backward-transition prior (ablatable)")
    ap.add_argument("--backbone", default="resnet18",
                    choices=["resnet18", "resnet50", "convnext_tiny"],
                    help="per-frame feature extractor; resnet18 reproduces the "
                         "published grid")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="train only the head on frozen pretrained features")
    ap.add_argument("--gate-init", type=float, default=2.0,
                    help="initial gate logit; sigmoid(2.0)=0.881 (ablatable)")
    ap.add_argument("--cpu", action="store_true", help="force CPU (GPU busy or unstable)")
    ap.add_argument("--summarize-only", action="store_true", help="re-aggregate without training")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad:
        ap.error(f"unknown mode(s) {bad}; choose from {MODES}")
    temporals = [t.strip() for t in args.temporals.split(",") if t.strip()]
    bad = [t for t in temporals if t not in TEMPORAL_ENCODERS]
    if bad:
        ap.error(f"unknown temporal encoder(s) {bad}; choose from {TEMPORAL_ENCODERS}")
    arms = [(m, t) for m in modes for t in temporals]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.csv"

    if args.summarize_only:
        print_summary(summarize(results_path, summary_path))
        return

    sites = list_sequence_dirs(args.data)
    if len(sites) < 2:
        ap.error(f"need at least 2 labeled sequences, found {len(sites)} under {args.data}")
    if args.folds:
        sites_held = sites[: args.folds]
    else:
        sites_held = sites

    device = resolve_device(args.cpu)
    done = load_done(results_path)
    planned = len(arms) * len(sites_held) * args.seeds
    print(f"Corpus: {len(sites)} sites | folds: {len(sites_held)} | "
          f"arms: {[arm_name(m, t) for m, t in arms]} | seeds: {args.seeds} | device: {device}")
    print(f"Runs planned: {planned} | already done: {len(done)}")

    started = time.time()
    completed = 0
    for mode, temporal in arms:
        for held in sites_held:
            train_dirs = [s for s in sites if s != held]
            for seed in range(args.seeds):
                key = run_key({
                    "mode": mode, "temporal": temporal, "held_out": held.name,
                    "seed": seed, "reg_weight": args.reg_weight,
                    "backward_bias": args.backward_bias, "gate_init": args.gate_init,
                    "backbone": args.backbone,
                    "freeze_backbone": args.freeze_backbone,
                })
                if key in done:
                    continue
                cfg = {
                    "mode": mode,
                    "temporal": temporal,
                    "num_stages": args.num_stages,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "img_size": args.img_size,
                    "max_len": args.max_len,
                    "backbone_lr": args.backbone_lr,
                    "head_lr": args.head_lr,
                    "reg_weight": args.reg_weight,
                    "backward_bias": args.backward_bias,
                    "gate_init": args.gate_init,
                    "backbone": args.backbone,
                    "freeze_backbone": args.freeze_backbone,
                    "seed": seed,
                }
                t0 = time.time()
                print(f"\n=== arm={arm_name(mode, temporal)} held_out={held.name} seed={seed} ===")
                # Release the previous run's cached blocks. Without this the
                # allocator's fragmentation accumulates across hundreds of runs
                # and eventually OOMs mid-grid on a small card.
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    res = run_fold(train_dirs, [held], cfg, device, verbose=False)
                except RuntimeError as exc:
                    # A transient CUDA/cuDNN fault must not kill a 300-run grid.
                    # Retry once on a cleared cache, then fall back to CPU, then
                    # record the failure and move on.
                    msg = str(exc)
                    print(f"  RuntimeError: {msg[:120]}")
                    res = None
                    # Retry on the SAME device first, after a pause: a CUDA OOM
                    # right after another process was killed usually means the
                    # driver has not reclaimed that process's context yet, not
                    # that the card is genuinely full. Falling straight through
                    # to CPU turns a 1-minute run into a 14-minute one.
                    for attempt, dev, wait in ((1, device, 45), (2, device, 90), (3, "cpu", 0)):
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        if wait:
                            print(f"  waiting {wait}s for the device to settle ...")
                            time.sleep(wait)
                        print(f"  retry {attempt} on {dev} ...")
                        try:
                            res = run_fold(train_dirs, [held], cfg, dev, verbose=False)
                            break
                        except RuntimeError as exc2:
                            print(f"  retry {attempt} failed: {str(exc2)[:120]}")
                    if res is None:
                        with open(out_dir / "failures.jsonl", "a") as f:
                            f.write(json.dumps({
                                "mode": mode, "temporal": temporal,
                                "held_out": held.name, "seed": seed, "error": msg[:500],
                            }) + "\n")
                        print("  giving up on this run; recorded in failures.jsonl")
                        continue
                res.pop("state_dict", None)  # 156 ResNet checkpoints is not worth the disk
                gs = res["gate_stats"]
                row = {
                    "mode": mode,
                    "temporal": temporal,
                    "held_out": held.name,
                    "seed": seed,
                    # Hyper-parameters MUST be recorded, not just applied. They
                    # are part of run_key, so a row without them is indistinguish-
                    # able from a default-settings run and an ablation becomes
                    # unattributable after the fact.
                    "reg_weight": args.reg_weight,
                    "backward_bias": args.backward_bias,
                    "gate_init": args.gate_init,
                    **res["final"],  # acc, violation, macro_f1, edit, f1@k, onset_*
                    "mean_backward_gate": gs["mean_backward_gate"] if gs else None,
                    "min_backward_gate": gs["min_backward_gate"] if gs else None,
                    "n_train_seq": res["n_train_seq"],
                    "eval_stage_counts": res["eval_stage_counts"],
                    "gold_violation_eval": res["gold_violation_eval"],
                    "seconds": round(time.time() - t0, 1),
                    "history": res["history"],
                }
                with open(results_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                completed += 1
                remaining = planned - len(done) - completed
                eta = (time.time() - started) / completed * remaining / 60
                print(f"  acc {row['acc']:.3f} | violation {row['violation']:.3f} | "
                      f"{row['seconds']:.0f}s | ~{eta:.0f} min left")

    print_summary(summarize(results_path, summary_path))
    print(f"\nRows: {results_path}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
