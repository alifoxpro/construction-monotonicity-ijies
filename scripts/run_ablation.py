"""Hyper-parameter ablation for the gated transition layer (paper section 7.5).

Varies one parameter at a time from the default used in every headline run
(lambda = 0.1, B = 8.0, gate init = 2.0), on the gated arm only -- no other arm
has a gate to ablate.

ONE seed per fold, not three. 13 folds x 7 settings at ~130 s is about 3.3 hours;
at three seeds it would be ten. Settings are therefore compared against each
other at seed 0 and never against the three-seed headline numbers, and the paper
says so.

Results go to experiments/ablation/, NOT the main grid, so an ablation run cannot
perturb the headline table. run_loso.py resumes on a key that includes these
hyper-parameters, so re-running this script is safe and skips completed work.

    python src/experiments/run_ablation.py            # run everything
    python src/experiments/run_ablation.py --dry-run  # list the settings

The two settings that carry most of the weight:

  gate_init = -2.0   starts the gate mostly OPEN (sigmoid(-2) = 0.119). The
                     mechanism claim predicts the regulariser closes it, since
                     no backward evidence holds it open.
  reg_weight = 0.0   removes the regulariser entirely. With no backward evidence
                     and no prior pressure the gate should not move at all.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "ablation"

# (label, reg_weight, backward_bias, gate_init)
SETTINGS = [
    ("default",        0.1,  8.0,  2.0),
    ("lambda=0.0",     0.0,  8.0,  2.0),
    ("lambda=1.0",     1.0,  8.0,  2.0),
    ("gate_init=0.0",  0.1,  8.0,  0.0),
    ("gate_init=-2.0", 0.1,  8.0, -2.0),
    ("B=2.0",          0.1,  2.0,  2.0),
    ("B=20.0",         0.1, 20.0,  2.0),
]

COMMON = [
    "--data", "seq_data", "--modes", "gated", "--temporals", "none",
    "--seeds", "1", "--num-stages", "5", "--epochs", "10",
    "--img-size", "224", "--max-len", "64", "--batch-size", "1",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"{len(SETTINGS)} settings x 13 folds x 1 seed")
    for label, reg, bias, init in SETTINGS:
        print(f"  {label:15s} lambda={reg}  B={bias}  gate_init={init}")
    if args.dry_run:
        return 0

    started = time.time()
    for i, (label, reg, bias, init) in enumerate(SETTINGS, 1):
        cmd = [sys.executable, "-u", str(REPO / "src" / "experiments" / "run_loso.py"),
               *COMMON, "--out", args.out,
               "--reg-weight", str(reg),
               "--backward-bias", str(bias),
               "--gate-init", str(init)]
        print(f"\n{'=' * 72}\n[{i}/{len(SETTINGS)}] {label}"
              f"   lambda={reg} B={bias} gate_init={init}\n{'=' * 72}", flush=True)
        r = subprocess.run(cmd, cwd=str(REPO))
        if r.returncode != 0:
            print(f"\nFAILED on setting {label} (exit {r.returncode}); "
                  f"re-run this script to resume from here")
            return r.returncode
        print(f"[{i}/{len(SETTINGS)}] {label} done "
              f"({(time.time() - started) / 60:.0f} min elapsed)", flush=True)

    print(f"\nall settings complete in {(time.time() - started) / 60:.0f} min")
    print(f"rows -> {Path(args.out) / 'results.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
