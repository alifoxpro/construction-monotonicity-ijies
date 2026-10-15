# Reproducibility package

Built by `scripts/build_release.py`. Every path below is relative to this
directory. `CHECKSUMS.sha256` covers every file in the package.

## What the reviewer asked for, and where it is

| Requested | Where |
|---|---|
| corpus manifest | `corpus/manifest.csv` — one row per frame: site, raw index, filename, SHA-256, annotator A label, annotator B label, whether excluded, whether kept in the 64, and its position in the subsample |
| exclusion masks | `corpus/excluded/<site>.json` — the exact raw indices dropped, and the rule |
| both annotation versions | `corpus/annotation_a/<site>.csv` (human, the labels used for training) and `corpus/annotation_b/<site>.csv` (second pass). `corpus/sites.csv` records which protocol round each site was labelled under; `corpus/labelling-protocol.md` is the protocol |
| exact 64-frame indices | `corpus/indices_64.json` — per site, the kept positions AND the raw frame index each maps back to |
| site splits | `config/grid.json` — leave-one-site-out over the 13 sites in `corpus/sites.csv`; no validation split |
| seeds | `config/grid.json` — 0, 1, 2 for the grid; 0-4 for the synthetic studies |
| epoch count | `config/grid.json` — 10, fixed in advance, no epoch selected on held-out data |
| preprocessing | `config/preprocessing.json` — stride, resize, the subsampling formula, the exclusion rule |
| baseline configurations | `config/baselines.json` — backbone, LSTM, transformer, MS-TCN, ASFormer, optimiser |
| per-site metrics | `metrics/loso_results.jsonl` as published, plus `metrics/per_site/loso_per_run.csv` |
| per-site predictions | `predictions/pred_<site>_<arm>_seed<n>.json` — decoded track, gold track, and the index mapping back to raw frames (195 of 195 present in this build) |
| synthetic settings and outputs | `synthetic/` — generator settings and every per-seed row |
| rerun discrepancies | `rerun/notes.md`, `rerun/rerun_full_grid.jsonl`, `rerun/discarded_batchsize2_results.jsonl` |
| scripts producing all tables | `scripts/`, with `scripts/SOURCES.txt` mapping each back to its repository path |

## What is not here

* **Source frames.** Not redistributed, as the review allows. `corpus/sites.csv`
  carries the source URL, licence note and stride for every sequence, and
  `corpus/manifest.csv` carries a SHA-256 per frame, so a reader who re-extracts
  from the sources can prove they hold the same pixels that were scored.
* **A DOI.** Minting a public identifier is an act of publication and is left to
  the authors; this package is what gets deposited.

## Environment the published numbers were produced on

{
 "python": "3.12.10",
 "torch": "2.7.0+cu128",
 "cuda": "12.8",
 "torchvision": "0.22.0+cu128",
 "gpu": "NVIDIA GeForce RTX 5070 Laptop GPU",
 "platform": "Windows-11-10.0.26300-SP0"
}

## Regenerating the displayed numbers

Run from the repository root, not from this directory.

```
python scripts/paper_stats.py          # Table 1, and every paired per-site test
python scripts/monotone_floor.py       # Table 2, the image-free controls
python scripts/onset_coverage.py       # onset coverage beside every onset error
python scripts/onset_matched.py        # the matched-set and penalised onset
python scripts/f1_aggregation.py       # the aggregation audit behind Section 6.2
python scripts/penalty_bound.py        # the score bound behind the rigid arm
python scripts/gate_gradient.py        # likelihood vs regulariser gate gradient
python scripts/rework_sweep.py         # Table 3, the synthetic sweep
python scripts/rework_concentration.py # concentrated vs scattered re-work
python scripts/compare_tas.py          # MS-TCN and ASFormer
python scripts/compare_backbone.py     # ResNet-50 and ConvNeXt-Tiny
python scripts/run_loso.py             # the grid itself (~9 h on the GPU above)
python scripts/dump_all_predictions.py # predictions + the full-grid rerun (~6 h)
```

A run at `--batch-size 2` does **not** reproduce the grid; see
`rerun/notes.md`. The decode is not bit-deterministic, and the margin is
measured rather than claimed: `rerun/rerun_full_grid.jsonl` carries the
published and reproduced value for every cell.
