# Construction monotonicity: reproducibility package

Code, annotations and results for **"How much is construction monotonicity
worth? Rigid versus learnable sequence priors for image-based progress
recognition"** (International Journal of Intelligent Engineering and Systems).

The paper asks a narrow question: a construction site normally moves forward,
so what is it worth to *tell* a stage-recognition model that? And is it better
to impose that ordering rigidly, or to let the model learn when to relax it?

## What the study found

* **Encoding the domain beats adding architecture.** Moving from a free
  linear-chain CRF to a rigid monotonicity prior raises segmental F1@50 by
  **+0.274** across 13 held-out sites (p = .004). Structured decoding on its own
  buys almost nothing (+0.005, p = .15).
* **The learnable gate does not pay here, and we can say why.** The corpus
  contains no backward transitions for it to learn from, so the gate stays at
  its initialisation. A fixed soft-prior control shows the published
  rigid-versus-gated gap measures *penalty strength*, not learnability.
* **An image-free control disqualifies part of the result.** A predictor that
  never sees a pixel — it just splits a sequence into equal runs and emits the
  stages in order — beats every trained arm on segmental edit score, and no
  trained arm significantly exceeds it on F1@50, accuracy or stage-onset error.
  Any claim resting on those metrics is reported against that floor.

The third point is the one we would most want carried into other work:
**measure the ground-truth violation rate, an image-free control, and annotation
consistency before training**, because those three quantities decide whether an
experiment can support the claim it was built to make.

## Layout

```
corpus/       per-frame manifest, both annotation passes, exclusion masks,
              exact subsample indices, labelling protocols
config/       grid, preprocessing and baseline configurations
metrics/      per-run metrics for every arm, exactly as published
predictions/  195 decoded stage tracks, one per (arm, site, seed)
synthetic/    generator settings and every per-seed outcome
rerun/        three disclosed reproductions, including a discarded one
scripts/      every script that produces a table in the paper or supplement
```

`CHECKSUMS.sha256` covers every file above.

## Reproducing a number

Run from the repository root. Each script prints the figures it supports and
writes a JSON record beside them.

```bash
python scripts/paper_stats.py            # Table 1 and every paired per-site test
python scripts/monotone_floor.py         # Table 2, the image-free controls
python scripts/uncertainty.py            # intervals, site sensitivity, multiplicity
python scripts/f1_aggregation.py         # the aggregation rule behind Section 6.2
python scripts/penalty_bound.py          # the score bound behind the rigid arm
python scripts/gate_gradient.py          # likelihood vs regulariser gate gradient
python scripts/violation_provenance.py   # why the violation rate is 0.000
python scripts/onset_denominators.py     # every onset figure with its denominators
python scripts/asformer_equivalence.py   # ASFormer against the reference code
python scripts/subsample_sensitivity.py  # does the 64-frame choice create the result
```

Re-running the experiments themselves is slower: `scripts/run_loso.py` is about
nine hours on the GPU below, and `scripts/dump_all_predictions.py` another six.

**Use `--batch-size 1`.** A grid run at batch size 2 reproduces only 6 of 39
cells; that run is kept in `rerun/` as evidence and used for nothing.

## What is not here, and why

**No video frames.** The sequences come from third-party time-lapse videos
published under the YouTube Standard License, which conveys no redistribution
rights. The corpus ships in pointer-plus-annotation form instead:
`corpus/sites.csv` carries each source URL, licence note and extraction
parameters, and `corpus/manifest.csv` carries a **SHA-256 for every frame**, so
anyone who re-extracts from the originals can prove they hold the same pixels
that were scored. Re-extracting binds you to the original videos' terms, not to
this repository's licence.

**No pretrained weights.** Every result is a 13-fold cross-validation, so there
is no single model to ship; the configurations and seeds are in `config/`.

## Honest limits

* The decode is **not bit-deterministic**. Training reproduces to about four
  significant figures, but Viterbi can return a different path when two paths
  score within float noise. Over the full grid **138 of 195** accuracies
  reproduce exactly; the per-cell table is `rerun/rerun_full_grid.jsonl`, and
  the published numbers are not replaced by it.
* **13 sites is small**, and the folds share training data — any two train on 11
  of the same 12 sites. `scripts/uncertainty.py` reports intervals, a
  leave-one-site-out sensitivity check and the fold correlation that would be
  needed to overturn the headline.
* The corpus is **edited, monotone, new-build footage**. Its zero violation rate
  is not evidence that re-work does not happen; see
  `scripts/violation_provenance.py` for what that number can and cannot mean.

## Where specific artifacts live

| Looking for | Path |
|---|---|
| per-frame corpus manifest | `corpus/manifest.csv` — site, raw index, filename, SHA-256, both annotators' labels, exclusion flag, subsample position |
| exclusion masks | `corpus/excluded/<site>.json` |
| both annotation passes | `corpus/annotation_a/<site>.csv`, `corpus/annotation_b/<site>.csv` |
| exact 64-frame indices | `corpus/indices_64.json`, with the mapping back to raw frames |
| splits, seeds, epochs | `config/grid.json` — leave-one-site-out, seeds 0-2, 10 epochs, no validation split |
| preprocessing | `config/preprocessing.json` |
| baseline configurations | `config/baselines.json` |
| per-run metrics | `metrics/loso_results.jsonl`, `metrics/per_site/loso_per_run.csv` |
| decoded predictions | `predictions/pred_<site>_<arm>_seed<n>.json` |
| synthetic studies | `synthetic/` |
| disclosed reruns | `rerun/notes.md` |
| blinded annotation study | `scripts/blind_pairs.py`, `scripts/blind_pairs_analysis.py` |

## Environment the published numbers were produced on

```json
{
 "python": "3.12.10",
 "torch": "2.7.0+cu128",
 "cuda": "12.8",
 "torchvision": "0.22.0+cu128",
 "gpu": "NVIDIA GeForce RTX 5070 Laptop GPU",
 "platform": "Windows-11-10.0.26300-SP0"
}
```

## Licence

MIT for the code, annotations and derived results — see `LICENSE`. The source
videos are **not** covered by it.
