"""Assemble the reproducibility package — IJIES reviewer 1, point 2.

THE OBJECTION, verbatim: "Provide a versioned, review-accessible reproducibility
package corresponding to the release claimed in the Data Availability Statement.
The submitted file refers repeatedly to Supplementary Sections S1-S7 and Tables
S1-S13, but supplies only a generic supplementary placeholder and no resolvable
artifact identifier. Verification requires the corpus manifest and exclusion
masks, both annotation versions, exact 64-frame indices, site splits, seeds,
epoch count, preprocessing and baseline configurations, per-site predictions and
metrics, and synthetic generator settings and outputs. Include the rerun
discrepancies and the scripts producing all tables. This is necessary to audit
the reported tests and gate mechanism; redistribution of source frames is not
required."

This script builds every item on that list except the per-site predictions, which
are produced by dump_all_predictions.py because they require retraining, and
except the DOI: minting a public identifier is an act of publication and is left
to the authors.

Source frames are NOT copied, as the reviewer allows. Instead the manifest
carries a SHA-256 per frame, so a reader who fetches the sources from the
recorded URLs can prove they hold the same pixels we scored.

CONFIGURATION IS CROSS-CHECKED, NOT RETYPED. Every value written into
config/grid.json is asserted against the argparse default in the source file it
came from, so the package cannot silently drift from the code. If someone
changes a default without rebuilding, this script fails and says which flag.

    python src/experiments/build_release.py            # build into release/
    python src/experiments/build_release.py --no-hash  # skip frame checksums
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
from datetime import date
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from data.sequence_dataset import list_sequence_dirs  # noqa: E402

RAW = REPO / "seq_data_raw"
OUT = REPO / "release"

#: Figures published with the release, as (diagrams/ stem, name in figures/).
#: paper-fig0-stages is absent by design -- it is built from real frames.
#: The results panel is the post-correction rebuild, not the submitted one.
FIGURES_SHIPPED = [
    ("paper-fig1-method", "method.png"),
    ("paper-fig2-results-r1", "results.png"),
    ("paper-fig3-rework-sweep", "rework-sweep.png"),
]
FIG_DPI = 170

#: Authors of the manuscript, in submission order, from its front matter.
#: (family-names, given-names, affiliation). Their e-mail addresses are in the
#: manuscript as well and are intentionally not republished here.
AUTHORS = [
    ("Mohammed", "Omar Amer",
     "Department of Computer Science, College of Science, "
     "University of Diyala, Iraq"),
    ("Al-Tuwaijari", "Jamal Mustafa",
     "Department of Computer Science, College of Science, "
     "University of Diyala, Iraq"),
    ("Al-Mahdawi", "Amer Mohammed",
     "Department of Civil Engineering, College of Engineering, "
     "University of Diyala, Iraq"),
]
PAPER_TITLE = ("How much is construction monotonicity worth? Rigid versus "
               "learnable sequence priors for image-based progress recognition")
PAPER_JOURNAL = "International Journal of Intelligent Engineering and Systems"
PAPER_STATUS = "Under review (manuscript 20266482)"
PAPER_YEAR = 2026
MAX_LEN = 64

#: Scripts that produce a displayed table, figure or test. Copied verbatim so the
#: package is self-contained; SOURCES.txt records where each came from.
SCRIPTS = [
    "src/train_sequence.py",
    "src/metrics.py",
    "src/models/sequence_constraint.py",
    "src/models/stage_sequence_model.py",
    "src/models/tas_encoders.py",
    "src/data/sequence_dataset.py",
    "src/experiments/run_loso.py",
    "src/experiments/paper_stats.py",
    "src/experiments/monotone_floor.py",
    "src/experiments/onset_coverage.py",
    "src/experiments/onset_matched.py",
    "src/experiments/onset_denominators.py",
    "src/experiments/f1_aggregation.py",
    "src/experiments/uncertainty.py",
    "src/experiments/subsample_sensitivity.py",
    "src/experiments/violation_provenance.py",
    "src/data_collection/blind_pairs.py",
    "src/data_collection/blind_pairs_analysis.py",
    "src/data_collection/make_blind_batches.py",
    "src/data_collection/anonymise_blind_frames.py",
    "src/experiments/gate_gradient.py",
    "src/experiments/penalty_bound.py",
    "src/experiments/rework_sweep.py",
    "src/experiments/rework_concentration.py",
    "src/experiments/run_ablation.py",
    "src/experiments/compare_tas.py",
    "src/experiments/asformer_equivalence.py",
    "src/experiments/asformer_rerun_compare.py",
    "src/experiments/compare_backbone.py",
    "src/experiments/compare_replicate.py",
    "src/experiments/dump_predictions.py",
    "src/experiments/dump_all_predictions.py",
    "src/experiments/schedule_linkage.py",
    "src/data_collection/annotator_b_agreement.py",
    "src/experiments/build_release.py",
    # Figure generation. The Data Availability Statement promises these by
    # name, and the package shipped without them. make_stage_figure.py is
    # included although its figure is NOT published: the script is ours, the
    # frames it draws are not, and shipping the code is what lets a reader see
    # how the excluded figure was built.
    "src/experiments/plot_results.py",
    "src/experiments/plot_concentration.py",
    "src/experiments/make_stage_figure.py",
]

#: The published grid, as run. Each entry is checked against the argparse default
#: in `source` before it is written out.
GRID = {
    "source": "src/experiments/run_loso.py",
    "values": {
        "data": "seq_data",
        "num-stages": 5,
        "epochs": 10,
        "img-size": 224,
        "max-len": 64,
        "batch-size": 1,
        "backbone-lr": 1e-4,
        "head-lr": 1e-3,
        "reg-weight": 0.1,
        "backward-bias": 8.0,
        "gate-init": 2.0,
        "backbone": "resnet18",
    },
}


def verify_defaults(spec: dict) -> dict:
    """Return spec['values'], after asserting each is the source's own default."""
    text = (REPO / spec["source"]).read_text(encoding="utf-8")
    bad = []
    for flag, value in spec["values"].items():
        pat = re.compile(
            r"add_argument\(\s*[\"']--" + re.escape(flag) + r"[\"'][^)]*?default\s*=\s*([^,)\s]+)",
            re.S)
        m = pat.search(text)
        if not m:
            bad.append(f"--{flag}: no default found in {spec['source']}")
            continue
        got = m.group(1).strip().strip("\"'")
        try:
            same = float(got) == float(value)
        except ValueError:
            same = got == str(value)
        if not same:
            bad.append(f"--{flag}: package says {value!r}, source default is {got!r}")
    if bad:
        raise SystemExit("CONFIG DRIFT — rebuild aborted:\n  " + "\n  ".join(bad))
    return dict(spec["values"])


def subsampled_indices(n: int, max_len: int) -> list[int]:
    if n <= max_len:
        return list(range(n))
    step = (n - 1) / (max_len - 1)
    return [round(i * step) for i in range(max_len)]


def read_label_csv(path: Path) -> list[tuple[str, int]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().lower() in {"filename", "file", ""}:
                continue
            rows.append((row[0].strip(), int(row[1])))
    return rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="seq_data")
    ap.add_argument("--no-hash", action="store_true",
                    help="skip per-frame SHA-256 (fast rebuild while iterating)")
    args = ap.parse_args()

    grid = verify_defaults(GRID)
    print(f"config cross-checked against {GRID['source']}: "
          f"{len(grid)} defaults agree\n")

    corpus_dirs = list_sequence_dirs(REPO / args.data)
    corpus_names = [d.name for d in corpus_dirs]

    for sub in ("corpus/annotation_a", "corpus/annotation_b", "corpus/excluded",
                "config", "metrics/per_site", "rerun", "scripts", "synthetic"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    # ---- corpus manifest, exclusion masks, 64-frame indices ----------------
    from data_collection.annotator_b_agreement import ROUND1

    manifest_rows = []
    site_rows = []
    indices = {}
    for site in corpus_names:
        raw_dir = RAW / site
        a = read_label_csv(raw_dir / "labels.csv")
        b_path = raw_dir / "labels.B.csv"
        b = read_label_csv(b_path) if b_path.exists() else []
        b_by_name = {n: s for n, s in b}
        prov = json.loads((raw_dir / "provenance.json").read_text(encoding="utf-8"))

        # The dataset indexes over ASSESSABLE frames only: it drops every frame
        # annotator A marked -1 BEFORE subsampling, so the 64 kept positions are
        # positions within the assessable subsequence, not within the raw video.
        assessable = [(i, n, s) for i, (n, s) in enumerate(a) if s >= 0]
        keep = set(subsampled_indices(len(assessable), MAX_LEN))
        indices[site] = {
            "assessable_frames": len(assessable),
            "kept_positions": sorted(keep),
            "kept_raw_frame_index": [assessable[p][0] for p in sorted(keep)],
        }

        excluded = [i for i, (n, s) in enumerate(a) if s < 0]
        (OUT / "corpus/excluded" / f"{site}.json").write_text(json.dumps({
            "site": site,
            "rule": "label -1 in annotator A's labels.csv; excluded before "
                    "subsampling, so these frames are in no scored sequence",
            "n_raw_frames": len(a),
            "n_excluded": len(excluded),
            "excluded_raw_indices": excluded,
        }, indent=1) + "\n", encoding="utf-8")

        shutil.copy2(raw_dir / "labels.csv", OUT / "corpus/annotation_a" / f"{site}.csv")
        if b_path.exists():
            shutil.copy2(b_path, OUT / "corpus/annotation_b" / f"{site}.csv")

        pos_of = {assessable[p][0]: p for p in sorted(keep)}
        for i, (name, stage) in enumerate(a):
            frame = raw_dir / name
            manifest_rows.append({
                "site": site,
                "raw_frame_index": i,
                "filename": name,
                "sha256": "" if (args.no_hash or not frame.exists()) else sha256(frame),
                "label_a": stage,
                "label_b": b_by_name.get(name, ""),
                "excluded_by_a": int(stage < 0),
                "kept_in_64": int(i in pos_of),
                "subsample_position": pos_of.get(i, ""),
            })

        stages = sorted({s for _, _, s in assessable})
        scored = [assessable[p][2] for p in sorted(keep)]
        site_rows.append({
            "site": site,
            "in_corpus": 1,
            "source_url": prov.get("source_url", ""),
            "license_note": prov.get("license_note", ""),
            "stride_sec": prov.get("stride_sec", ""),
            "n_raw_frames": len(a),
            "n_labelled": len(assessable),
            "n_excluded": len(excluded),
            "n_scored": len(scored),
            "stages_present": "|".join(str(s) for s in stages),
            "n_stages": len(stages),
            "annotation_b": int(b_path.exists()),
            "protocol_round": 1 if site in ROUND1 else 2,
        })

    # Sequences collected but kept OUT of the corpus: the reader needs these to
    # audit the selection, not only the selected set.
    for d in sorted(RAW.iterdir()):
        if not d.is_dir() or d.name in corpus_names:
            continue
        prov_p = d / "provenance.json"
        prov = json.loads(prov_p.read_text(encoding="utf-8")) if prov_p.exists() else {}
        has_a = (d / "labels.csv").exists()
        site_rows.append({
            "site": d.name, "in_corpus": 0,
            "source_url": prov.get("source_url", ""),
            "license_note": prov.get("license_note", ""),
            "stride_sec": prov.get("stride_sec", ""),
            "n_raw_frames": prov.get("n_frames", ""),
            "n_labelled": len(read_label_csv(d / "labels.csv")) if has_a else 0,
            "n_excluded": "", "n_scored": 0, "stages_present": "", "n_stages": "",
            "annotation_b": int((d / "labels.B.csv").exists()),
            "protocol_round": "",
        })

    def write_csv(path: Path, rows: list[dict]):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(OUT / "corpus/manifest.csv", manifest_rows)
    write_csv(OUT / "corpus/sites.csv", site_rows)
    (OUT / "corpus/indices_64.json").write_text(
        json.dumps(indices, indent=1) + "\n", encoding="utf-8")
    shutil.copy2(REPO / "docs/labelling-protocol.md",
                 OUT / "corpus/labelling-protocol.md")
    # The frozen pairwise protocol is part of the corpus documentation even
    # though its study has not been run: it is what a reader needs in order to
    # repeat the blinded test, and freezing it is only meaningful if the frozen
    # text is published alongside the instrument.
    shutil.copy2(REPO / "docs/labelling-protocol-pairs-v2.0-frozen.md",
                 OUT / "corpus/labelling-protocol-pairs-v2.0-frozen.md")
    print(f"corpus: {len(manifest_rows)} frame rows over {len(corpus_names)} "
          f"sites, {len(site_rows) - len(corpus_names)} collected-but-excluded "
          f"sequences listed")

    # ---- configuration -----------------------------------------------------
    (OUT / "config/grid.json").write_text(json.dumps({
        "protocol": "leave-one-site-out; each of the 13 sites is held out once, "
                    "the other 12 train. No validation split: the epoch count is "
                    "fixed in advance and no epoch is selected on held-out data.",
        "arms": {"none": "per-frame cross-entropy + argmax (no CRF)",
                 "free": "CRF, learned transitions, no prior",
                 "hard": "CRF, backward penalty 10000 (rigid)",
                 "gated": "CRF, sigmoid-gated prior, gate_init 2.0 (ours)",
                 "soft": "CRF, fixed effective penalty 7.046, gate frozen, no "
                         "regulariser (added in revision, reviewer 2)",
                 "none+lstm": "bidirectional LSTM + per-frame CE",
                 "none+mstcn": "MS-TCN", "none+asformer": "ASFormer"},
        "seeds": [0, 1, 2],
        "folds": 13,
        "runs_per_arm": 39,
        "defaults_verified_against": GRID["source"],
        "defaults": grid,
        "launcher": "experiments/loso/run_grid.bat",
        "batch_size_note": "batch-size 1 is required to reproduce the published "
                           "grid. A run at batch-size 2 reproduced only 6 of 39 "
                           "cells; that grid is archived under "
                           "rerun/discarded_batchsize2_results.jsonl.",
    }, indent=1) + "\n", encoding="utf-8")

    (OUT / "config/preprocessing.json").write_text(json.dumps({
        "source": "src/data/sequence_dataset.py (default_transform)",
        "frame_extraction": "fixed stride per sequence, stored in capture order; "
                            "per-site stride_sec is in corpus/sites.csv",
        "max_width_px": 1280,
        "img_size": grid["img-size"],
        "subsampling": "sequences longer than max-len are subsampled EVENLY, not "
                       "truncated: index round(i*(n-1)/(max_len-1)) for i in "
                       "0..max_len-1, over ASSESSABLE frames only. Exact kept "
                       "indices per site: corpus/indices_64.json.",
        "exclusion_rule": "a frame labelled -1 by annotator A is dropped before "
                          "subsampling and appears in no scored sequence",
    }, indent=1) + "\n", encoding="utf-8")

    (OUT / "config/baselines.json").write_text(json.dumps({
        "source": "src/models/stage_sequence_model.py, src/models/tas_encoders.py",
        "backbone": {"default": grid["backbone"],
                     "alternatives_tested": ["resnet50", "convnext_tiny"],
                     "head": "classification head replaced by Identity",
                     "weights": "torchvision DEFAULT, locally cached"},
        "temporal_encoders": {
            "none": "frames independent; emissions are per-frame CNN outputs",
            "lstm": "nn.LSTM, bidirectional, hidden 256, 1 layer",
            "transformer": "nn.TransformerEncoder, 8 heads, ff 4*256, dropout 0.1, "
                           "sinusoidal positional encoding",
            "mstcn": "MS-TCN, emits class logits directly, per-stage auxiliary losses",
            "asformer": "ASFormer, same slot and loss structure as mstcn"},
        "optimiser": {"name": "Adam", "backbone_lr": grid["backbone-lr"],
                      "head_lr": grid["head-lr"], "epochs": grid["epochs"]},
    }, indent=1) + "\n", encoding="utf-8")

    # ---- metrics: as published, per run and per site -----------------------
    res_path = REPO / "experiments/loso/results.jsonl"
    rows = [json.loads(l) for l in res_path.open(encoding="utf-8") if l.strip()]
    shutil.copy2(res_path, OUT / "metrics/loso_results.jsonl")
    for p in ("experiments/loso/summary.csv",):
        if (REPO / p).exists():
            shutil.copy2(REPO / p, OUT / "metrics" / Path(p).name)

    keys = ["acc", "macro_f1", "f1@10", "f1@25", "f1@50", "edit", "violation",
            "onset_mae_frames", "onset_missing", "mean_backward_gate"]
    per_site = []
    for r in rows:
        t = r.get("temporal", "none")
        per_site.append({"arm": r["mode"] if t == "none" else f"{r['mode']}+{t}",
                         "held_out": r["held_out"], "seed": r["seed"],
                         **{k: r.get(k, "") for k in keys},
                         "n_train_seq": r.get("n_train_seq", ""),
                         "seconds": r.get("seconds", "")})
    per_site.sort(key=lambda d: (d["arm"], d["held_out"], d["seed"]))
    write_csv(OUT / "metrics/per_site/loso_per_run.csv", per_site)
    print(f"metrics: {len(rows)} published runs, per-run CSV written")

    # ---- synthetic generator settings and outputs --------------------------
    for src, dst in (("experiments/rework_sweep/results.json",
                      "synthetic/rework_sweep_results.json"),
                     ("experiments/rework_concentration/results.json",
                      "synthetic/rework_concentration_results.json"),
                     ("experiments/rework_scan.json",
                      "synthetic/rework_scan.json")):
        if (REPO / src).exists():
            shutil.copy2(REPO / src, OUT / dst)
    (OUT / "synthetic/README.md").write_text(
        "# Synthetic generator\n\n"
        "Settings are the `config` object at the top of each results file "
        "(K, T, n_train, n_test, noise, strength, rework_len, lr, steps, "
        "reg_weight), and `rows` holds every per-seed outcome, including the "
        "measured ground-truth violation rate of each configuration. The "
        "generator itself is scripts/rework_sweep.py and "
        "scripts/rework_concentration.py; emissions are synthetic, so no images "
        "are involved and both files regenerate from the recorded settings.\n",
        encoding="utf-8")

    # ---- rerun discrepancies ----------------------------------------------
    for src in ("experiments/loso/discarded_batchsize2_results.jsonl",):
        if (REPO / src).exists():
            shutil.copy2(REPO / src, OUT / "rerun" / Path(src).name)
    pred_rerun = OUT / "predictions/rerun.jsonl"
    if pred_rerun.exists():
        shutil.copy2(pred_rerun, OUT / "rerun/rerun_full_grid.jsonl")
    (OUT / "rerun/notes.md").write_text(
        "# Reproduction margin\n\n"
        "Three separate reruns are disclosed here, none of which replaces a "
        "published number.\n\n"
        "1. `rerun_full_grid.jsonl` — every cell of the grid retrained by the "
        "published code path while dumping its predictions "
        "(`scripts/dump_all_predictions.py`). Each row carries the published and "
        "the reproduced accuracy and F1@50. Training reproduces to about four "
        "significant figures; the Viterbi decode does not always return the same "
        "path when two paths score within float noise, which is where the "
        "remaining differences come from.\n\n"
        "2. `discarded_batchsize2_results.jsonl` — a grid run at `--batch-size 2` "
        "by mistake. It reproduced only 6 of 39 cells and is archived, not used: "
        "accuracy moved 0.436 to 0.478 and one cell swung from 0.893 to 0.036. "
        "The published grid is batch-size 1, and that is now the code's default.\n\n"
        "3. The fixed-soft arm was run in a later session than the rest. "
        "Re-running the rigid arm under identical settings reproduced 32 of 39 "
        "cells exactly and moved its F1@50 by 0.007, which bounds the "
        "comparability of that row and is stated in the paper.\n",
        encoding="utf-8")

    # ---- CITATION.cff -----------------------------------------------------
    # Gives GitHub its "Cite this repository" button. No DOI, volume or pages:
    # the article is under review and does not have them yet.
    cff = ["cff-version: 1.2.0",
           "message: >-",
           "  If you use this package, please cite the article below rather",
           "  than the repository.",
           "title: \"Construction monotonicity: reproducibility package\"",
           f"date-released: \"{date.today().isoformat()}\"",
           "license: MIT",
           "type: dataset",
           "authors:"]
    for family, given, affil in AUTHORS:
        cff += [f"  - family-names: \"{family}\"",
                f"    given-names: \"{given}\"",
                f"    affiliation: \"{affil}\""]
    cff += ["preferred-citation:",
            "  type: article",
            f"  title: \"{PAPER_TITLE}\"",
            f"  journal: \"{PAPER_JOURNAL}\"",
            f"  year: {PAPER_YEAR}",
            f"  notes: \"{PAPER_STATUS}\"",
            "  authors:"]
    for family, given, affil in AUTHORS:
        cff += [f"    - family-names: \"{family}\"",
                f"      given-names: \"{given}\"",
                f"      affiliation: \"{affil}\""]
    (OUT / "CITATION.cff").write_text("\n".join(cff) + "\n", encoding="utf-8")
    print(f"CITATION.cff: {len(AUTHORS)} authors, status {PAPER_STATUS!r}")

    # ---- figures ----------------------------------------------------------
    # Derived artwork only. paper-fig0-stages is excluded on purpose: it is
    # built from real example frames, which the source videos' licence does not
    # let us redistribute. The assertion below is the guard, not this comment --
    # a photograph inside a vector figure is an embedded raster, so a figure
    # carrying one fails the build instead of being published by accident.
    import fitz

    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig_lines = []
    for stem, out_name in FIGURES_SHIPPED:
        pdf = REPO / "diagrams" / f"{stem}.pdf"
        if not pdf.exists():
            print(f"figures: MISSING {pdf}")
            return 1
        doc = fitz.open(pdf)
        rasters = sum(len(page.get_images(full=True)) for page in doc)
        if rasters:
            print(f"figures: REFUSED {stem} -- {rasters} embedded raster(s); "
                  f"a corpus frame may be in it")
            return 1
        doc[0].get_pixmap(dpi=FIG_DPI).save(str(OUT / "figures" / out_name))
        doc.close()
        fig_lines.append(f"{out_name}\tdiagrams/{stem}.pdf")
    # plot_results.py's --tas DEFAULT is experiments/tas, which holds the
    # SUPERSEDED ASFormer arm: running it plainly reproduces the stale figure.
    # The command that actually produced results.png is recorded here rather
    # than left for a reader to rediscover by getting a different picture.
    fig_header = [
        "# published figure <- source artwork",
        "#",
        "# results.png is NOT the default output of plot_results.py. Its",
        "# ASFormer arm comes from the corrected implementation, not the",
        "# superseded one that --tas points at by default. The two grids were",
        "# merged first:",
        "#",
        "#   mstcn    <- experiments/tas/results.jsonl              (published)",
        "#   asformer <- experiments/tas_asformer_fixed/results.jsonl (corrected)",
        "#",
        "# then:",
        "#   python scripts/plot_results.py \\",
        "#       --results experiments/loso/results.jsonl \\",
        "#       --tas <merged file> --out figures/results",
        "#",
        "# This is the same per-arm provenance that uncertainty.py's ARM_SOURCE",
        "# uses for every contrast reported in the paper.",
        "",
    ]
    (OUT / "figures/SOURCES.txt").write_text(
        "\n".join(fig_header + fig_lines) + "\n", encoding="utf-8")
    print(f"figures: {len(fig_lines)} rendered, 0 embedded rasters in any")

    # ---- scripts ----------------------------------------------------------
    missing = []
    lines = []
    for rel in SCRIPTS:
        src = REPO / rel
        if not src.exists():
            missing.append(rel)
            continue
        shutil.copy2(src, OUT / "scripts" / src.name)
        lines.append(f"{src.name}\t{rel}")
    (OUT / "scripts/SOURCES.txt").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")
    print(f"scripts: {len(lines)} copied"
          + (f"; MISSING {missing}" if missing else ""))

    # ---- the README, which is the artifact the reviewer actually reads -----
    n_pred_now = len(list((OUT / "predictions").glob("pred_*.json"))) \
        if (OUT / "predictions").exists() else 0
    env = {}
    try:
        import platform

        import torch
        import torchvision
        env = {"python": sys.version.split()[0], "torch": torch.__version__,
               "cuda": torch.version.cuda, "torchvision": torchvision.__version__,
               "gpu": (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else "CPU only"),
               "platform": platform.platform()}
    except Exception as exc:                         # pragma: no cover
        env = {"note": f"environment not recorded: {exc}"}

    (OUT / "README.md").write_text(f"""\
# Construction monotonicity: reproducibility package

Code, annotations and results for **"How much is construction monotonicity
worth? Rigid versus learnable sequence priors for image-based progress
recognition"** (International Journal of Intelligent Engineering and Systems).

Omar Amer Mohammed<sup>1</sup>, Jamal Mustafa Al-Tuwaijari<sup>1</sup>,
Amer Mohammed Al-Mahdawi<sup>2</sup>

<sup>1</sup> Department of Computer Science, College of Science, University of
Diyala, Iraq<br>
<sup>2</sup> Department of Civil Engineering, College of Engineering,
University of Diyala, Iraq

The paper asks a narrow question: a construction site normally moves forward,
so what is it worth to *tell* a stage-recognition model that? And is it better
to impose that ordering rigidly, or to let the model learn when to relax it?

![The method: a frozen ResNet18 feeds a sequence layer whose transition matrix
is the only thing varied, decoded by Viterbi. The three parameterisations are a
free CRF, a rigid monotonic prior, and a gate that can learn to reopen backward
transitions.](figures/method.png)

The backbone, the features and the training schedule are identical across arms.
**Only the transition matrix changes**, so a difference between arms is
attributable to the prior and not to capacity. Panel (C) shows what this buys:
the per-frame baseline decodes stage sequences that step backwards -- a
building that un-builds itself -- and the rigid arm cannot.

## What the study found

![Per-arm results over 13 held-out sites. Accuracy, macro-F1, edit score, F1@50
and stage-onset error, with the image-free control drawn as a dashed line
across every panel; below them the order-violation rate per arm and the
rigid-versus-gated contrast paired by site.](figures/results.png)

The dashed line in every panel is the image-free control. Where a bar does not
clear it, the arm has not been shown to use the images at all.

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

## The counterfactual the corpus cannot provide

![Synthetic sweep: as re-work is injected into generated sequences at a rising
rate, the rigid prior's advantage narrows and then inverts, while the gate --
now with backward transitions to learn from -- recovers.](figures/rework-sweep.png)

The real corpus contains no re-work, so it cannot show what either prior does
when a site goes backwards. The synthetic study supplies that: with re-work
injected at a known rate the gate does learn to reopen backward transitions,
which places the flat gate in the real runs on **this corpus** rather than on
the parameterisation. It is a generator, not a site, and is reported as such.

## Layout

```
corpus/       per-frame manifest, both annotation passes, exclusion masks,
              exact subsample indices, labelling protocols
config/       grid, preprocessing and baseline configurations
metrics/      per-run metrics for every arm, exactly as published
predictions/  {n_pred_now} decoded stage tracks, one per (arm, site, seed)
synthetic/    generator settings and every per-seed outcome
rerun/        three disclosed reproductions, including a discarded one
scripts/      every script that produces a table in the paper or supplement
figures/      the paper's derived figures (no photographs -- see below)
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

The figures in `figures/` are exempt because they contain no photograph: each
is rendered from a vector PDF, and `build_release.py` **fails the build** if
that PDF carries an embedded raster. The paper's stage-taxonomy figure, which
is built from real example frames, is for that reason not in this repository.

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
| splits, seeds, epochs | `config/grid.json` — leave-one-site-out, seeds 0-2, {grid['epochs']} epochs, no validation split |
| preprocessing | `config/preprocessing.json` |
| baseline configurations | `config/baselines.json` |
| per-run metrics | `metrics/loso_results.jsonl`, `metrics/per_site/loso_per_run.csv` |
| decoded predictions | `predictions/pred_<site>_<arm>_seed<n>.json` |
| synthetic studies | `synthetic/` |
| disclosed reruns | `rerun/notes.md` |
| figure-generation scripts | `scripts/plot_results.py`, `scripts/plot_concentration.py`, `scripts/rework_sweep.py`, `scripts/make_stage_figure.py` |
| blinded annotation study | `scripts/blind_pairs.py`, `scripts/blind_pairs_analysis.py` |

## Environment the published numbers were produced on

```json
{json.dumps(env, indent=1)}
```

## Citing this

Please cite the article rather than the repository; `CITATION.cff` carries the
same details in machine-readable form. The article is under review, so it has
no DOI, volume or page numbers yet — those are added here on acceptance.

## Licence

MIT for the code, annotations and derived results — see `LICENSE`, which names
the same three authors. The source videos are **not** covered by it.
""", encoding="utf-8")

    # ---- checksums over everything in the package -------------------------
    # Skip version-control and cache directories. Once the package was turned
    # into a git repository, .git's 327 internal files started appearing in the
    # manifest: they are not package content, they change on every commit, and
    # a manifest that churns against itself is worse than none.
    SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}
    entries = []
    for p in sorted(OUT.rglob("*")):
        if any(part in SKIP_DIRS for part in p.relative_to(OUT).parts):
            continue
        if p.is_file() and p.name != "CHECKSUMS.sha256":
            entries.append(f"{sha256(p)}  {p.relative_to(OUT).as_posix()}")
    (OUT / "CHECKSUMS.sha256").write_text("\n".join(entries) + "\n",
                                          encoding="utf-8")
    print(f"checksums: {len(entries)} files")

    n_pred = len(list((OUT / "predictions").glob("pred_*.json"))) \
        if (OUT / "predictions").exists() else 0
    print(f"\npredictions present: {n_pred} of 195 "
          f"(dump_all_predictions.py produces these)")
    print(f"package at {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
