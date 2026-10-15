"""
Evaluation metrics for construction-stage sequences.

Per-frame accuracy alone hides the failure mode this work cares about: a model
can score well per frame while chopping a clean 3-segment sequence into 40 noisy
fragments. The temporal-action-segmentation literature standardized on segmental
metrics for exactly that reason, and reviewers in that area expect them.

Implemented here
----------------
* ``macro_f1``        -- per-frame F1 averaged over classes (not frames), so rare
                         early stages count as much as the long finishing stage.
* ``segmental_f1``    -- F1@{10,25,50} (Lea et al.; the MS-TCN/ASFormer standard):
                         segment-level matching by temporal IoU, tolerant of
                         boundary jitter but punishing over-segmentation.
* ``edit_score``      -- normalized Levenshtein distance over the *sequence of
                         segment labels*, ignoring durations entirely; the purest
                         measure of whether the predicted ORDER is right.
* ``onset_mae``       -- domain metric: mean absolute error, in frames, of the
                         first frame assigned to each stage. "How late did we
                         notice the roof went on?"

Conventions (documented because they differ between papers)
-----------------------------------------------------------
* All values are fractions in [0, 1], NOT percentages. Multiply by 100 for the
  paper's tables.
* ``segmental_f1`` accumulates TP/FP/FN across ALL sequences and computes F1 once
  (the MS-TCN convention), rather than averaging per-sequence F1.
* ``edit_score`` is averaged over sequences (also the MS-TCN convention).
* ``onset_mae`` is measured in FRAMES, over stages present in both the gold and
  the prediction; stages the model never predicts are counted separately in
  ``onset_missing`` rather than being silently dropped or given an arbitrary
  penalty.
"""

from __future__ import annotations

Sequence = list[int]


def segments_from_labels(labels: Sequence) -> list[tuple[int, int, int]]:
    """Collapse a frame-label sequence into ``(label, start, end_inclusive)``."""
    if not labels:
        return []
    segs, start = [], 0
    for t in range(1, len(labels)):
        if labels[t] != labels[t - 1]:
            segs.append((labels[start], start, t - 1))
            start = t
    segs.append((labels[start], start, len(labels) - 1))
    return segs


def macro_f1(preds: list[Sequence], golds: list[Sequence]) -> float:
    """Per-frame F1 averaged over classes present in gold or prediction."""
    tp: dict[int, int] = {}
    fp: dict[int, int] = {}
    fn: dict[int, int] = {}
    classes: set[int] = set()
    for p_seq, g_seq in zip(preds, golds):
        for p, g in zip(p_seq, g_seq):
            classes.update((p, g))
            if p == g:
                tp[p] = tp.get(p, 0) + 1
            else:
                fp[p] = fp.get(p, 0) + 1
                fn[g] = fn.get(g, 0) + 1
    if not classes:
        return 0.0
    total = 0.0
    for c in classes:
        t, f_p, f_n = tp.get(c, 0), fp.get(c, 0), fn.get(c, 0)
        prec = t / (t + f_p) if (t + f_p) else 0.0
        rec = t / (t + f_n) if (t + f_n) else 0.0
        total += (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return total / len(classes)


def _temporal_iou(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    inter = min(a[2], b[2]) - max(a[1], b[1]) + 1
    if inter <= 0:
        return 0.0
    union = max(a[2], b[2]) - min(a[1], b[1]) + 1
    return inter / union


def segmental_f1(
    preds: list[Sequence], golds: list[Sequence],
    thresholds: tuple[float, ...] = (0.10, 0.25, 0.50),
) -> dict[str, float]:
    """F1@k by segment matching on temporal IoU (Lea et al.).

    A predicted segment is a true positive if it shares a label with an unmatched
    gold segment at IoU > k. Each gold segment can be claimed once, so splitting
    one true segment into five predictions yields 1 TP and 4 FP -- this is what
    makes the metric punish over-segmentation.
    """
    out: dict[str, float] = {}
    for k in thresholds:
        tp = fp = fn = 0
        for p_seq, g_seq in zip(preds, golds):
            p_segs = segments_from_labels(p_seq)
            g_segs = segments_from_labels(g_seq)
            used = [False] * len(g_segs)
            for ps in p_segs:
                best_iou, best_j = 0.0, -1
                for j, gs in enumerate(g_segs):
                    if gs[0] != ps[0] or used[j]:
                        continue
                    iou = _temporal_iou(ps, gs)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_j >= 0 and best_iou > k:
                    used[best_j] = True
                    tp += 1
                else:
                    fp += 1
            fn += used.count(False)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        out[f"f1@{int(k * 100)}"] = f1
    return out


def _levenshtein(a: list[int], b: list[int]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def edit_score(preds: list[Sequence], golds: list[Sequence]) -> float:
    """Mean normalized segment-level edit similarity, in [0, 1].

    Durations are discarded: only the ORDER of predicted stages is compared. For
    a monotonicity claim this is the most direct summary -- a model that gets the
    sequence 0->1->2->3 right but the boundaries wrong still scores 1.0.
    """
    if not preds:
        return 0.0
    scores = []
    for p_seq, g_seq in zip(preds, golds):
        p = [s[0] for s in segments_from_labels(p_seq)]
        g = [s[0] for s in segments_from_labels(g_seq)]
        denom = max(len(p), len(g))
        scores.append(1.0 if denom == 0 else 1.0 - _levenshtein(p, g) / denom)
    return sum(scores) / len(scores)


def onset_mae(preds: list[Sequence], golds: list[Sequence]) -> tuple[float, int, float]:
    """(mean absolute onset error in frames, stages never predicted, coverage).

    Onset = first frame assigned to a stage. Stages absent from the prediction
    have no defined error, so they are counted rather than imputed -- a model
    that skips a stage entirely must not look accurate on the stages it kept.

    COVERAGE is the fraction of gold stages the error is actually computed over,
    and it must be read next to the error, never without it. The error averages
    only the stages the model did predict, so dropping a stage REMOVES a term
    instead of penalising it: a model that predicts one easy stage and misses
    the rest can report a near-zero onset error, and a model that predicts
    nothing reports NaN and then vanishes from any mean taken over runs. Both
    occur in this corpus. Added 2026-10-14 for IJIES reviewer 2, who asked for
    coverage beside every error for exactly this reason.
    """
    errors: list[int] = []
    missing = 0
    total = 0
    for p_seq, g_seq in zip(preds, golds):
        for stage in sorted(set(g_seq)):
            total += 1
            g_onset = g_seq.index(stage)
            if stage in p_seq:
                errors.append(abs(p_seq.index(stage) - g_onset))
            else:
                missing += 1
    coverage = (len(errors) / total) if total else float("nan")
    return (sum(errors) / len(errors) if errors else float("nan")), missing, coverage


def compute_all(preds: list[Sequence], golds: list[Sequence]) -> dict[str, float]:
    """Every metric in one pass, ready to drop into a result record."""
    mae, missing, coverage = onset_mae(preds, golds)
    out: dict[str, float] = {
        "macro_f1": macro_f1(preds, golds),
        "edit": edit_score(preds, golds),
        "onset_mae_frames": mae,
        "onset_missing": missing,
        "onset_coverage": coverage,
    }
    out.update(segmental_f1(preds, golds))
    return out


if __name__ == "__main__":
    # ---- self-test on hand-checkable cases ----------------------------------
    perfect_g = [[0, 0, 1, 1, 2, 2]]
    assert segments_from_labels(perfect_g[0]) == [(0, 0, 1), (1, 2, 3), (2, 4, 5)]
    m = compute_all(perfect_g, perfect_g)
    assert m["macro_f1"] == 1.0 and m["edit"] == 1.0 and m["f1@50"] == 1.0
    assert m["onset_mae_frames"] == 0.0 and m["onset_missing"] == 0
    print("[ok] perfect prediction scores 1.0 everywhere")

    # over-segmentation: same frame labels flicker -> accuracy stays high,
    # segmental metrics must collapse. This is the case per-frame accuracy hides.
    gold = [[0] * 10 + [1] * 10]
    flick = [[0, 1] * 5 + [1, 0] * 5]
    acc = sum(p == g for p, g in zip(flick[0], gold[0])) / len(gold[0])
    mf = compute_all(flick, gold)
    print(f"[ok] flickering: per-frame acc {acc:.2f} but edit {mf['edit']:.2f}, "
          f"f1@10 {mf['f1@10']:.2f}")
    assert acc >= 0.5 and mf["edit"] < 0.2 and mf["f1@10"] < 0.3

    # boundary jitter: right order, late boundary -> edit perfect, onset MAE > 0
    late = [[0] * 13 + [1] * 7]
    mj = compute_all(late, gold)
    assert mj["edit"] == 1.0 and mj["onset_mae_frames"] == 1.5
    print(f"[ok] boundary jitter: edit {mj['edit']:.2f}, "
          f"onset MAE {mj['onset_mae_frames']:.1f} frames")

    # a skipped stage is counted, not hidden
    skip_g = [[0] * 4 + [1] * 4 + [2] * 4]
    skip_p = [[0] * 6 + [2] * 6]
    ms = compute_all(skip_p, skip_g)
    assert ms["onset_missing"] == 1
    assert abs(ms["onset_coverage"] - 2 / 3) < 1e-9
    print(f"[ok] skipped stage: onset_missing {ms['onset_missing']}, "
          f"coverage {ms['onset_coverage']:.2f}, macro_f1 {ms['macro_f1']:.2f}")

    # The case reviewer 2 describes: dropping a stage REMOVES a term from the
    # error instead of penalising it, so a worse model can post a better onset
    # error. Only coverage reveals it, which is why the two must be read as a
    # pair. Here the skipping model's surviving onsets are exact.
    # every stage present but each boundary 2 frames late
    keeps_all = [[0] * 6 + [1] * 4 + [2] * 2]
    ma, mb = compute_all(keeps_all, skip_g), compute_all(skip_p, skip_g)
    assert mb["onset_mae_frames"] < ma["onset_mae_frames"]
    assert mb["onset_coverage"] < ma["onset_coverage"]
    print(f"[ok] onset error alone is misleading: the model that SKIPS a stage "
          f"scores {mb['onset_mae_frames']:.2f} frames at coverage "
          f"{mb['onset_coverage']:.2f}, better than the model that keeps every "
          f"stage ({ma['onset_mae_frames']:.2f} frames at coverage "
          f"{ma['onset_coverage']:.2f})")
    print("\nall metric self-tests passed")
