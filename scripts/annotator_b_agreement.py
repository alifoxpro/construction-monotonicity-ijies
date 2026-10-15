"""Second-annotator pass over the full 13-sequence corpus, and the agreement it yields.

Annotator B is an AI vision-language system applying docs/labelling-protocol.md to
raw frames only. For every sequence, B wrote its marks before reading annotator
A's labels.csv. Sequences were labelled in two rounds: four in the first
(seq0003, seq0010, seq0012, seq0018, protocol v1.1) and the remaining nine in the
second (protocol v1.2, which added R2 for non-current footage).

This is human-vs-model agreement, NOT human-vs-human. It bounds protocol
ambiguity -- whether the written rules are specific enough that a second party
reaches the same labels -- and must be reported as such.

    python src/data_collection/annotator_b_agreement.py
"""
import csv
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RAW = REPO / "seq_data_raw"

# boundary marks: frame -> stage, every later frame inherits until the next mark
B_MARKS = {
    "seq0001": {0: -1, 8: 0, 44: 1, 84: 2},
    "seq0003": {0: 0, 12: 1, 45: 2, 82: 3, 164: 4},
    "seq0004": {0: -1, 3: 0, 5: 1, 13: 2, 34: 3, 58: 4},
    "seq0005": {0: 0, 6: 1, 11: 2, 94: 3, 325: 4},
    "seq0007": {0: -1, 3: 0, 17: 1, 45: 2, 130: 3, 199: 4},
    "seq0008": {0: 0, 22: 1},
    "seq0009": {0: 0},
    "seq0010": {0: 0, 23: 1, 27: 2, 46: 3, 71: 4, 97: -1},
    "seq0011": {0: 0, 2: 1, 5: 2, 16: 3, 18: 4, 20: -1},
    "seq0012": {0: 0, 12: 1, 42: 2, 75: 3},
    "seq0014": {0: 0, 3: 1, 27: 2, 31: 3, 36: 4},
    "seq0015": {0: 0, 7: 1, 10: 2, 13: 3},
    "seq0018": {0: -1, 2: 0, 20: 1, 35: 2, 81: 3, 85: 4,
                132: -1, 133: 4, 136: -1, 139: 4, 142: -1},
}

# Frames excluded on top of the boundary track. seq0007 captions every phase with
# a full-screen black card; these were located by detecting mostly-black frames,
# then verified by eye. 354-356 are the channel end card.
B_EXCLUDE = {
    "seq0007": [0, 1, 2, 57, 58, 109, 110, 152, 161, 175, 186, 198, 213, 233,
                250, 276, 285, 295, 306, 323, 332, 354, 355, 356],
}

ROUND1 = {"seq0003", "seq0010", "seq0012", "seq0018"}

MAX_LEN = 64            # run_loso.py --max-len default: every logged run used it


def propagate(seq, n):
    out, cur = [], None
    marks = B_MARKS[seq]
    for i in range(n):
        if i in marks:
            cur = marks[i]
        out.append(cur)
    for i in B_EXCLUDE.get(seq, []):
        if i < n:
            out[i] = -1
    return out


def read_a(seq):
    rows = csv.reader(open(RAW / seq / "labels.csv", newline="", encoding="utf-8"))
    return [int(r[1]) for r in rows if r]


def kappa(a, b):
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(L) / n) * (b.count(L) / n) for L in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def onsets(lab):
    return {s: next((i for i, v in enumerate(lab) if v == s), None) for s in range(5)}


def violations(lab):
    p = [(x, y) for x, y in zip(lab, lab[1:]) if x >= 0 and y >= 0]
    return sum(1 for x, y in p if y < x), len(p)


def main():
    print(f"{'seq':9s} {'rnd':>3s} {'frames':>6s} {'cmp':>5s} {'agree':>6s} {'kappa':>7s} "
          f"{'onset':>6s}   violations A / B")
    print("-" * 78)

    all_a, all_b, onset_all = [], [], []
    for seq in sorted(B_MARKS):
        a = read_a(seq)
        b = propagate(seq, len(a))
        (RAW / seq / "labels.B.csv").write_text(
            "".join(f"frames/{i:06d}.jpg,{v}\n" for i, v in enumerate(b)),
            encoding="utf-8")

        both = [(x, y) for x, y in zip(a, b) if x >= 0 and y >= 0]
        all_a += [x for x, _ in both]
        all_b += [y for _, y in both]

        oa, ob = onsets(a), onsets(b)
        d = [abs(ob[s] - oa[s]) for s in range(5)
             if oa[s] is not None and ob[s] is not None]
        onset_all += d
        va, na = violations(a)
        vb, nb = violations(b)

        agr = sum(x == y for x, y in both) / len(both) if both else float("nan")
        k = kappa([x for x, _ in both], [y for _, y in both])
        rnd = "1" if seq in ROUND1 else "2"
        om = f"{sum(d)/len(d):.1f}" if d else "--"
        print(f"{seq:9s} {rnd:>3s} {len(a):6d} {len(both):5d} {agr:6.3f} {k:7.3f} "
              f"{om:>6s}   {va}/{na}  {vb}/{nb}")

    print("-" * 78)
    agr = sum(x == y for x, y in zip(all_a, all_b)) / len(all_a)
    print(f"{'POOLED':9s} {'':3s} {'':6s} {len(all_a):5d} {agr:6.3f} "
          f"{kappa(all_a, all_b):7.3f} {sum(onset_all)/len(onset_all):6.1f}")
    print(f"\n  compared frames  : {len(all_a)}")
    print(f"  disagreements    : {sum(x != y for x, y in zip(all_a, all_b))}")
    print(f"  onset diffs      : n={len(onset_all)}  mean={sum(onset_all)/len(onset_all):.2f}  "
          f"median={sorted(onset_all)[len(onset_all)//2]}  max={max(onset_all)}")

    print("\nconfusion (rows = A, cols = B):")
    print("      " + "".join(f"{j:>7d}" for j in range(5)))
    for i in range(5):
        print(f"  {i}   " + "".join(
            f"{sum(1 for x, y in zip(all_a, all_b) if x == i and y == j):7d}"
            for j in range(5)))

    tot_a = tot_b = tot_p = 0
    for seq in sorted(B_MARKS):
        va, na = violations(read_a(seq))
        vb, nb = violations(propagate(seq, len(read_a(seq))))
        tot_a += va
        tot_b += vb
        tot_p += na
    print(f"\ncorpus order-violation rate:  A {tot_a}/{tot_p}   B {tot_b}/{tot_p} (B pairs differ slightly)")

    # The onset figures above are in RAW frame indices. The trained models are not
    # scored on those: evaluation keeps A's labelled frames and subsamples each
    # sequence evenly to MAX_LEN (SequenceFrameDataset). Comparing a model's onset
    # MAE with the raw-frame disagreement mixed two units -- up to 13x apart on the
    # longest sequence. This measures disagreement on the frames the models see.
    print(f"\n=== onset disagreement in EVALUATION units (A-labelled frames, "
          f"subsampled to {MAX_LEN}) ===")
    per_seq, per_onset = [], []
    for seq in sorted(B_MARKS):
        a_raw = read_a(seq)
        b_raw = propagate(seq, len(a_raw))
        keep = [i for i, v in enumerate(a_raw) if v >= 0]
        if len(keep) > MAX_LEN:
            step = (len(keep) - 1) / (MAX_LEN - 1)
            keep = [keep[round(i * step)] for i in range(MAX_LEN)]
        oa = onsets([a_raw[i] for i in keep])
        ob = onsets([b_raw[i] for i in keep])
        d = [abs(ob[s] - oa[s]) for s in range(5)
             if oa[s] is not None and ob[s] is not None]
        per_onset += d
        per_seq.append(sum(d) / len(d) if d else 0.0)
        print(f"  {seq}  {len(keep):3d} frames  mean onset disagreement "
              f"{per_seq[-1]:5.2f}")
    q = sorted(per_seq)
    print(f"  per sequence: mean {sum(q)/len(q):.2f}  median {q[len(q)//2]:.2f}  "
          f"Q3 {q[(3*len(q))//4]:.2f}  max {q[-1]:.2f}")
    print(f"  per onset   : n={len(per_onset)}  mean {sum(per_onset)/len(per_onset):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
