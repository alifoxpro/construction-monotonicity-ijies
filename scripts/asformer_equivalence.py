"""Is our ASFormer the published ASFormer? — IJIES reviewer 2, point 4.

THE OBJECTION, verbatim: "Check the custom ASFormer implementation against the
reference implementation on a public benchmark or provide a focused equivalence
check."

Running the reference on a public benchmark would compare two models on data
this paper does not use, and would answer a question nobody asked: whether the
AUTHORS' code reproduces the AUTHORS' numbers. The question that matters here is
whether the arm labelled "ASFormer" in our tables is the architecture that name
denotes. That is settled by comparing our module against the reference
implementation (github.com/ChinaYi/ASFormer, model.py) component by component,
and then TESTING the differences that a reader cannot verify by eye.

The structural comparison is tabulated below from the reference source. The
behavioural checks that follow are the point of the script: they probe
properties the architecture is defined by, independent of any trained weights,
so a divergence shows up as a measurement rather than as a reading of the code.

THE DECISIVE ONE IS ATTENTION LOCALITY. The reference computes attention in
blocks of `bl = 2**i` with a halo of `bl // 2` padded on each side, so a query
near a block edge attends across that edge and information flows between blocks
within a single layer. Our module splits the sequence into NON-OVERLAPPING
windows of `2 * dilation` and attends only inside each one. If that is a real
difference and not a reading error, perturbing a frame just across a window
boundary will change the reference's output at a given position and will not
change ours. The check below measures exactly that.

    python src/experiments/asformer_equivalence.py
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

from models.tas_encoders import ASFormer, AttLayer, exponential_decrease  # noqa: E402

#: (component, reference behaviour, source of the claim, ours)
#: Reference = github.com/ChinaYi/ASFormer, model.py, as read on 2026-10-14.
COMPARISON = [
    ("attention block size", "bl = 2**i, equal to the layer's dilation",
     "AttModule(2 ** i, ...); self.bl = bl", "same"),
    ("key/value window",
     "block plus a halo of bl//2 on each side, span bl + 2*(bl//2), so query t "
     "attends to keys [t - bl//2, t - bl//2 + bl)",
     "k = cat([zeros(bl//2), k, zeros(bl//2)]); spans of bl + 2*(bl//2); "
     "window_mask row i opens columns [i, i+bl)", "same"),
    ("decoder cross-attention scaling", "alpha = exp(-3 * decoder_index)",
     "alpha=exponential_descrease(s)", "same"),
    ("what the decoder takes from the encoder",
     "the VALUE only; query and key come from the decoder's own input",
     "value = self.value_conv(x2) if decoder else self.value_conv(x1)",
     "same"),
    ("residual structure",
     "one residual path: ff -> alpha*att(norm(ff)) + ff -> 1x1 -> dropout -> +x",
     "out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out",
     "same"),
    ("q/k/v width", "reduced by r1, r1, r2 (defaults 2, 2, 2)",
     "AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, ...)",
     "same"),
    ("attention heads", "1 (MultiHeadAttLayer is defined but unused)",
     "AttModule instantiates AttLayer directly", "1"),
    ("normalisation before attention",
     "InstanceNorm1d(track_running_stats=False), affine defaults to False",
     "nn.InstanceNorm1d(in_channels, track_running_stats=False)", "same"),
    ("dropout in the attention module", "0.5 (nn.Dropout() default)",
     "self.dropout = nn.Dropout()", "same"),
    ("input channel masking", "Dropout2d(channel_masking_rate), 0.3 in main.py",
     "if self.channel_masking_rate > 0: x = self.dropout(x.unsqueeze(2))",
     "same"),
    ("layers / maps / decoders", "10 layers, 64 maps, 3 decoders",
     "num_layers = 10; num_f_maps = 64; MyTransformer(3, ...)", "same"),
    ("masked attention", "additive in log space: + log(mask + 1e-6)",
     "attention = attention + torch.log(padding_mask + 1e-6)", "same"),
    ("batch size on the sliding path", "asserts batch size 1",
     "assert m_batchsize == 1",
     "any batch size; the grid runs at batch size 1 regardless"),
]


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * len(text))


def check_locality(ch: int = 16, T: int = 64, dilation: int = 8) -> dict:
    """Does a perturbation cross a block boundary within one attention layer?

    The reference pads a halo of bl//2 on each side of every block, so a query
    near a block edge attends across it. Perturb the last frame of the first
    block: positions in the NEXT block that lie within bl//2 of the boundary
    should respond. If none do, the halo is missing and the layer is not the
    reference's -- which is exactly what this check reported before the rewrite.
    """
    torch.manual_seed(0)
    att = AttLayer(ch, ch, ch, 2, 2, 2, dilation, "encoder").eval()
    mask = torch.ones(1, 1, T)
    x = torch.randn(1, ch, T)
    with torch.no_grad():
        base = att(x, x, mask)

    bl = dilation
    pos = bl - 1                      # last frame of the first block
    x2 = x.clone()
    x2[0, :, pos] += 5.0
    with torch.no_grad():
        pert = att(x2, x2, mask)

    delta = (pert - base).abs().sum(dim=1)[0]
    # Threshold at 1e-3, not at float epsilon. The reference masks additively in
    # log space (+ log(mask + 1e-6)) rather than with -inf, so a disallowed key
    # keeps about 1e-6 of relative weight and positions well outside the window
    # move by ~2e-06. That leak is the reference's own behaviour, not an
    # indexing error, and counting it as "moved" would hide the real structure.
    moved = (delta > 1e-3).nonzero().flatten().tolist()
    leak = [q for q in (delta > 1e-9).nonzero().flatten().tolist()
            if q not in moved]
    own = [q for q in moved if q // bl == pos // bl]
    other = [q for q in moved if q // bl != pos // bl]
    expected = [t for t in range(T) if t - bl // 2 <= pos < t + bl // 2]
    return {"block_len": bl, "halo": bl // 2, "perturbed_position": pos,
            "positions_moved": len(moved), "moved_positions": moved,
            "moved_in_own_block": len(own),
            "moved_in_other_blocks": len(other),
            "crosses_block_boundary": bool(other),
            "other_block_positions": other[:8],
            "expected_window": expected,
            "matches_expected_window": moved == expected,
            "soft_mask_leak_positions": leak,
            "soft_mask_leak_max": float(delta[leak].max()) if leak else 0.0}


def check_receptive_field(num_layers: int = 9, ch: int = 16, T: int = 256,
                          K: int = 5) -> dict:
    """How far does one frame's influence reach through a whole encoder stage?

    The dilated convolutions alone give a receptive field of 1 + 2*sum(2**i).
    Attention inside windows cannot extend it beyond the widest window. This
    measures the reach actually realised, which is the quantity a reader would
    want to compare against the reference's.
    """
    torch.manual_seed(0)
    model = ASFormer(ch, K, num_decoders=0, num_layers=num_layers, ch=32,
                     channel_masking_rate=0.0).eval()
    feats = torch.zeros(1, T, ch)
    mask = torch.ones(1, T)
    with torch.no_grad():
        base = model(feats, mask)
    feats2 = feats.clone()
    feats2[0, T // 2, :] = 10.0
    with torch.no_grad():
        pert = model(feats2, mask)
    delta = (pert - base).abs().sum(dim=-1)[0]
    moved = (delta > 1e-6).nonzero().flatten()
    conv_rf = 1 + 2 * sum(2 ** i for i in range(num_layers))
    if moved.numel() == 0:
        return {"measured_span": 0, "conv_only_receptive_field": conv_rf}
    span = int(moved.max() - moved.min()) + 1
    return {"measured_span": span, "conv_only_receptive_field": conv_rf,
            "reaches_sequence_edge": bool(moved.min() == 0 or moved.max() == T - 1)}


def check_masking(ch: int = 16, T: int = 32, K: int = 5) -> dict:
    """Do padded positions influence real ones?

    Every arm is scored on variable-length sequences padded to the batch max, so
    a leak from padding into real frames would corrupt the comparison this paper
    rests on, whatever the architecture.
    """
    torch.manual_seed(0)
    model = ASFormer(ch, K, num_decoders=1, num_layers=4, ch=32,
                     channel_masking_rate=0.0).eval()
    real = 20
    mask = torch.zeros(1, T)
    mask[0, :real] = 1.0
    feats = torch.randn(1, T, ch)
    with torch.no_grad():
        base = model(feats, mask)
    feats2 = feats.clone()
    feats2[0, real:, :] += 100.0          # garbage in the padded region
    with torch.no_grad():
        pert = model(feats2, mask)
    leak = (pert[0, :real] - base[0, :real]).abs().max().item()
    return {"real_frames": real, "padded_frames": T - real,
            "max_change_in_real_region": leak, "leaks": leak > 1e-5}


def check_stage_supervision(ch: int = 16, T: int = 32, K: int = 5) -> dict:
    """Are all stages exposed for supervision, as the published design requires?"""
    torch.manual_seed(0)
    model = ASFormer(ch, K, num_decoders=3, num_layers=3, ch=32,
                     channel_masking_rate=0.0).eval()
    with torch.no_grad():
        model(torch.randn(1, T, ch), torch.ones(1, T))
    return {"stage_logits_exposed": len(model.stage_logits),
            "expected": 4, "ok": len(model.stage_logits) == 4}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="experiments/asformer_equivalence.json")
    args = ap.parse_args()

    banner("STRUCTURAL COMPARISON against github.com/ChinaYi/ASFormer model.py")
    diffs = []
    for comp, ref, src, ours in COMPARISON:
        same = ours.strip().lower() in ("same", "1", "3")
        flag = "same " if same else "DIFFERS"
        if not same:
            diffs.append(comp)
        print(f"\n[{flag}] {comp}")
        print(f"    reference : {ref}")
        print(f"    source    : {src}")
        print(f"    ours      : {ours}")
    print(f"\n{len(diffs)} of {len(COMPARISON)} components differ: "
          f"{', '.join(diffs)}")

    banner("BEHAVIOURAL CHECK 1 — does attention cross a block boundary?")
    loc = check_locality()
    print(f"  block length / halo           : {loc['block_len']} / {loc['halo']}")
    print(f"  perturbed position            : {loc['perturbed_position']} "
          f"(last frame of the first block)")
    print(f"  output positions that moved   : {loc['positions_moved']}")
    print(f"    inside the same block       : {loc['moved_in_own_block']}")
    print(f"    in other blocks             : {loc['moved_in_other_blocks']} "
          f"at {loc['other_block_positions']}")
    print(f"  crosses a block boundary      : {loc['crosses_block_boundary']}")
    print(f"  positions that moved          : {loc['moved_positions']}")
    print(f"  window the indexing implies   : {loc['expected_window']}")
    print(f"  the two agree                 : {loc['matches_expected_window']}")
    print(f"  soft-mask leak outside it     : "
          f"{len(loc['soft_mask_leak_positions'])} position(s), at most "
          f"{loc['soft_mask_leak_max']:.1e}")
    print("\n  The reference pads a halo of bl//2 on each side of every block, so")
    print("  a query at a block edge attends across it. Before the rewrite this")
    print("  read False, which is what showed the two were not the same layer.")
    print("  The leak is the reference's additive log masking, which leaves a")
    print("  disallowed key about 1e-6 of relative weight rather than zero.")

    banner("BEHAVIOURAL CHECK 2 — receptive field of one encoder stage")
    rf = check_receptive_field()
    print(f"  conv-only receptive field     : "
          f"{rf['conv_only_receptive_field']} frames")
    print(f"  measured span of influence    : {rf['measured_span']} frames")
    print(f"  reaches the sequence edge     : "
          f"{rf.get('reaches_sequence_edge')}")

    banner("BEHAVIOURAL CHECK 3 — do padded frames leak into real ones?")
    mk = check_masking()
    print(f"  {mk['real_frames']} real frames, {mk['padded_frames']} padded")
    print(f"  largest change in the real region after corrupting the padding: "
          f"{mk['max_change_in_real_region']:.3e}")
    print(f"  leaks : {mk['leaks']}")

    banner("BEHAVIOURAL CHECK 4 — every stage exposed for supervision")
    ss = check_stage_supervision()
    print(f"  stage logits exposed : {ss['stage_logits_exposed']} "
          f"(expected {ss['expected']}) -> {'ok' if ss['ok'] else 'WRONG'}")

    banner("VERDICT")
    n_real = len([c for c in diffs if c != "batch size on the sliding path"])
    equivalent = (n_real == 0 and loc["matches_expected_window"]
                  and not mk["leaks"] and ss["ok"])
    if equivalent:
        print("Our module matches the authors' released implementation on every")
        print("component that affects what it computes. The one remaining entry")
        print("in the table is the reference's assertion that the batch size is")
        print("1, a restriction we do not impose; the published grid runs at")
        print("batch size 1 regardless, so no result depends on it.")
        print()
        print("The sliding window reproduces the indexing exactly: a key")
        print("influences precisely the queries the halo construction implies,")
        print("and nothing beyond the reference's own soft-masking leak.")
        print()
        print("What is still NOT claimed: this is not a reproduction of the")
        print("authors' numbers on their benchmarks. It is their architecture at")
        print("their defaults, trained on our corpus under our protocol.")
    else:
        print(f"{n_real} component(s) still differ in a way that affects the")
        print("computation, and the arm cannot be labelled ASFormer until they")
        print("are resolved or stated:")
        for c in diffs:
            if c != "batch size on the sliding path":
                print(f"  - {c}")

    out = Path(REPO / args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "reference": "github.com/ChinaYi/ASFormer model.py, read 2026-10-14",
        "components": [{"component": c, "reference": r, "source": s, "ours": o,
                        "differs": c in diffs}
                       for c, r, s, o in COMPARISON],
        "n_differing": len(diffs),
        "locality": loc, "receptive_field": rf,
        "masking": mk, "stage_supervision": ss,
    }, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
