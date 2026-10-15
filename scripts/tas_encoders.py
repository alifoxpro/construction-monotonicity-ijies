"""Temporal action segmentation baselines: MS-TCN and ASFormer.

Why these exist here
--------------------
Section 2.3 of the paper positions this work against the temporal action
segmentation literature and borrows its metrics (segmental F1@k, edit score).
Citing that literature while benchmarking only against an LSTM we wrote
ourselves is a fair thing for a reviewer to object to, so both standard models
are implemented and run on the same corpus, under the same backbone, the same
folds and the same protocol as every other arm.

They occupy the SAME slot as the LSTM baseline: a temporal model over per-frame
backbone features, carrying no construction-domain prior. That is what makes the
comparison informative -- it separates "a strong sequence architecture helps"
from "the domain prior helps".

Interface
---------
Unlike the LSTM encoder, these two emit class logits directly rather than hidden
features, because their published designs are defined that way: MS-TCN refines a
class-probability map stage by stage, and ASFormer's decoders do the same. The
model therefore sets `classifier = Identity` for them, and `out_dim = K`.

Faithfulness, stated plainly
----------------------------
MS-TCN follows Farha and Gall (2019) closely: dilated residual layers with
exponentially growing dilation, multi-stage refinement over softmax outputs, and
the truncated-MSE smoothing loss with their published weights (lambda = 0.15,
tau = 4, i.e. clamp at 16 in squared terms).

ASFormer follows Yi, Wen and Jiang (2021) AND the authors' released code
(github.com/ChinaYi/ASFormer, model.py), component by component: the sliding
window with its halo, the value-only cross-attention, the single residual path,
the exponentially decreasing decoder alpha, and their own default
hyper-parameters (10 layers, 64 maps, r1 = r2 = 2, channel masking 0.3).

It did not always. An earlier version here was written from the paper's prose
alone, and src/experiments/asformer_equivalence.py found seven of ten
components differing from the released code -- including non-overlapping
attention windows, which let no information cross a window edge inside a layer.
That script is kept and re-run after this rewrite; its purpose is to make the
claim checkable rather than to be believed.

What is still NOT claimed: this is not a reproduction of the authors' published
numbers on their benchmarks. It is their architecture, at their defaults,
trained on our corpus under the same protocol as every other arm.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# MS-TCN
# --------------------------------------------------------------------------- #
class DilatedResidualLayer(nn.Module):
    """conv(k=3, dilated) -> ReLU -> conv(1x1) -> dropout, added to the input."""

    def __init__(self, dilation: int, ch: int, dropout: float = 0.5):
        super().__init__()
        self.conv_dilated = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(ch, ch, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.dropout(self.conv_1x1(out))
        # padded frames must not leak through the residual either
        return (x + out) * mask


class SingleStageTCN(nn.Module):
    """One MS-TCN stage: in_ch -> num_f -> ... -> K logits."""

    def __init__(self, num_layers: int, num_f: int, in_ch: int, num_classes: int):
        super().__init__()
        self.conv_in = nn.Conv1d(in_ch, num_f, 1)
        self.layers = nn.ModuleList(
            [DilatedResidualLayer(2 ** i, num_f) for i in range(num_layers)]
        )
        self.conv_out = nn.Conv1d(num_f, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_in(x) * mask
        for layer in self.layers:
            out = layer(out, mask)
        return self.conv_out(out) * mask


class MSTCN(nn.Module):
    """Multi-stage TCN (Farha & Gall, CVPR 2019).

    Returns the final stage's logits; the intermediate stages' logits are kept
    on `.stage_logits` so the training loss can supervise every stage, which is
    the point of the multi-stage design and not an optional extra.
    """

    def __init__(self, in_dim: int, num_classes: int, num_stages: int = 4,
                 num_layers: int = 10, num_f: int = 64):
        super().__init__()
        self.stage1 = SingleStageTCN(num_layers, num_f, in_dim, num_classes)
        self.stages = nn.ModuleList([
            copy.deepcopy(SingleStageTCN(num_layers, num_f, num_classes, num_classes))
            for _ in range(num_stages - 1)
        ])
        self.stage_logits: list[torch.Tensor] = []

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T, K), with mask (B, T)."""
        x = feats.transpose(1, 2)                     # (B, D, T)
        m = mask.unsqueeze(1)                         # (B, 1, T)
        out = self.stage1(x, m)
        self.stage_logits = [out.transpose(1, 2)]
        for stage in self.stages:
            out = stage(F.softmax(out, dim=1) * m, m)
            self.stage_logits.append(out.transpose(1, 2))
        return out.transpose(1, 2)                    # (B, T, K)


# --------------------------------------------------------------------------- #
# ASFormer
# --------------------------------------------------------------------------- #
# Rewritten 2026-10-14 to match the authors' released implementation
# (github.com/ChinaYi/ASFormer, model.py) rather than only the paper's prose.
# The previous version was an independent reading of the architecture, and
# src/experiments/asformer_equivalence.py found seven of ten components
# differing from the released code -- two of them functional:
#
#   * attention ran in NON-OVERLAPPING windows of 2*dilation, so no information
#     crossed a window edge inside a layer. The reference slides a window of
#     `bl = dilation` keys centred on each query, implemented as blocks of `bl`
#     with a halo of `bl // 2` padded on each side.
#   * the decoders' exponentially decreasing cross-attention weight,
#     alpha = exp(-3 * decoder_index), was absent.
#
# and three more that shape capacity rather than wiring: cross-attention took
# BOTH key and value from the encoder where the reference takes only the value;
# the residual structure was transformer-style (two residual blocks) where the
# reference has one; and q/k/v were full width where the reference reduces them
# by r1 and r2.
#
# A baseline implemented with less information flow than its published form
# understates that baseline, and this paper compares itself against it, so the
# divergences are corrected here rather than documented and kept.
#
# Hyper-parameters below are the reference's own defaults (main.py):
# num_layers 10, num_f_maps 64, r1 = r2 = 2, channel_masking_rate 0.3,
# three decoders. They are NOT tuned on this corpus.


def exponential_decrease(idx_decoder: int, p: float = 3.0) -> float:
    """alpha for decoder `idx_decoder`: 1, 0.0498, 0.00248 for p = 3."""
    return math.exp(-p * idx_decoder)


class ConvFeedForward(nn.Module):
    """Dilated conv + ReLU, the reference's per-layer feed-forward."""

    def __init__(self, dilation: int, in_ch: int, out_ch: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class AttLayer(nn.Module):
    """Sliding-window attention over a window of `bl` keys centred on each query.

    The reference computes this blockwise: queries are chunked into blocks of
    `bl`, keys and values are padded by `bl // 2` on each side and chunked into
    overlapping spans of `bl + 2 * (bl // 2)`, and a fixed window mask lets row
    i of a block see columns [i, i + bl) of its span. Unpacking the indices,
    query t attends to keys [t - bl//2, t - bl//2 + bl). We reproduce that
    construction rather than a window of our own, so the two can be compared.

    In a decoder only the VALUE comes from the encoder: query and key are both
    taken from the decoder's own input. Taking the key from the encoder as well
    is a different layer, and was one of the divergences corrected here.
    """

    def __init__(self, q_dim: int, k_dim: int, v_dim: int,
                 r1: int, r2: int, r3: int, bl: int, stage: str):
        super().__init__()
        assert stage in ("encoder", "decoder")
        self.query_conv = nn.Conv1d(q_dim, q_dim // r1, 1)
        self.key_conv = nn.Conv1d(k_dim, k_dim // r2, 1)
        self.value_conv = nn.Conv1d(v_dim, v_dim // r3, 1)
        self.conv_out = nn.Conv1d(v_dim // r3, v_dim, 1)
        self.bl = bl
        self.stage = stage
        self.register_buffer("window_mask", self._window_mask(), persistent=False)

    def _window_mask(self) -> torch.Tensor:
        bl = self.bl
        m = torch.zeros(1, bl, bl + 2 * (bl // 2))
        for i in range(bl):
            m[:, i, i:i + bl] = 1.0
        return m

    @staticmethod
    def _attend(q, k, v, mask):
        """Reference AttentionHelper.scalar_dot_att.

        Masking is additive in log space (`+ log(mask + 1e-6)`) rather than a
        fill with -inf: a block consisting entirely of padding then softmaxes to
        a finite uniform row that the subsequent multiply by `mask` zeroes,
        instead of producing NaN.
        """
        c = k.shape[1]
        energy = torch.bmm(q.permute(0, 2, 1), k) / math.sqrt(c)
        att = torch.softmax(energy + torch.log(mask + 1e-6), dim=-1) * mask
        return torch.bmm(v, att.permute(0, 2, 1))

    def forward(self, x1, x2, mask):
        # x1, x2: (B, C, T); mask: (B, 1, T)
        q = self.query_conv(x1)
        k = self.key_conv(x1)
        v = self.value_conv(x2 if self.stage == "decoder" else x1)

        B, _, T = q.shape
        bl = self.bl
        nb = (T + bl - 1) // bl
        pad = nb * bl - T
        half = bl // 2

        def chunk_kv(t):
            # pad to nb*bl, then a halo of `half` on each side, then overlapping
            # spans of bl + 2*half, one per query block
            t = F.pad(t, (0, pad))
            t = F.pad(t, (half, half))
            spans = [t[:, :, i * bl: (i + 1) * bl + 2 * half] for i in range(nb)]
            return torch.cat(spans, dim=0)          # (B*nb, C, bl + 2*half)

        qc = F.pad(q, (0, pad)).reshape(B, -1, nb, bl) \
              .permute(0, 2, 1, 3).reshape(B * nb, -1, bl)
        kc, vc = chunk_kv(k), chunk_kv(v)
        mc = chunk_kv(F.pad(mask, (0, pad)) if pad else mask)[:, :1]

        wm = self.window_mask.to(qc.dtype).repeat(B * nb, 1, 1)
        out = self._attend(qc, kc, vc, wm * mc)     # (B*nb, Cv, bl)
        out = out.reshape(B, nb, -1, bl).permute(0, 2, 1, 3) \
                 .reshape(B, -1, nb * bl)[:, :, :T]
        return self.conv_out(out) * mask


class AttModule(nn.Module):
    """ff -> alpha * att(norm(ff)) + ff -> 1x1 -> dropout -> + x.

    One residual path, as in the reference. The earlier version here had two
    (attention block then feed-forward block, transformer-style), which is a
    different module however similar it reads.
    """

    def __init__(self, dilation: int, in_ch: int, out_ch: int,
                 r1: int, r2: int, stage: str, alpha: float):
        super().__init__()
        self.feed_forward = ConvFeedForward(dilation, in_ch, out_ch)
        self.instance_norm = nn.InstanceNorm1d(in_ch, track_running_stats=False)
        self.att_layer = AttLayer(in_ch, in_ch, out_ch, r1, r1, r2, dilation, stage)
        self.conv_1x1 = nn.Conv1d(out_ch, out_ch, 1)
        self.dropout = nn.Dropout()          # the reference's default p = 0.5
        self.alpha = alpha

    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.dropout(self.conv_1x1(out))
        return (x + out) * mask


class ASFormerStage(nn.Module):
    """Encoder (alpha = 1, self-attention) or decoder (alpha < 1, value from f)."""

    def __init__(self, in_ch, ch, num_classes, num_layers, r1, r2,
                 stage: str, alpha: float, channel_masking_rate: float = 0.0):
        super().__init__()
        self.channel_masking_rate = channel_masking_rate
        self.channel_dropout = nn.Dropout2d(channel_masking_rate)
        self.conv_in = nn.Conv1d(in_ch, ch, 1)
        self.layers = nn.ModuleList([
            AttModule(2 ** i, ch, ch, r1, r2, stage, alpha)
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(ch, num_classes, 1)

    def forward(self, x, f, mask):
        if self.channel_masking_rate > 0:
            x = self.channel_dropout(x.unsqueeze(2)).squeeze(2)
        h = self.conv_in(x) * mask
        for layer in self.layers:
            h = layer(h, f, mask)
        return self.conv_out(h) * mask, h


class ASFormer(nn.Module):
    """Encoder + three refinement decoders (Yi, Wen and Jiang, BMVC 2021).

    Follows the authors' released implementation; see the note at the head of
    this section for what changed and why. Like MS-TCN it exposes every stage's
    logits on `.stage_logits`, because the decoders are supervised individually
    in the published design.
    """

    def __init__(self, in_dim: int, num_classes: int, num_decoders: int = 3,
                 num_layers: int = 10, ch: int = 64, r1: int = 2, r2: int = 2,
                 channel_masking_rate: float = 0.3):
        super().__init__()
        self.encoder = ASFormerStage(in_dim, ch, num_classes, num_layers,
                                     r1, r2, "encoder", alpha=1.0,
                                     channel_masking_rate=channel_masking_rate)
        self.decoders = nn.ModuleList([
            ASFormerStage(num_classes, ch, num_classes, num_layers, r1, r2,
                          "decoder", alpha=exponential_decrease(s))
            for s in range(num_decoders)
        ])
        self.stage_logits: list[torch.Tensor] = []

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = feats.transpose(1, 2)
        m = mask.unsqueeze(1)
        logits, enc_h = self.encoder(x, None, m)
        self.stage_logits = [logits.transpose(1, 2)]
        for dec in self.decoders:
            logits, _ = dec(F.softmax(logits, dim=1) * m, enc_h, m)
            self.stage_logits.append(logits.transpose(1, 2))
        return logits.transpose(1, 2)


# --------------------------------------------------------------------------- #
def stagewise_loss(stage_logits, labels, mask, num_classes,
                   smooth_weight: float = 0.15, tau: float = 4.0):
    """CE + truncated-MSE smoothing, summed over stages.

    This is part of the METHOD, not a training convenience: MS-TCN's reported
    performance depends on supervising every stage and on the smoothing term
    that penalises rapid frame-to-frame change in the log-probabilities. Leaving
    either out would understate the baseline, which is the one direction of error
    this paper cannot afford in a baseline it is comparing itself against.
    """
    total = labels.new_zeros((), dtype=torch.float)
    m = mask.float()
    denom = m.sum().clamp(min=1.0)
    for logits in stage_logits:
        B, T, K = logits.shape
        ce = F.cross_entropy(
            logits.reshape(B * T, K), labels.reshape(B * T), reduction="none"
        ).view(B, T)
        total = total + (ce * m).sum() / denom

        lp = F.log_softmax(logits, dim=-1)
        d = torch.clamp((lp[:, 1:] - lp[:, :-1]) ** 2, min=0.0, max=tau ** 2)
        pair = (m[:, 1:] * m[:, :-1]).unsqueeze(-1)
        total = total + smooth_weight * (d * pair).sum() / (pair.sum().clamp(min=1.0) * K)
    return total
