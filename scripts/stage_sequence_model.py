"""
StageSequenceModel — ResNet18 backbone + learnable sequence-constraint head.

This is the wiring of Contribution #1 (`sequence_constraint.py`) on top of the real
Milestone-1 backbone. A batch of *ordered frame sequences* goes in; per-frame stage
emissions come out of the ResNet, and the CRF layer turns them into a globally
consistent stage path.

The `use_constraint` flag is the decisive ablation switch:
    * True  -> CRF likelihood loss  + Viterbi decode  (our method)
    * False -> per-frame cross-entropy + argmax decode (vision-only baseline)
Everything else (backbone, data, schedule) is held identical, so any difference in
order-violation rate / segmental scores is attributable to the constraint layer alone.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torchvision import models

from .sequence_constraint import ConstructionSequenceConstraint

#: temporal encoders inserted between the backbone and the decoder. `none` keeps
#: frames independent (the emissions are pure per-frame CNN outputs); `lstm` and
#: `transformer` are evaluation-plan baseline (b) -- a temporal model WITHOUT any
#: domain constraint, which separates "temporal context helps" from "the
#: construction prior helps".
#: `mstcn` and `asformer` are the standard temporal-action-segmentation models
#: (see tas_encoders.py). They sit in the same slot as `lstm` -- a sequence model
#: over backbone features with no domain prior -- so the three are directly
#: comparable. Unlike `lstm` they emit class logits rather than hidden features,
#: and they carry per-stage auxiliary losses; both are handled below.
TEMPORAL_ENCODERS = ("none", "lstm", "transformer", "mstcn", "asformer")

#: encoders whose output IS the emission, so no separate classifier is applied
LOGIT_ENCODERS = ("mstcn", "asformer")

#: Per-frame feature extractors. The paper's grid ran entirely on `resnet18`, and
#: that stays the DEFAULT so every earlier result reproduces untouched.
#:
#: The others exist to answer one reviewer objection: the paper concludes that
#: the domain prior carries the result while structured decoding alone does not,
#: and a reader can reasonably reply "only because your emissions are weak."
#: That is a claim about the BACKBONE, so it is tested by varying the backbone
#: and nothing else. Each entry is (constructor, weights-enum name, how to reach
#: the feature dimension).
#:
#: All are torchvision models with locally cached weights: no download is needed
#: at training time, which matters because the grid runs unattended.
BACKBONES = {
    "resnet18":      ("resnet18",      "ResNet18_Weights",      "fc"),
    "resnet50":      ("resnet50",      "ResNet50_Weights",      "fc"),
    "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights", "classifier"),
}


def build_backbone(name: str, pretrained: bool = True):
    """(module, feature_dim) with the classification head removed.

    The head is replaced by Identity rather than sliced off, so the module still
    returns a flat feature vector per image and the rest of the model does not
    need to know which family it is holding.
    """
    if name not in BACKBONES:
        raise ValueError(f"backbone must be one of {sorted(BACKBONES)}, got {name!r}")
    ctor_name, weights_enum, head_attr = BACKBONES[name]
    weights = getattr(models, weights_enum).DEFAULT if pretrained else None
    net = getattr(models, ctor_name)(weights=weights)
    if head_attr == "fc":
        dim = net.fc.in_features
        net.fc = nn.Identity()
    else:                                   # convnext: classifier is a Sequential
        dim = net.classifier[-1].in_features
        net.classifier[-1] = nn.Identity()
    return net, dim


class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed positional encoding; length-agnostic, no parameters."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class StageSequenceModel(nn.Module):
    def __init__(
        self,
        num_stages: int,
        use_constraint: bool = True,
        pretrained: bool = True,
        backward_bias: float = 8.0,
        gate_init: float = 2.0,
        transition_mode: str = "gated",
        temporal: str = "none",
        temporal_hidden: int = 256,
        temporal_layers: int = 1,
        backbone: str = "resnet18",
        freeze_backbone: bool = False,
    ):
        """`transition_mode` selects the CRF baseline family (see
        `sequence_constraint.TRANSITION_MODES`); it is ignored when
        `use_constraint=False`, which is the vision-only baseline.

        `temporal` selects the encoder between backbone and decoder (see
        `TEMPORAL_ENCODERS`). It is an axis ORTHOGONAL to `transition_mode`:
        `temporal=lstm, use_constraint=False` is evaluation-plan baseline (b),
        while `temporal=lstm, transition_mode=gated` would stack both."""
        super().__init__()
        if temporal not in TEMPORAL_ENCODERS:
            raise ValueError(f"temporal must be one of {TEMPORAL_ENCODERS}, got {temporal!r}")
        self.num_stages = int(num_stages)
        self.use_constraint = bool(use_constraint)
        self.transition_mode = transition_mode if self.use_constraint else "none"
        self.temporal = temporal

        self.backbone_name = backbone
        self.freeze_backbone = bool(freeze_backbone)
        self.backbone, self.feat_dim = build_backbone(backbone, pretrained)
        if self.freeze_backbone:
            # Frozen features + trainable head. With ~2.3k labelled frames a large
            # backbone fine-tunes into overfitting, and freezing also removes the
            # backbone from the optimizer entirely -- see param_groups, which must
            # not hand Adam a group with no trainable tensors in it.
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # Temporal encoder (baseline b). Both variants are bidirectional /
        # full-attention: stage recognition is an offline task, so a frame may
        # use later frames as context.
        out_dim = self.feat_dim
        self.pos_enc = None
        if temporal == "lstm":
            self.temporal_enc = nn.LSTM(
                self.feat_dim, temporal_hidden, num_layers=temporal_layers,
                batch_first=True, bidirectional=True,
            )
            out_dim = temporal_hidden * 2
        elif temporal == "transformer":
            self.pos_enc = SinusoidalPositionalEncoding(self.feat_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=self.feat_dim, nhead=8, dim_feedforward=4 * temporal_hidden,
                dropout=0.1, batch_first=True,
            )
            # enable_nested_tensor=False: the nested-tensor fast path is a
            # prototype API and warns on every masked forward
            self.temporal_enc = nn.TransformerEncoder(
                layer, num_layers=temporal_layers, enable_nested_tensor=False
            )
        elif temporal in LOGIT_ENCODERS:
            from .tas_encoders import ASFormer, MSTCN
            cls = MSTCN if temporal == "mstcn" else ASFormer
            self.temporal_enc = cls(self.feat_dim, self.num_stages)
            out_dim = self.num_stages
        else:
            self.temporal_enc = None

        # a logit encoder already produced the emissions; a second linear layer
        # on top would add a free parameter the published architectures do not
        # have and would blur what is being compared
        self.classifier = (nn.Identity() if temporal in LOGIT_ENCODERS
                           else nn.Linear(out_dim, self.num_stages))

        if self.use_constraint:
            self.crf = ConstructionSequenceConstraint(
                self.num_stages, backward_bias=backward_bias,
                gate_init=gate_init, mode=transition_mode,
            )
        else:
            self.crf = None
        self._ce = nn.CrossEntropyLoss(reduction="none")

    # ------------------------------------------------------------------ #
    def _encode_temporal(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, T, feat_dim) -> (B, T, out_dim), padded frames excluded from context."""
        if self.temporal in LOGIT_ENCODERS:
            return self.temporal_enc(feats, mask)
        if self.temporal == "lstm":
            lengths = mask.sum(1).long().clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                feats, lengths, batch_first=True, enforce_sorted=False
            )
            out, _ = self.temporal_enc(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=feats.size(1)
            )
            return out
        # transformer: keep padded keys out of every frame's attention
        return self.temporal_enc(self.pos_enc(feats), src_key_padding_mask=(mask == 0))

    def emissions(self, frames: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """(B, T, 3, H, W) images -> (B, T, K) per-frame stage scores.

        `mask` is required whenever a temporal encoder is active, otherwise
        padding frames leak into the context of real ones.
        """
        B, T = frames.shape[:2]
        x = frames.flatten(0, 1)                          # (B*T, 3, H, W)
        feats = self.backbone(x).view(B, T, self.feat_dim)
        if self.temporal_enc is not None:
            if mask is None:
                mask = feats.new_ones(B, T)
            feats = self._encode_temporal(feats, mask.float())
        return self.classifier(feats)                     # (B, T, K)

    def compute_loss(self, emissions, labels, mask=None, reg_weight: float = 0.1):
        """Training loss from precomputed emissions (so metrics can reuse them)."""
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2])
        mask = mask.float()
        if self.use_constraint:
            return self.crf.nll(emissions, labels, mask, reg_weight=reg_weight)
        if self.temporal in LOGIT_ENCODERS:
            # supervise every refinement stage and apply the smoothing term, as
            # both published designs do. Training these on the last stage alone
            # would weaken a baseline we are comparing ourselves against, which
            # is the one direction of error that would flatter this paper.
            from .tas_encoders import stagewise_loss
            return stagewise_loss(self.temporal_enc.stage_logits, labels, mask,
                                  self.num_stages)
        # ablation baseline: masked per-frame cross-entropy
        B, T, K = emissions.shape
        ce = self._ce(emissions.reshape(B * T, K), labels.reshape(B * T)).view(B, T)
        return (ce * mask).sum() / mask.sum().clamp(min=1.0)

    @torch.no_grad()
    def decode_emissions(self, emissions, mask=None) -> list[list[int]]:
        """Predicted stage path per sequence (Viterbi if constrained, else argmax)."""
        B, T, K = emissions.shape
        if mask is None:
            mask = emissions.new_ones(B, T)
        if self.use_constraint:
            return self.crf.decode(emissions, mask)
        lengths = mask.float().sum(1).long().tolist()
        preds = emissions.argmax(-1)  # (B, T)
        return [preds[b, : max(1, int(lengths[b]))].tolist() for b in range(B)]

    def train(self, mode: bool = True):
        """Keep a frozen backbone in eval mode even inside model.train().

        Without this, BatchNorm in a "frozen" ResNet still updates its running
        statistics on every forward pass, so the features drift during training
        and the run is not the frozen-feature experiment it claims to be. The
        parameters would be frozen; the module would not.
        """
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def param_groups(self, backbone_lr: float, head_lr: float):
        """Backbone gets a smaller LR (fine-tune); head/temporal/CRF a larger one."""
        head = list(self.classifier.parameters())
        if self.temporal_enc is not None:
            head += list(self.temporal_enc.parameters())
        if self.crf is not None:
            head += list(self.crf.parameters())
        groups = [{"params": head, "lr": head_lr}]
        if not self.freeze_backbone:
            # a frozen backbone contributes no group at all, rather than an empty
            # one -- an empty group is legal but makes the optimizer state harder
            # to compare across arms
            groups.insert(0, {"params": self.backbone.parameters(),
                              "lr": backbone_lr})
        return groups
