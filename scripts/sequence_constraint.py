"""
Learnable Construction-Sequence-Constrained layer  (PhD contribution #1).

A differentiable, domain-prior-regularized linear-chain CRF that sits on top of a
per-frame vision backbone and turns independent frame scores into a globally
consistent construction-stage sequence.

Why this is the contribution (vs. prior art)
--------------------------------------------
* Classic linear-chain CRF  -> transitions are free / hand-set; no domain prior.
* Hard monotonic constraint  -> forbids ANY backward transition (brittle: breaks
  under occlusion, weather, re-work, or label noise).
* THIS layer               -> transitions are LEARNABLE, but each transition owns
  a learnable gate g_ij in (0,1) that blends the free transition with a strong
  construction prior (monotonic progress + precedence). A regularizer pulls the
  gates toward "trust the prior", and the data is allowed to override a gate only
  where genuine backward evidence exists. The whole thing trains end-to-end with
  the backbone.

Notation
--------
K  = number of ordered construction stages (0 = earliest, K-1 = latest).
B  = batch, T = sequence length (frames).
Transition matrix `trans[i, j]` = score of moving FROM stage i TO stage j.
"trans[i, j] with j < i" is a *backward* (progress-reversing) transition.
"""

from __future__ import annotations

import torch
import torch.nn as nn


#: magnitude of the backward bias that makes a transition effectively impossible
HARD_BIAS = 1e4

#: transition parameterizations — `gated` is ours, the others are the baselines
#: that isolate *which* part of it does the work (see contribution-01 §5).
#:
#: `soft` was added 2026-10-14 in answer to IJIES reviewer 2, whose objection is
#: the sharpest one the paper received: `hard` forbids backward moves with a
#: penalty of 1e4 while `gated` starts at an effective penalty near 7, so any
#: difference between those two arms can come from CONSTRAINT STRENGTH rather
#: than from LEARNABILITY, which is what the paper claims to be measuring.
#: `soft` removes that confound by fixing the penalty at exactly the gated arm's
#: initial effective value, with no learnable gate and no regularizer. The
#: contrast that carries the paper's negative result is therefore `soft` vs
#: `gated` — same strength, learnability varied — and no longer `hard` vs
#: `gated`, which varied both at once.
TRANSITION_MODES = ("gated", "free", "hard", "soft")


class ConstructionSequenceConstraint(nn.Module):
    """Differentiable CRF over construction stages, in one of three modes.

    The `mode` argument is the decisive ablation *inside* the CRF family — it
    holds the CRF machinery fixed and varies only how the transition matrix is
    parameterized:

    * ``"gated"`` (ours) -- ``trans_free + sigmoid(theta) * prior``: the prior is
      applied through a learnable per-transition gate, regularized toward "trust
      the prior", so the data can re-open a specific backward transition where
      genuine evidence exists (re-work, label noise).
    * ``"free"`` (classic CRF baseline) -- ``trans_free`` alone. No domain prior;
      transitions are learned purely from data.
    * ``"hard"`` (hard-monotonic baseline) -- ``trans_free - HARD_BIAS`` on every
      backward transition, gate frozen at 1. Backward moves are forbidden, not
      discouraged. This baseline also scores 0.000 order violations by
      construction, which is exactly why it is required: it is the baseline that
      makes the order-violation metric non-trivial.
    * ``"soft"`` (fixed-strength control) -- the prior applied at a FIXED
      magnitude ``sigmoid(gate_init) * backward_bias``, gate frozen, no
      regularizer. That magnitude is by construction the effective penalty the
      ``"gated"`` arm starts from, so ``soft`` and ``gated`` differ in exactly
      one thing: whether the penalty can move. This is the arm that isolates
      *learnability*; ``hard`` vs ``gated`` does not, because it changes the
      penalty magnitude by three orders of magnitude at the same time.

    Args:
        num_stages: K, number of construction stages in true order.
        backward_bias: magnitude of the monotonic prior applied to backward
            transitions (the prior pushes them to -backward_bias). Larger =>
            stronger default "you cannot go back" belief. Ignored in mode
            ``"free"``; overridden by ``HARD_BIAS`` in mode ``"hard"``.
        gate_init: initial logit of every gate (positive => prior active by
            default; 2.0 => sigmoid ~= 0.88, i.e. mostly trust the prior).
        mode: one of :data:`TRANSITION_MODES`.
    """

    def __init__(
        self,
        num_stages: int,
        backward_bias: float = 8.0,
        gate_init: float = 2.0,
        mode: str = "gated",
    ):
        super().__init__()
        if mode not in TRANSITION_MODES:
            raise ValueError(f"mode must be one of {TRANSITION_MODES}, got {mode!r}")
        self.num_stages = int(num_stages)
        self.mode = mode

        K = self.num_stages
        # Free, fully-learnable transition potentials (present in every mode, so
        # the three modes differ ONLY in the prior term).
        self.trans_free = nn.Parameter(torch.zeros(K, K))
        # Per-frame start / end potentials.
        self.start = nn.Parameter(torch.zeros(K))
        self.end = nn.Parameter(torch.zeros(K))

        # Mode-specific prior magnitude and gate behaviour.
        if mode == "gated":
            self.backward_bias = float(backward_bias)
            gate_value = float(gate_init)
            gate_trainable = True
        elif mode == "hard":
            self.backward_bias = HARD_BIAS
            gate_value = 20.0        # sigmoid(20) ~= 1 -> prior fully applied
            gate_trainable = False   # ...and not negotiable
        elif mode == "soft":
            # The penalty the gated arm STARTS at, frozen there. Folding the
            # initial gate into the bias (rather than leaving gate_init on a
            # frozen gate) keeps the arm's strength readable in one number, and
            # `gate_value = 20.0` then makes sigmoid(gate) ~= 1 so the effective
            # backward penalty is exactly this bias.
            self.backward_bias = float(torch.sigmoid(torch.tensor(gate_init)).item()
                                       * backward_bias)
            gate_value = 20.0
            gate_trainable = False
        else:  # "free"
            self.backward_bias = 0.0
            gate_value = 0.0
            gate_trainable = False

        # Learnable gate (one logit per transition) — the "learnable constraint".
        self.gate_logit = nn.Parameter(
            torch.full((K, K), gate_value), requires_grad=gate_trainable
        )

        # Fixed domain prior: a large negative bias on every backward transition
        # (j < i). Self/forward transitions get zero bias.
        idx = torch.arange(K)
        backward_mask = (idx.view(K, 1) > idx.view(1, K)).float()  # 1 where j < i
        self.register_buffer("backward_mask", backward_mask)
        self.register_buffer("prior_bias", -self.backward_bias * backward_mask)

    # ------------------------------------------------------------------ #
    #  Effective transitions and the domain-prior regularizer
    # ------------------------------------------------------------------ #
    def effective_transitions(self) -> torch.Tensor:
        """trans_eff = free_transition + gate * monotonic_prior."""
        gate = torch.sigmoid(self.gate_logit)
        return self.trans_free + gate * self.prior_bias

    def prior_regularizer(self) -> torch.Tensor:
        """Pull gates on backward transitions toward 1 (apply the prior).

        Combined with the CRF likelihood, the data can still drive a gate down
        wherever genuine backward transitions are observed. Only mode ``"gated"``
        has gates to regularize; the baselines return 0 so the loss stays
        comparable across modes.
        """
        if self.mode != "gated":
            return self.gate_logit.new_zeros(())
        gate = torch.sigmoid(self.gate_logit)
        denom = self.backward_mask.sum().clamp(min=1.0)
        return ((1.0 - gate) * self.backward_mask).sum() / denom

    @torch.no_grad()
    def gate_stats(self) -> dict:
        """Learned gate values on backward transitions (the interpretability
        evidence: which prior did the data decide to *stop* trusting?)."""
        gate = torch.sigmoid(self.gate_logit)
        bw = self.backward_mask.bool()
        backward_gates = gate[bw]
        return {
            "mode": self.mode,
            "mean_backward_gate": backward_gates.mean().item(),
            "min_backward_gate": backward_gates.min().item(),
            "gate_matrix": gate.tolist(),
            "trans_eff": self.effective_transitions().tolist(),
        }

    # ------------------------------------------------------------------ #
    #  CRF core
    # ------------------------------------------------------------------ #
    def _forward_alg(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Log partition function log Z, per batch element.  emissions: (B,T,K)."""
        B, T, K = emissions.shape
        trans = self.effective_transitions()  # (K, K)
        alpha = self.start.view(1, K) + emissions[:, 0]  # (B, K)
        for t in range(1, T):
            # (B,K_from,1) + (1,K_from,K_to) + (B,1,K_to) -> (B,K_from,K_to)
            scores = alpha.unsqueeze(2) + trans.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            new_alpha = torch.logsumexp(scores, dim=1)  # (B, K_to)
            m = mask[:, t].unsqueeze(1)  # (B, 1)
            alpha = m * new_alpha + (1.0 - m) * alpha
        alpha = alpha + self.end.view(1, K)
        return torch.logsumexp(alpha, dim=1)  # (B,)

    def _score_sequence(self, emissions, tags, mask) -> torch.Tensor:
        """Score of the gold tag sequence, per batch element."""
        B, T, K = emissions.shape
        trans = self.effective_transitions()
        score = self.start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        for t in range(1, T):
            emit = emissions[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            tr = trans[tags[:, t - 1], tags[:, t]]
            score = score + mask[:, t] * (tr + emit)
        last_idx = mask.sum(1).long() - 1  # (B,)
        last_tags = tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        score = score + self.end[last_tags]
        return score

    def nll(self, emissions, tags, mask=None, reg_weight: float = 0.1) -> torch.Tensor:
        """Mean negative log-likelihood + prior regularizer (the training loss)."""
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2])
        mask = mask.float()
        log_Z = self._forward_alg(emissions, mask)
        gold = self._score_sequence(emissions, tags, mask)
        loss = (log_Z - gold).mean()
        return loss + reg_weight * self.prior_regularizer()

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask=None) -> list[list[int]]:
        """Viterbi decode -> most likely stage path per sequence."""
        B, T, K = emissions.shape
        if mask is None:
            mask = emissions.new_ones(B, T)
        lengths = mask.float().sum(1).long().tolist()
        trans = self.effective_transitions()
        paths: list[list[int]] = []
        for b in range(B):
            L = max(1, int(lengths[b]))
            emis = emissions[b, :L]  # (L, K)
            score = self.start + emis[0]  # (K,)
            back = []
            for t in range(1, L):
                s = score.unsqueeze(1) + trans  # (K_from, K_to)
                best_prev, idx = s.max(dim=0)  # (K_to,)
                score = best_prev + emis[t]
                back.append(idx)
            score = score + self.end
            last = int(score.argmax())
            path = [last]
            for idx in reversed(back):
                last = int(idx[last])
                path.append(last)
            path.reverse()
            paths.append(path)
        return paths


# ---------------------------------------------------------------------- #
#  Evaluation helper: construction-specific order-violation rate
# ---------------------------------------------------------------------- #
def order_violation_rate(paths: list[list[int]]) -> float:
    """Fraction of adjacent frame pairs that go backward (stage decreases).

    This is a domain metric the vision-only baselines cannot optimize directly;
    a good sequence-constraint layer should drive it toward 0.
    """
    bad = total = 0
    for p in paths:
        for a, b in zip(p[:-1], p[1:]):
            total += 1
            if b < a:
                bad += 1
    return bad / max(1, total)


if __name__ == "__main__":
    # -------- Self-test / demo on synthetic monotone data -------------- #
    torch.manual_seed(0)
    K, T, B = 4, 24, 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def make_batch(n, noise=1.4, strength=2.0, rework_prob=0.0, rework_len=3):
        """Stage labels + noisy emissions (argmax will misfire).

        `rework_prob` > 0 injects genuine *backward* segments (re-work: a finished
        stage is torn out and redone), which is the situation the three transition
        modes are supposed to disagree about.
        """
        labels = torch.zeros(n, T, dtype=torch.long)
        for i in range(n):
            # 3 random increasing change-points -> monotone 0..K-1
            cps = torch.sort(torch.randint(1, T, (K - 1,))).values
            lab, s = [], 0
            for t in range(T):
                while s < K - 1 and t >= cps[s]:
                    s += 1
                lab.append(s)
            if rework_prob > 0 and torch.rand(()) < rework_prob:
                # pick a moment where we are past stage 0 and step one stage back
                cands = [t for t in range(T - rework_len) if lab[t] > 0]
                if cands:
                    t0 = cands[int(torch.randint(len(cands), ()))]
                    base = lab[t0] - 1  # capture before mutating lab[t0]
                    for t in range(t0, t0 + rework_len):
                        lab[t] = base
            labels[i] = torch.tensor(lab)
        onehot = torch.nn.functional.one_hot(labels, K).float() * strength
        emissions = onehot + noise * torch.randn(n, T, K)
        return emissions, labels

    def compare_modes(emissions, tags, header):
        print(f"\n{header}")
        print(f"{'mode':<8} {'acc':>7} {'viol':>7} {'mean_gate':>10} {'min_gate':>9}")
        print("-" * 45)
        for mode in TRANSITION_MODES:
            torch.manual_seed(0)
            m = ConstructionSequenceConstraint(K, mode=mode).to(device)
            opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=0.05)
            for _ in range(150):
                opt.zero_grad()
                m.nll(emissions, tags, reg_weight=0.1).backward()
                opt.step()
            paths = m.decode(emissions)
            acc = (torch.tensor(paths, device=device) == tags).float().mean().item()
            st = m.gate_stats()
            print(f"{mode:<8} {acc:>7.3f} {order_violation_rate(paths):>7.3f} "
                  f"{st['mean_backward_gate']:>10.3f} {st['min_backward_gate']:>9.3f}")

    emX, emY = make_batch(B)
    emX, emY = emX.to(device), emY.to(device)

    # sanity: log Z >= gold score for every sequence (partition dominates any path)
    probe = ConstructionSequenceConstraint(K).to(device)
    with torch.no_grad():
        logZ = probe._forward_alg(emX, emX.new_ones(B, T))
        gold = probe._score_sequence(emX, emY, emX.new_ones(B, T))
    assert (logZ + 1e-4 >= gold).all(), "CRF partition function is inconsistent!"
    print(f"[sanity]  log Z >= gold score for all {B} sequences  OK")

    # baseline: per-frame argmax (no sequence model)
    argmax_paths = emX.argmax(-1).tolist()
    print(f"[before]  argmax order-violation rate      = {order_violation_rate(argmax_paths):.3f}\n")

    # ---- the transition-mode comparison (the ablation inside the CRF family) --
    compare_modes(emX, emY, "[A] strictly monotone data (no re-work)")

    # Same test, but 40% of sequences contain a genuine backward segment. This is
    # the case that separates the modes: 'hard' CANNOT represent re-work at all.
    rwX, rwY = make_batch(B, rework_prob=0.4)
    rwX, rwY = rwX.to(device), rwY.to(device)
    gold_viol = order_violation_rate(rwY.tolist())
    compare_modes(rwX, rwY, f"[B] data with re-work (gold violation rate = {gold_viol:.3f})")

    print("\nReading: on [A] every mode ties, so monotone synthetic data proves")
    print("nothing about the gate. On [B] 'hard' is structurally unable to follow the")
    print("gold path, and the gate in 'gated' does drop below 1 where re-work is real.")
    print()
    print("The arm to read against 'gated' is 'soft', NOT 'hard'. 'soft' carries the")
    print("same backward penalty that 'gated' starts from and simply cannot move it,")
    print("so 'soft' vs 'gated' varies learnability alone, while 'hard' vs 'gated'")
    print("also varies the penalty by three orders of magnitude. If 'gated' and 'soft'")
    print("score the same on [B] while the gate visibly moves, then the gate is moving")
    print("without buying anything, and the advantage over 'hard' was strength rather")
    print("than learnability -- which is what IJIES reviewer 2 asked this test to show.")
