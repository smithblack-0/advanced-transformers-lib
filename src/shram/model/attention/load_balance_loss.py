"""Log-probability auxiliary loss functions for MoSRAH load balancing.

This module provides four load-balance loss formulations, two token-reduction
helpers, and a factory that selects among the formulations. All formulations
share the same external contract:

    loss_fn(
        logits:          Tensor[B, N, L],
        assignment_mask: Tensor[B, N, L],
        active_mask:     Tensor[B, N],
    ) -> scalar Tensor

    logits:          Pre-softmax routing scores, shape (B, N, L). Gradient flows
                     through this tensor.
    assignment_mask: Per-token head-assignment indicators. assignment_mask[b, n, l]
                     is 1.0 if token (b, n) was assigned to head l. Dead tokens
                     should carry zero entries.
    active_mask:     Boolean mask, shape (B, N). True means the token is
                     semantically live.

Token reduction is split into two helpers with distinct roles:

    reduce_frequency_tokens — produces per-batch-item routing frequencies f_bl (B, L).
        Called by gshard, ce, and bce. Output is detached; f_bl carries no gradient.

    reduce_probability_tokens — produces per-batch-item mean assignment probabilities
        p_bl (B, L). Called only by gshard and bce. Gradient flows through the
        internal softmax over logits.

CE delegates probability computation to F.cross_entropy, which handles its own
log_softmax and operates directly on the raw (B, N, L) logits.

``make_load_balance_loss`` is the sole public entry point. The individual loss
functions are internal implementation details; their signatures may change between
units. Callers and tests must construct loss callables through the factory, not by
importing or invoking the loss functions directly.
"""

import torch
import torch.nn.functional as F
from typing import Callable


# ---------------------------------------------------------------------------
# Token-reduction helpers
# ---------------------------------------------------------------------------

def reduce_frequency_tokens(
    assignment_mask: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Reduce per-token head assignments to per-batch-item routing frequencies.

    f_bl[b, l] is the fraction of active-token assignments in batch item b going
    to head l. Values sum to 1 per batch item when routing is valid.

    The output is detached from the autograd graph: routing frequencies are
    derived from discrete TopK selections and must not carry gradients.

    Denominators are clamped to 1 to handle the all-dead-tokens edge case.

    Args:
        assignment_mask: Per-token head-assignment indicators, shape (B, N, L).
        active_mask:     Boolean active-token mask, shape (B, N).

    Returns:
        f_bl: Per-batch-item routing frequencies, shape (B, L). Detached.
    """
    active_float = active_mask.float().unsqueeze(-1)                               # (B, N, 1)
    active_assignments = assignment_mask * active_float                             # (B, N, L)
    assignment_totals = (
        active_assignments.sum(dim=(1, 2)).clamp(min=1.0).unsqueeze(-1)            # (B, 1)
    )
    return (active_assignments.sum(dim=1) / assignment_totals).detach()            # (B, L)


def reduce_probability_tokens(
    logits: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Reduce per-token load-balancing logits to per-batch-item assignment probabilities.

    p_bl[b, l] is the mean softmax probability for head l over active tokens in
    batch item b. Values sum to 1 per batch item. Gradient flows to expert_bias
    through the internal softmax.

    Denominators are clamped to 1 to handle the all-dead-tokens edge case.

    Args:
        logits:      Load-balancing logits, shape (B, N, L). Gradient flows through.
        active_mask: Boolean active-token mask, shape (B, N).

    Returns:
        p_bl: Per-batch-item mean assignment probabilities, shape (B, L).
    """
    per_token_probs = F.softmax(logits, dim=-1)                                    # (B, N, L)
    active_float = active_mask.float().unsqueeze(-1)                               # (B, N, 1)
    active_count = active_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)     # (B, 1)
    return (per_token_probs * active_float).sum(dim=1) / active_count              # (B, L)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def gshard_loss(
    logits: torch.Tensor,
    assignment_mask: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """GShard-style linear load-balance loss.

    Computes (1/L) * Σ_l f_bl * p_bl per batch item, averaged over B, where
    f_bl comes from reduce_frequency_tokens and p_bl from reduce_probability_tokens.

    The linear signal is the weakest of the three formulations; gradient magnitude
    does not grow with violation severity. Provided for comparison.

    Args:
        logits:          Load-balancing logits, shape (B, N, L).
        assignment_mask: Per-token head-assignment indicators, shape (B, N, L).
        active_mask:     Boolean active-token mask, shape (B, N).

    Returns:
        Scalar loss tensor.
    """
    L = logits.shape[-1]
    f_bl = reduce_frequency_tokens(assignment_mask, active_mask)
    p_bl = reduce_probability_tokens(logits, active_mask)
    return (f_bl * p_bl).sum(dim=-1).mean() / L


def ce_loss(
    logits: torch.Tensor,
    assignment_mask: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy load-balance loss.

    Constructs per-batch-item soft target distributions from routing frequencies
    and delegates to F.cross_entropy operating directly on (B, N, L) logits.
    Inactive tokens receive all-zero targets, producing zero loss and zero gradient.

    The soft target for head l in batch item b is (1 - f_bl) / (L - 1). This
    distribution sums to 1 per batch item (since Σ_l (1 - f_bl) = L - 1) and
    weights underloaded heads (low f_bl → high target) more strongly than
    overloaded ones.

    The total CE over active tokens is normalised by the active token count rather
    than B*N to avoid dilution from inactive positions.

    Args:
        logits:          Load-balancing logits, shape (B, N, L).
        assignment_mask: Per-token head-assignment indicators, shape (B, N, L).
        active_mask:     Boolean active-token mask, shape (B, N).

    Returns:
        Scalar loss tensor.
    """
    B, N, L = logits.shape
    f_bl = reduce_frequency_tokens(assignment_mask, active_mask)               # (B, L)
    active_count = active_mask.float().sum().clamp(min=1.0)

    # Soft target: (1 - f_bl) / (L - 1) for active tokens, zeros for inactive.
    # Zeros give zero CE loss and zero gradient at inactive positions.
    target = (1.0 - f_bl) / (L - 1)                                           # (B, L)
    target_per_token = (
        target.unsqueeze(1).expand(-1, N, -1)                                  # (B, N, L)
        * active_mask.float().unsqueeze(-1)                                    # zero inactive
    )

    # F.cross_entropy requires the class dimension to be dim 1.
    # Permute (B, N, L) → (B, L, N) to satisfy the (N, C, d) contract.
    return F.cross_entropy(
        logits.permute(0, 2, 1),             # (B, L, N)
        target_per_token.permute(0, 2, 1),   # (B, L, N)
        reduction='sum',
    ) / active_count


def bce_loss(
    logits: torch.Tensor,
    assignment_mask: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy load-balance loss.

    Treats each head as an independent binary target with label (1 - f_bl).
    Uses reduce_probability_tokens to produce per-batch-item probabilities,
    then delegates to F.binary_cross_entropy over (B, L) tensors.

    Unlike CE, BCE maintains a repulsion signal from saturated experts: when
    f_bl → 1 the target → 0, driving p_bl away from 1 and preventing runaway
    concentration.

    Active masking is handled inside reduce_frequency_tokens and
    reduce_probability_tokens, so the (B, L) output tensors already exclude
    inactive tokens from both frequencies and probabilities.

    Args:
        logits:          Load-balancing logits, shape (B, N, L).
        assignment_mask: Per-token head-assignment indicators, shape (B, N, L).
        active_mask:     Boolean active-token mask, shape (B, N).

    Returns:
        Scalar loss tensor.
    """
    f_bl = reduce_frequency_tokens(assignment_mask, active_mask)
    p_bl = reduce_probability_tokens(logits, active_mask)
    # Clamp for numerical safety: softmax outputs are strictly positive in
    # normal operation; the clamp guards the all-dead-tokens edge case where
    # the mean defaults to zero. log1p(-p) avoids cancellation near p=1.
    p = p_bl.clamp(min=1e-7, max=1.0 - 1e-7)
    target = 1.0 - f_bl
    return -(target * torch.log(p) + (1.0 - target) * torch.log1p(-p)).mean()


def _temporal_overcapacity_loss(
    logits: torch.Tensor,
    assignment_mask: torch.Tensor,
    active_mask: torch.Tensor,
    expected_tokens_rate: float,
    maximum_expert_overclaim: int,
) -> torch.Tensor:
    """Temporal overcapacity loss for MoSRAH load balancing.

    Penalises routing decisions that select a head already overloaded relative to
    its ideal allocation trajectory. A head is considered overloaded when the number
    of active tokens before position n assigned to that head exceeds
    cumulative_active_tokens * M + C, where M is the expected_tokens_rate (K/L) and
    C is the maximum_expert_overclaim slack.

    Loss is exactly zero when no head exceeds its trajectory, making it safe to
    weight strongly — it stays out of the way when routing is balanced.

    Args:
        logits:                   Pre-softmax routing scores, shape (B, N, L).
        assignment_mask:          Per-token head-assignment indicators, shape (B, N, L).
                                  1.0 if token (b, n) is assigned to head l.
        active_mask:              Boolean active-token mask, shape (B, N).
        expected_tokens_rate (M): Ideal per-head allocation rate K/L. Pre-computed
                                  by the factory so the division is not repeated each
                                  forward pass.
        maximum_expert_overclaim (C): Slack above the ideal trajectory before
                                  imbalance fires. Larger C tolerates more deviation.

    Returns:
        Scalar loss tensor. Exactly 0.0 when no head exceeds its allowed trajectory.
    """
    # ── Algorithm overview ──────────────────────────────────────────────────────
    #
    # Problem: token routing is stateless — each token's TopK selection is blind to
    # how many times each expert has already been chosen earlier in the sequence. A
    # router that develops a strong preference for certain experts will overload them
    # far beyond their K/L fair share with no correction signal at the moment of
    # selection.
    #
    # Approach: track per-head assignment history as exclusive cumulative counts
    # (assignments by all active tokens strictly before position n) and compare
    # against an ideal trajectory S·M, where S is the inclusive cumulative active
    # token count and M is the amount of tokens expected given ideal balancing
    #  A head is overloaded when its prior count exceeds that trajectory
    # by more than C. When a token selects an already-overloaded head, the loss
    # moment — mean(violating logits) minus mean(non-overloaded logits) — penalises
    # the gap and pushes future routing toward underloaded alternatives.

    # ── Routing history and imbalance threshold ──────────────────────────────────
    #
    # prior_assignment_counts is the exclusive routing history at each position:
    # active assignments to each head by all tokens strictly before position n.
    # Exclusive because it reflects only what was known when token n was being routed.
    # cumulative_active_tokens grows by 1 per active token; the ideal per-head
    # allocation at n is S·M. Exceeding that by more than C triggers imbalance.

    active_float = active_mask.float()                                              # (B, N)
    active_assignments = assignment_mask * active_float.unsqueeze(-1)               # (B, N, L)

    # exclusive cumsums: subtract self to exclude position n
    prior_assignment_counts = active_assignments.cumsum(dim=1) - active_assignments  # (B, N, L)
    cumulative_active_tokens = active_float.cumsum(dim=1) - active_float             # (B, N)

    maximum_supportable_assignments = (
        cumulative_active_tokens.unsqueeze(-1) * expected_tokens_rate
        + maximum_expert_overclaim
    )                                                                                # (B, N, 1) → broadcasts to (B, N, L)

    # ── Mask construction ────────────────────────────────────────────────────────
    #
    # Three derived masks:
    #   imbalance_mask:           any head exceeding its trajectory.
    #   violating_selection_mask: selected AND imbalanced — the penalty target.
    #   non_overloaded_head_mask: NOT imbalanced, regardless of selection.
    #
    # Masking is deliberately assymetric. We have a problem when something is over
    # capacity AND gets chosen by topk. We can transfer it elsewhere only if we
    # are not overcapacity.

    imbalance_mask           = prior_assignment_counts > maximum_supportable_assignments  # (B, N, L)
    violating_selection_mask = assignment_mask.bool() & imbalance_mask                   # (B, N, L)
    non_overloaded_head_mask = ~imbalance_mask                                            # (B, N, L)
    has_violation_mask       = violating_selection_mask.any(dim=-1)                       # (B, N)

    # ── Loss moment ────────────────────────────────────────────────────────
    #
    # Epsilons on the count denominators guard against NaN when violation_count or
    # non_overloaded_count is zero. has_violation_mask zeros positions with no
    # violations at the gating step, so the epsilon-inflated denominator never
    # contributes to the loss.
    #
    # One notable property of this moment is it keeps the amount of transferred
    # logit mass constant. That is the gradient reduces violating logits and increases
    # non-overloaded logits by equal magnitude. Routing is redirected, not suppressed.

    violation_count           = violating_selection_mask.float().sum(dim=-1).clamp(min=1.0)   # (B, N)
    non_overloaded_count      = non_overloaded_head_mask.float().sum(dim=-1).clamp(min=1.0)   # (B, N)
    mean_violating_logit      = (violating_selection_mask.float() * logits).sum(dim=-1) / violation_count      # (B, N)
    mean_non_overloaded_logit = (non_overloaded_head_mask.float() * logits).sum(dim=-1) / non_overloaded_count  # (B, N)
    raw_loss                  = mean_violating_logit - mean_non_overloaded_logit                                 # (B, N)

    # ── Loss reduction ───────────────────────────────────────────────────────────
    #
    # Reduction is over active positions only; dead tokens are excluded from both
    # numerator (gated by active_float) and denominator (active_count_per_seq).
    # clamp(min=1.0) handles the all-dead-tokens edge case: gated_loss is zero
    # there since active_float gates it, so the result is 0/1 = 0.
    #
    # Exact-zero guarantee: when no head exceeds its trajectory, has_violation_mask
    # is all-False, gated_loss is zeroed everywhere, and the scalar return is
    # exactly 0.0. The loss is inert when routing is balanced.

    gated_loss           = active_float * has_violation_mask.float() * raw_loss           # (B, N)
    active_count_per_seq = active_float.sum(dim=1).clamp(min=1.0)                         # (B,)
    sequence_loss        = gated_loss.sum(dim=1) / active_count_per_seq                   # (B,)
    final_loss           = sequence_loss.mean()
    return final_loss


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _gshard_factory(**kwargs: object) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    return gshard_loss


def _ce_factory(**kwargs: object) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    return ce_loss


def _bce_factory(**kwargs: object) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    return bce_loss


def _temporal_overcapacity_factory(
    num_selected_heads: int,
    num_total_heads: int,
    maximum_expert_overclaim: int,
    **kwargs: object,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    expected_tokens_rate = num_selected_heads / num_total_heads
    def _runtime(
        logits: torch.Tensor,
        assignment_mask: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        return _temporal_overcapacity_loss(
            logits, assignment_mask, active_mask,
            expected_tokens_rate=expected_tokens_rate,
            maximum_expert_overclaim=maximum_expert_overclaim,
        )
    return _runtime


_LOSS_REGISTRY: dict[str, Callable[..., Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]] = {
    "gshard": _gshard_factory,
    "ce": _ce_factory,
    "bce": _bce_factory,
    "temporal_overcapacity": _temporal_overcapacity_factory,
}


def make_load_balance_loss(
    loss_type: str,
    **loss_parameters: object,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a load-balance loss callable for the requested formulation.

    All returned callables share the external contract:

        loss_fn(
            logits:          Tensor[B, N, L],
            assignment_mask: Tensor[B, N, L],
            active_mask:     Tensor[B, N],
        ) -> scalar Tensor

    Keyword arguments are forwarded to the selected factory. The gshard, ce, and bce
    factories silently ignore all kwargs; this allows callers to pass loss-type-specific
    parameters (e.g. for temporal_overcapacity) without branching on loss_type.

    Args:
        loss_type:        One of ``"gshard"``, ``"ce"``, ``"bce"``, or
                          ``"temporal_overcapacity"``.
        **loss_parameters: Construction-time parameters forwarded to the factory.

    Returns:
        Loss callable matching the shared contract.

    Raises:
        ValueError: If loss_type is not one of the supported values.
    """
    if loss_type not in _LOSS_REGISTRY:
        supported = ", ".join(f'"{k}"' for k in _LOSS_REGISTRY)
        raise ValueError(
            f"load_balance_loss_type must be one of {supported}, got {loss_type!r}."
        )
    return _LOSS_REGISTRY[loss_type](**loss_parameters)
