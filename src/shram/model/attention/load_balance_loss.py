"""Log-probability auxiliary loss functions for MoSRAH load balancing.

This module provides three load-balance loss formulations, two token-reduction
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
        Called by all three formulations. Output is detached; f_bl carries no gradient.

    reduce_probability_tokens — produces per-batch-item mean assignment probabilities
        p_bl (B, L). Called only by gshard and bce. Gradient flows to expert_bias
        through the internal softmax over logits.

CE delegates probability computation to F.cross_entropy, which handles its own
log_softmax and operates directly on the raw (B, N, L) logits.

The factory is the intended entry point. MoSRAHRouter constructs the loss callable
once at init and invokes it each forward pass.
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _gshard_factory(**kwargs: object) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    return gshard_loss


def _ce_factory(**kwargs: object) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    return ce_loss


def _bce_factory(**kwargs: object) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    return bce_loss


_LOSS_REGISTRY: dict[str, Callable[..., Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]] = {
    "gshard": _gshard_factory,
    "ce": _ce_factory,
    "bce": _bce_factory,
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
        loss_type:        One of ``"gshard"``, ``"ce"``, or ``"bce"``.
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
