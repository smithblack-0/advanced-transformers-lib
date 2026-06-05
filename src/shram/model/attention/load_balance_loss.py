"""Log-probability auxiliary loss functions for MoSRAH load balancing.

This module provides three load-balance loss formulations and a factory that selects
among them. All formulations share the same external contract and the same gradient
isolation property: assignment probabilities are computed from detached logits plus
expert_bias, so only expert_bias receives gradients from the loss signal. The routing
projection weights are not reachable from any returned loss.

The factory is the intended entry point. The caller (MoSRAHRouter) constructs the
loss callable once at init and invokes it each forward pass.

Log-probability formulations (ce, bce) are preferred over linear ones (gshard) because
their gradient magnitude scales with how far the distribution deviates from the target.
A linear signal can be outrun by routing concentrations that diverge nonlinearly; a
log-probability signal cannot.

The external contract for all returned callables is:

    loss_fn(routing_freqs, assignment_probs) -> scalar Tensor

    routing_freqs:    (L,) realized routing frequencies f_i, detached.
    assignment_probs: (L,) soft assignment probabilities p_i with gradient through
                      expert_bias. Caller must compute these via
                      softmax(logits.detach() + expert_bias) to preserve isolation.
"""

import torch
from typing import Callable


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def gshard_loss(
    routing_freqs: torch.Tensor,
    assignment_probs: torch.Tensor,
) -> torch.Tensor:
    """GShard-style linear load-balance loss.

    Computes (1/L) * Σ_i f_i * p_i, where L is the number of expert heads,
    f_i is the realized routing frequency for head i, and p_i is the soft
    assignment probability for head i.

    The fixed point of this loss under gradient descent is uniform routing:
    when p_i = 1/L for all i, the loss is minimized at 1/L (independent of f_i).
    The linear signal is the weakest of the three formulations — gradient magnitude
    does not grow with deviation from the target. Provided for comparison.

    Args:
        routing_freqs: Realized routing frequencies f_i, shape (L,). Detached.
        assignment_probs: Soft assignment probabilities p_i, shape (L,). Gradient
            flows to expert_bias through this tensor.

    Returns:
        Scalar loss tensor.
    """
    L = routing_freqs.shape[0]
    return (routing_freqs * assignment_probs).sum() / L


def ce_loss(
    routing_freqs: torch.Tensor,
    assignment_probs: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy load-balance loss.

    Computes -(1/(L-1)) * Σ_i (1 - f_i) * log(p_i), where the weight (1 - f_i)
    suppresses the signal for overloaded heads (high f_i → weight near zero) and
    amplifies it for underloaded heads (low f_i → weight near 1). This makes the
    loss push probability mass toward under-utilized experts.

    The (1/(L-1)) normalization makes the coefficient interpretable as a controller
    strength independent of expert count. The log-probability signal grows as p_i
    deviates from the target, providing correction that scales with violation severity.

    Args:
        routing_freqs: Realized routing frequencies f_i, shape (L,). Detached.
        assignment_probs: Soft assignment probabilities p_i, shape (L,). Gradient
            flows to expert_bias through this tensor.

    Returns:
        Scalar loss tensor.
    """
    L = routing_freqs.shape[0]
    # Numerical stability: torch.log is safe here because softmax outputs are
    # strictly positive. The (1 - f_i) weight goes to zero exactly when f_i = 1,
    # which can only occur with a single head, so the 0 * (-inf) degenerate case
    # does not arise in practice.
    return -(((1.0 - routing_freqs) * torch.log(assignment_probs)).sum()) / (L - 1)


def bce_loss(
    routing_freqs: torch.Tensor,
    assignment_probs: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy load-balance loss.

    Computes -(1/L) * Σ_i [(1 - f_i) * log(p_i) + f_i * log(1 - p_i)], where
    each head is treated as an independent binary target. Unlike CE, BCE maintains
    a repulsion signal from saturated experts: when f_i → 1, the weight on
    log(1 - p_i) drives p_i away from 1, preventing runaway concentration.

    log(1 - p_i) is computed as log1p(-p_i) for numerical safety near p_i = 1.

    Args:
        routing_freqs: Realized routing frequencies f_i, shape (L,). Detached.
        assignment_probs: Soft assignment probabilities p_i, shape (L,). Gradient
            flows to expert_bias through this tensor.

    Returns:
        Scalar loss tensor.
    """
    L = routing_freqs.shape[0]
    positive_term = (1.0 - routing_freqs) * torch.log(assignment_probs)
    # log1p(-p) instead of log(1-p): avoids catastrophic cancellation when p is
    # close to 1, where (1 - p) loses precision and log produces large errors.
    negative_term = routing_freqs * torch.log1p(-assignment_probs)
    return -(positive_term + negative_term).sum() / L


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_LOSS_REGISTRY: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "gshard": gshard_loss,
    "ce": ce_loss,
    "bce": bce_loss,
}


def make_load_balance_loss(
    loss_type: str,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a load-balance loss callable for the requested formulation.

    All returned callables share the same external contract:

        loss_fn(routing_freqs: Tensor, assignment_probs: Tensor) -> scalar Tensor

    The caller is responsible for computing assignment_probs via
    softmax(logits.detach() + expert_bias) to ensure gradient isolation.

    Args:
        loss_type: One of ``"gshard"``, ``"ce"``, or ``"bce"``.

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
    return _LOSS_REGISTRY[loss_type]
