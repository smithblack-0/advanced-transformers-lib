"""Token-choice router for the MoSRAH sparse attention path.

This module implements the routing mechanism described in Appendix A.Routing of the
paper. Given an input hidden state x, the router produces two outputs used downstream:

  - selected_heads (I): which K of the L available expert heads each token routes to,
    determined by TopK over biased routing scores.
  - routing_probs (P): the weights used for the weighted output reduction, gathered from
    *unbiased* routing scores at the selected indices and renormalized. The learned expert
    bias b must not influence P.

This separation is architecturally critical: expert_bias drives selection (and thus load
balancing) but does not corrupt the gradient path from the output through routing_probs
back to the routing projection weights.

The router also computes and returns the load balance loss via the LoadBalanceLoss custom
autograd operator (see load_balance_loss.py). This loss is a scalar that the training
loop can weight and add to the language modeling loss.

The router additionally computes and returns MaxVio, a detached scalar summarising
routing imbalance for the current forward pass:

    MaxVio = L · max_l(f_l − 1/L)

where f_l is the realised routing frequency of head l and 1/L is the perfectly balanced
target. MaxVio is a monitoring quantity only; it never contributes gradients.

Paper ref: Appendix A.Routing, Appendix A.Load Balancing, §MaxVio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shram.model.configuration import ShramConfig
from src.shram.model.attention.load_balance_loss import LoadBalanceLoss


class MoSRAHRouter(nn.Module):
    """Token-choice router for MoSRAH sparse attention.

    Each input token independently selects K of the L available expert heads. Selection
    is driven by biased routing scores to enable load balancing, but the routing
    probabilities used for output reduction are computed from unbiased scores so that
    the expert bias does not interfere with the gradient path to the router weights.

    The routing projection W_r has no bias term — the paper specifies xW_r with no
    additional projection bias. The only bias-like parameter is expert_bias (b), which
    has an entirely separate role and update mechanism.

    Args:
        config: Model configuration. Must expose ``hidden_size``, ``num_mosrah_heads``
            (L), and ``num_selected_heads`` (K).
    """

    def __init__(self, config: ShramConfig) -> None:
        super().__init__()
        self.num_mosrah_heads = config.num_mosrah_heads
        self.num_selected_heads = config.num_selected_heads

        # W_r: routing projection, no bias (paper specifies xW_r, no additional term).
        self.routing_projection = nn.Linear(
            config.hidden_size, config.num_mosrah_heads, bias=False
        )

        # b: learned per-head bias for load balancing. Initialized to zero so that all
        # heads start with equal selection probability. Updated by the main optimizer
        # via the LoadBalanceLoss custom backward.
        self.expert_bias = nn.Parameter(torch.zeros(config.num_mosrah_heads))

    def forward(
        self, x: torch.Tensor, active_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route input tokens to K expert heads each and compute routing probabilities.

        Args:
            x: Input hidden states of shape (batch, seq_len, hidden_size).
            active_mask: Current-chunk active mask of shape (batch, seq_len), where
                True means the token is semantically live. Dead tokens do not
                contribute to routing frequencies, load_balance_loss, or max_vio.

        Returns:
            selected_heads: Head indices I of shape (batch, seq_len, num_selected_heads).
                Each token's K selected head indices, determined by TopK on biased scores.
            routing_probs: Routing probabilities P of shape (batch, seq_len,
                num_selected_heads). Gathered from unbiased scores at selected_heads
                indices and renormalized to sum to 1 per token.
            load_balance_loss: Scalar load balance imbalance loss for this forward pass.
                Training loop scales this by a weight and adds it to the main loss.
            max_vio: Detached scalar routing-imbalance summary for this forward pass.
                Equal to L · max_l(f_l − 1/L). Zero means perfect balance. Not a loss;
                never contributes gradients.
        """
        B, N, _ = x.shape
        L = self.num_mosrah_heads
        K = self.num_selected_heads

        # Unbiased routing scores R = Softmax(xW_r). These are the scores used to
        # compute routing_probs — expert_bias must not influence them.
        logits = self.routing_projection(x)                    # (B, N, L)
        routing_scores = F.softmax(logits, dim=-1)             # R, (B, N, L)

        # Biased routing scores R̂ = Softmax(xW_r + b). Used only for TopK head
        # selection. expert_bias is added to logits before softmax so that the bias
        # shifts selection probability without rescaling the unbiased distribution.
        biased_routing_scores = F.softmax(                     # R̂, (B, N, L)
            logits + self.expert_bias, dim=-1
        )

        # selected_heads I = TopK(R̂): K head indices per token, shape (B, N, K).
        selected_heads = biased_routing_scores.topk(K, dim=-1).indices

        # Routing probabilities P: gathered from unbiased R at selected_heads indices,
        # then renormalized so they sum to 1 per token. Gathering from routing_scores
        # (not biased_routing_scores) is the invariant that keeps the gradient path from
        # the output back to the router weights free of expert_bias influence.
        gathered = routing_scores.gather(dim=-1, index=selected_heads)   # V, (B, N, K)
        routing_probs = gathered / gathered.sum(dim=-1, keepdim=True)    # P, (B, N, K)

        # Routing frequency f_l: fraction of active (batch, token, head_slot) triples
        # assigned to each head. Dead tokens are excluded by zeroing their rows in the
        # assignment mask before reduction. Normalization uses the active assignment
        # count so frequencies remain properly scaled regardless of how many tokens
        # are live in this chunk.
        assignment_mask = torch.zeros(B, N, L, device=x.device, dtype=x.dtype)
        assignment_mask.scatter_(-1, selected_heads, 1.0)
        active_assignments = assignment_mask * active_mask.unsqueeze(-1)
        num_active_assignments = active_mask.sum() * K
        routing_freqs = active_assignments.sum(dim=(0, 1)) / num_active_assignments  # f, (L,)

        # Load balance loss via custom autograd. expert_bias is an input so PyTorch
        # registers it as a graph node; the custom backward writes the DeepSeek-style
        # correction gradient to expert_bias.grad for the optimizer to consume.
        load_balance_loss = LoadBalanceLoss.apply(self.expert_bias, routing_freqs)

        # MaxVio is a detached monitoring scalar derived from routing_freqs. It must
        # not contribute gradients under any circumstance, so it is detached at the
        # point of computation rather than left to callers to detach.
        max_vio = self._compute_max_vio(routing_freqs, L)

        return selected_heads, routing_probs, load_balance_loss, max_vio

    @staticmethod
    def _compute_max_vio(routing_freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Compute the MaxVio routing-imbalance scalar.

        MaxVio = L · max_l(f_l − 1/L), where f_l is the realised routing frequency of
        head l and 1/L is the perfectly balanced target. A value of zero indicates
        perfect balance; a value of 1 means the most overloaded head received exactly
        double its fair share.

        The result is detached from the autograd graph — MaxVio is a monitoring scalar
        and must never contribute gradients to any parameter.

        Args:
            routing_freqs: Per-head routing frequencies of shape (L,). Sums to 1.
            num_heads: Total number of MoSRAH heads L.

        Returns:
            Detached scalar MaxVio tensor.
        """
        return (num_heads * (routing_freqs - 1.0 / num_heads).max()).detach()
