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

from ..configuration import ShramConfig
from .load_balance_loss import LoadBalanceLoss

from typing import Optional

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
        self.load_balance_p = config.load_balance_p
        if config.use_cache:
            self.capacity = config.mosrah_cache_length
        else:
            self.capacity = config.mosrah_packed_length

        # W_r: routing projection, no bias (paper specifies xW_r, no additional term).
        self.routing_projection = nn.Linear(
            config.embedding_width, config.num_mosrah_heads, bias=False
        )

        # b: learned per-head bias for load balancing. Initialized to zero so that all
        # heads start with equal selection probability. Updated by the main optimizer
        # via the LoadBalanceLoss custom backward.
        self.expert_bias = nn.Parameter(torch.zeros(config.num_mosrah_heads))

    @staticmethod
    def balance_capacity(logits: torch.Tensor,
                         used_capacity: torch.Tensor | None,
                         capacity: int,
                         )->torch.Tensor:
        """
        Balances capacity limits so that if choosing an
        expert would go over capacity, the expert is simply
        not chosen instead
        :param logits: The logits to balance. (B, N, L)
        :param used_capacity: The used capacity, if it exists. (B, L)
        :param capacity: The maximum available capacity. Int.
        :return: Modified logits.
        """

        if used_capacity is None:
            # Presume we are in training mode.

            # Looking up capacity limits only
            # matters if it is, in fact, possible
            # to exceed capacity limits.
            if logits.shape[-2] < capacity:
                return logits

            # Look up the kthvalue and use that as
            # the threshold to mask when below.
            # Note we negate then negate again to sort
            # in ascending order.
            response = torch.kthvalue(-logits, capacity, dim=-2)
            threshold = -response.values
            threshold = threshold.unsqueeze(-2) #(B, 1, L)
        else:
            # We are operating in inference mode.
            # We have to use padding to accomodate the
            # response physically not being long enough
            # to reach capacity

            # Note that padding at zero and shifting
            # the indexes prevents dereferencing a symint,
            # as a version that just patted at 0, 1 and set to
            # length + 1 would do. This prevents a graph break.
            remaining_capacity = capacity - used_capacity # 0 means all used, can be at most capacity
            response_length = logits.shape[-2]
            index = torch.clamp(remaining_capacity, 0, response_length+1)

            # Sort, and add padding. Anything asking for a sequence position
            # outside the current sequence will get a threshold of -1e8; always include
            # If we are asking for a value at zero, get 1e8, or full and we include
            # nothing.
            ordered_logits = torch.sort(logits, dim=-2, descending=True).values
            ordered_logits = F.pad(ordered_logits, (0,0, 1, 0), value=1e8)
            ordered_logits = F.pad(ordered_logits, (0, 0, 0, 1), value=-1e8)

            threshold = ordered_logits.gather(-2, index.unsqueeze(-2)) #(B, 1, L)

        mask = threshold > logits
        logits = logits.masked_fill(mask, -1e8)
        return logits
    def forward(
        self,
        x: torch.Tensor,
        active_mask: torch.Tensor,
        used_capacity: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route input tokens to K expert heads each and compute routing probabilities.

        Args:
            x: Input hidden states of shape (batch, seq_len, hidden_size).
            active_mask: Current-chunk active mask of shape (batch, seq_len), where
                True means the token is semantically live. Dead tokens do not
                contribute to routing frequencies, load_balance_loss, or max_vio.
            used_capacity: Used for capacity management during inference, missing during training.
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
        biased_logits = logits + self.expert_bias
        biased_logits = self.balance_capacity(biased_logits, used_capacity, self.capacity)
        biased_routing_scores = F.softmax(                     # R̂, (B, N, L)
           biased_logits, dim=-1
        )

        # selected_heads I = TopK(R̂): K head indices per token, shape (B, N, K).
        # and routing logits directly
        selected_heads = biased_routing_scores.topk(K, dim=-1).indices
        gathered = routing_scores.gather(dim=-1, index=selected_heads)   # V, (B, N, K)

        # Routing probabilities P: gathered from unbiased R at selected_heads indices,
        # then renormalized so they sum to 1 per token. Gathering from routing_scores
        # (not biased_routing_scores) is the invariant that keeps the gradient path from
        # the output back to the router weights free of expert_bias influence.
        routing_probs = gathered / gathered.sum(dim=-1, keepdim=True)    # P, (B, N, K)

        # Per-item routing frequencies f_{b,l}: for each batch item b and head l, what
        # fraction of that item's active K assignments over all tokens go to head l.
        # Dead tokens are excluded before reduction. Normalization is per batch item so
        # each item's frequencies sum to 1 independently of other items in the batch.
        assignment_mask = torch.zeros(B, N, L, device=x.device, dtype=x.dtype)
        assignment_mask.scatter_(-1, selected_heads, 1.0)
        active_assignments = assignment_mask * active_mask.unsqueeze(-1)
        per_item_counts = active_assignments.sum(dim=1)             # (B, L)
        per_item_total = active_mask.sum(dim=1, keepdim=True) * K   # (B, 1)
        per_item_freqs = per_item_counts / per_item_total            # (B, L)

        # p-mean of per_item_freqs over the batch dimension produces routing_freqs (L,).
        # p-mean weights aggregation toward the worst-case batch item relative to
        # arithmetic mean, making the load balance signal sensitive to per-item spikes
        # that cause packing overflow.
        p = self.load_balance_p
        routing_freqs = (per_item_freqs ** p).mean(dim=0) ** (1.0 / p)  # (L,)

        # Load balance loss via custom autograd. expert_bias is an input so PyTorch
        # registers it as a graph node; the custom backward writes the DeepSeek-style
        # correction gradient to expert_bias.grad for the optimizer to consume.
        load_balance_loss = LoadBalanceLoss.apply(self.expert_bias, routing_freqs)

        # MaxVio is a detached monitoring scalar following the paper's formula
        # L · max_l(f_l − 1/L) applied to routing_freqs. Must not contribute gradients.
        max_vio = self._compute_max_vio(routing_freqs, L)

        return selected_heads, routing_probs, load_balance_loss, max_vio

    @staticmethod
    def _compute_max_vio(routing_freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Compute the MaxVio routing-imbalance scalar.

        MaxVio = L · max_l(f_l − 1/L), where f_l is the realised routing frequency of
        head l and 1/L is the perfectly balanced target. Follows the paper's definition
        (Wang et al.) applied to routing_freqs. A value of zero indicates perfect
        balance; a value of 0.5 means the most overloaded head received 50% more routed
        tokens than ideal.

        The result is detached from the autograd graph — MaxVio is a monitoring scalar
        and must never contribute gradients to any parameter.

        Args:
            routing_freqs: Per-head routing frequencies of shape (L,).
            num_heads: Total number of MoSRAH heads L.

        Returns:
            Detached scalar MaxVio tensor.
        """
        return (num_heads * (routing_freqs - 1.0 / num_heads).max()).detach()
