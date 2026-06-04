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

        self.max_bid_rounds = config.max_bid_rounds

        # W_r: routing projection, no bias (paper specifies xW_r, no additional term).
        self.routing_projection = nn.Linear(
            config.embedding_width, config.num_mosrah_heads, bias=False
        )

        # b: learned per-head bias for load balancing. Initialized to zero so that all
        # heads start with equal selection probability. Updated by the main optimizer
        # via the LoadBalanceLoss custom backward.
        self.expert_bias = nn.Parameter(torch.zeros(config.num_mosrah_heads))

    @staticmethod
    def get_best_proposals(
            tensor: torch.Tensor,
            dim: int,
            n: int | torch.Tensor,
            capacity_scalar: int,
    ) -> torch.Tensor:
        """Return a boolean mask selecting the top-n entries along dim.

        Uses topk to select exactly min(n_per_slice, dim_length) True entries
        per slice along dim. Unlike a threshold comparison, this never
        over-selects under tied logit values, which occurs when padding tokens
        contribute identical scores to multiple expert slots.

        Args:
            tensor: Input tensor. Higher values rank first.
            dim: Dimension to select along.
            n: Per-slice selection count. Scalar int or tensor broadcastable
               to tensor with dim removed. Slices where n=0 produce all-False
               outputs.
            capacity_scalar: Static upper bound on n; used to derive topk k as
               min(tensor.shape[dim], capacity_scalar). Must be a Python int
           for compile compatibility.

        Returns:
            Boolean mask of the same shape as tensor.
        """
        positive_dim = dim % tensor.ndim
        dim_length = tensor.shape[positive_dim]
        k = min(dim_length, capacity_scalar)

        topk_indices = tensor.topk(k, dim=dim).indices

        # Rank tensor broadcast-compatible with topk_indices: rank r along dim
        # corresponds to the (r+1)-th highest value in that slice.
        rank_shape = [1] * tensor.ndim
        rank_shape[positive_dim] = k
        ranks = torch.arange(k, device=tensor.device, dtype=torch.long).view(rank_shape)

        # element_included: True where this rank falls within the per-slice budget.
        # For scalar n all k ranks satisfy rank < n (since k = min(dim_length, n)).
        # For tensor n per-slice budgets differ; rank >= n[slice] yields False,
        # correctly excluding excess slots including those with n=0.
        if isinstance(n, int):
            element_included = ranks < n
        else:
            element_included = ranks < n.unsqueeze(positive_dim)

        # Allocate from explicit logical shape rather than using zeros_like. This keeps
        # the output mask tied to tensor.shape, not to any stride/layout metadata carried
        # by tensor from earlier view operations or compiler lowering.
        mask = torch.zeros(
            tuple(tensor.shape),
            device=tensor.device,
            dtype=torch.bool,
        )

        # Materialize the scatter source shape explicitly. This avoids passing a
        # broadcast-view source into scatter while preserving the same logical rule:
        # every selected top-k index receives True iff its rank is within budget.
        scatter_values = torch.broadcast_to(element_included, topk_indices.shape)
        mask = mask.scatter(dim, topk_indices, scatter_values)
        return mask

    @staticmethod
    def _check_bidding_converged(acceptances: torch.Tensor,
                                 min_choices: int,
                                 max_rounds: int) -> None:
        """Raise if the bidding loop exhausted max_rounds without satisfying all tokens.

        Args:
            acceptances: bool tensor of shape (B, N, L) indicating what experts L accepted
                what tokens.
            min_choices: Convergence has been reached if acceptances are such that a sum along
                N always has at least min_choices choices.
            max_rounds: The iteration ceiling that was applied, for the error message. Used
                for reporting
        """
        msg = (
            f"balance_capacity bidding did not converge within {max_rounds} rounds. "
            f"Increase mosrah_overallocation_factor or max_bid_rounds."
        )
        converged = (acceptances.sum(dim=-1) >= min_choices).all()
        torch._assert_async(converged, msg)

    @classmethod
    def _run_bidding(
            cls,
            logits: torch.Tensor,
            remaining_capacity: int | torch.Tensor,
            min_choices: int,
            max_rounds: int,
            capacity_scalar: int,
    ) -> torch.Tensor:
        """Deferred-acceptance (Gale-Shapley) bidding solver for joint capacity enforcement.

        Tokens propose experts in descending preference order; experts provisionally
        accept their top-``remaining_capacity`` proposed tokens each round. Proposals
        are monotone (never retracted), so once all tokens are satisfied, subsequent
        iterations are no-ops. Runs unconditionally for exactly ``max_rounds`` iterations
        to keep the compiled graph flat and free of data-dependent control flow.

        Both the column bound (per-expert token count ≤ remaining_capacity) and the
        row bound (per-token expert count ≥ min_choices) are satisfied simultaneously
        on the returned mask by construction.

        Args:
            logits: Routing scores of shape (B, N, L).
            remaining_capacity: Per-expert token budget. Scalar int for training;
                (B, L) tensor for inference.
            min_choices: Minimum experts each token must have accepted (K).
            max_rounds: Number of iterations to run. Convergence is checked after
                all rounds via ``_check_bidding_converged``; raises if not met.
            capacity_scalar: Static upper bound on remaining_capacity, passed to
                ``get_mask`` as the topk k bound for the acceptance step.

        Returns:
            accepted: (B, N, L) bool — True at positions accepted by the solver.
        """
        proposals   = torch.zeros_like(logits, dtype=torch.bool)
        acceptances = torch.zeros_like(logits, dtype=torch.bool)

        for _ in range(max_rounds):
            # ── token proposal step ───────────────────────────────────────────
            #
            # Tokens with fewer than min_choices accepted experts propose their
            # next-best unproposed expert(s). The deficit determines how many new
            # proposals each token makes; satisfied tokens propose nothing
            # (deficit = 0 → get_mask returns all-False). Proposals are monotone:
            # once all tokens are satisfied, subsequent iterations are no-ops.
            accepted_per_token = acceptances.sum(dim=-1)           # (B, N)
            choices_deficit = (min_choices - accepted_per_token).clamp_min(0)

            unproposed_logits = logits.masked_fill(proposals, float('-inf'))
            new_proposals = cls.get_best_proposals(
                unproposed_logits, dim=-1, n=choices_deficit, capacity_scalar=min_choices,
            )
            proposals = proposals | new_proposals

            # ── expert acceptance step ────────────────────────────────────────
            #
            # Each expert accepts its top-remaining_capacity proposed tokens.
            # Acceptances are recomputed from scratch each round so that a
            # stronger new proposal can displace a weaker prior one.
            proposed_logits = logits.masked_fill(~proposals, float('-inf'))
            acceptances = cls.get_best_proposals(
                proposed_logits, dim=-2, n=remaining_capacity, capacity_scalar=capacity_scalar,
            )

        return acceptances

    @classmethod
    def balance_capacity(
            cls,
            logits: torch.Tensor,
            used_capacity: torch.Tensor | None,
            capacity: int,
            min_choices: int,
            max_rounds: int,
            mask_value: float = -1e8,
    ) -> torch.Tensor:
        """Mask logits so both capacity constraints hold simultaneously on the output.

        Two constraints must hold:
          - Column bound: per-expert unmasked token count ≤ remaining_capacity.
          - Row bound:    per-token unmasked expert count ≥ min_choices.

        A training fast path is attempted before the bidding solver:

        1. Training with N ≤ capacity: return logits unchanged.
        2. Bidding: deferred-acceptance solver guaranteeing both bounds simultaneously.

        Args:
            logits: Routing scores of shape (B, N, L).
            used_capacity: Tokens already accumulated per expert, shape (B, L).
                ``None`` during training (full capacity available).
            capacity: Maximum tokens per expert (from config).
            min_choices: Minimum experts each token must retain (K).
            max_rounds: Bidding iteration ceiling (from config.max_bid_rounds).
            mask_value: Value written to masked positions. Default -1e8.

        Returns:
            Logits with unavailable positions set to ``mask_value``, shape (B, N, L).
        """
        # ── Algorithm overview ────────────────────────────────────────────────
        #
        # Problem: mask (B, N, L) logits so that both the column bound (each
        # expert receives at most remaining_capacity tokens) and the row bound
        # (each token retains at least min_choices expert choices) hold
        # simultaneously. Satisfying either constraint greedily can violate the
        # other, requiring a joint solver for the hard case.
        #
        # Approach: deferred-acceptance (Gale-Shapley) bidding. Each round,
        # tokens that still lack min_choices accepted experts propose their
        # next-best unproposed expert. Each expert then provisionally accepts its
        # top-remaining_capacity proposed tokens, potentially displacing weaker
        # prior acceptances. Proposals are monotone (never retracted). The loop
        # terminates when every token has min_choices accepted experts or
        # max_bid_rounds is exhausted (RuntimeError in the latter case).
        #
        # Training fast path — when N ≤ capacity and all experts start empty,
        # no expert can overflow regardless of routing. No masking is needed.

        # Training fast path: N ≤ capacity with empty experts → no overflow possible.
        if used_capacity is None and logits.shape[-2] <= capacity:
            return logits

        # Compute per-expert remaining budget.
        # Training (N > capacity path): scalar — all experts start with full capacity.
        # Inference: subtract already-accumulated tokens; clamp prevents negatives
        #            when rounding causes used_capacity to slightly exceed capacity.
        if used_capacity is None:
            remaining_capacity = capacity
        else:
            remaining_capacity = (capacity - used_capacity).clamp(min=0)  # (B, L)

        # Bidding solver: jointly satisfies column and row bounds. Runs under
        # no_grad because the boolean mask is a hard routing decision and must
        # not accumulate gradient memory.
        with torch.no_grad():
            final_mask = cls._run_bidding(logits, remaining_capacity,
                                          min_choices, max_rounds, capacity)
            cls._check_bidding_converged(final_mask, min_choices, max_rounds)
        return logits.masked_fill(~final_mask, mask_value)

    def forward(
        self,
        x: torch.Tensor,
        active_mask: torch.Tensor,
        used_capacity: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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
            router_diagnostics: Dict of routing feedback scalars. Keys:
                - ``load_balance_loss``: scalar load-balance loss with gradient.
                - ``max_vio``: detached scalar routing-imbalance summary.
                - ``bias_std``: std of expert_bias; near-zero means corrections have not built up.
                - ``raw_logit_std``: mean per-token std of unbiased logits; the natural routing scale.
                - ``logit_std``: mean per-token std of (logits + expert_bias); lower than
                  raw_logit_std means bias is flattening preferences (healthy correction).
                - ``bias_alignment``: mean cosine similarity of expert_bias against per-token
                  logits. Negative means bias opposes routing direction (healthy correction);
                  positive means runaway reinforcement.
        """
        B, N, _ = x.shape
        L = self.num_mosrah_heads
        K = self.num_selected_heads

        # Unbiased routing scores R = Softmax(xW_r). These are the scores used to
        # compute routing_probs — expert_bias must not influence them.
        logits = self.routing_projection(x)                    # (B, N, L)

        # Diagnostic scalars characterising the load-balance mechanism. Must be
        # computed here — before balance_capacity injects -1e8 sentinels that
        # would corrupt std and cosine similarity.
        bias_std = self.expert_bias.std().detach()
        raw_logit_std = logits.std(dim=-1).mean().detach()
        logit_std = (logits + self.expert_bias).std(dim=-1).mean().detach()
        bias_alignment = F.cosine_similarity(
            logits, self.expert_bias.expand_as(logits), dim=-1
        ).mean().detach()

        routing_scores = F.softmax(logits, dim=-1)             # R, (B, N, L)

        # Biased routing scores R̂ = Softmax(xW_r + b). Used only for TopK head
        # selection. expert_bias is added to logits before softmax so that the bias
        # shifts selection probability without rescaling the unbiased distribution.
        biased_logits = logits + self.expert_bias
        biased_logits = self.balance_capacity(
            biased_logits,
            used_capacity,
            self.capacity,
            self.num_selected_heads,
            self.max_bid_rounds,
        )
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

        router_diagnostics = {
            "load_balance_loss": load_balance_loss,
            "max_vio": max_vio,
            "bias_std": bias_std,
            "raw_logit_std": raw_logit_std,
            "logit_std": logit_std,
            "bias_alignment": bias_alignment,
        }
        return selected_heads, routing_probs, router_diagnostics

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
