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
    def get_threshold(
            tensor: torch.Tensor,
            dim: int,
            n: int | torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the n-th largest value along dim, keepdim=True.

        A value >= threshold ranks within the top n along dim. Boundary cases
        follow the monotone descending contract:

            n == 0         ->  +inf   nothing qualifies
            n > dim_length ->  -inf   everything qualifies

        :param tensor: Floating-point input, no NaN.
        :param dim:    Dimension to reduce along.
        :param n:      1-indexed rank. Scalar int or tensor of ints broadcastable
                       to tensor with dim removed.
        :return:       Threshold with size 1 along dim, same dtype/device.
        """
        # -------------------------------------------------------------------------
        # Algorithm overview
        # -------------------------------------------------------------------------
        #
        # Scalar n does not need a full sorted table. kthvalue selects the n-th
        # rank directly, and the two boundary sentinels are returned explicitly.
        #
        # Tensor n requires a full sorted table because each position along the
        # complementary dimensions may request a different rank. The table is
        # built once by sorting descending, then sentinel values are padded at
        # both ends so that boundary n values resolve correctly via gather:
        #
        #     index 0            <- +inf sentinel  (n == 0)
        #     index 1..dim_length <- sorted values  (valid ranks, 1-indexed)
        #     index dim_length+1  <- -inf sentinel  (n > dim_length)
        #
        # The critical invariant is that n is 1-indexed. This means valid ranks
        # map directly to their gather index without any offset, and index 0 is
        # naturally free for the +inf sentinel. n == 0 gathers +inf without
        # special-casing, and overflow n gathers -inf after clamping.
        #
        # F.pad specifies padding from the last dimension inward. Targeting an
        # arbitrary dim requires a positive index to compute how many trailing
        # dimensions to skip over in the pad spec.
        positive_dim = dim % tensor.ndim
        dim_length = tensor.shape[positive_dim]

        if isinstance(n, int):
            # Scalar rank selection does not need a full sorted table. kthvalue
            # finds the k-th smallest; negating input and output flips the order
            # to give the k-th largest. Boundary sentinels follow the descending
            # contract: +inf sits above every real value (nothing qualifies),
            # -inf sits below every real value (everything qualifies).
            if n == 0:
                shape = list(tensor.shape)
                shape[positive_dim] = 1
                return tensor.new_full(shape, float('inf'))
            if n > dim_length:
                shape = list(tensor.shape)
                shape[positive_dim] = 1
                return tensor.new_full(shape, float('-inf'))
            return -torch.kthvalue(-tensor, n, dim=dim, keepdim=True).values

        else:
            # Build the rank table once; each position gathers its own threshold.
            sorted_desc = torch.sort(tensor, dim=dim, descending=True).values

            # Each trailing dimension after positive_dim contributes one (left,
            # right) zero-pair before the target padding entry in the F.pad spec.
            num_padding_skips = 2 * (tensor.ndim - positive_dim - 1)
            leading_pad = [0] * num_padding_skips + [1, 0]
            trailing_pad = [0] * num_padding_skips + [0, 1]

            sorted_desc = F.pad(sorted_desc, leading_pad, value=float('inf'))
            sorted_desc = F.pad(sorted_desc, trailing_pad, value=float('-inf'))

            # unsqueeze restores the reduced dimension so gather sees the same
            # rank as the padded table along dim.
            gather_index = n.clamp(0, dim_length + 1).long().unsqueeze(dim)
            return sorted_desc.gather(dim, gather_index)
    @staticmethod
    def _check_bidding_converged(converged: torch.Tensor, max_rounds: int) -> None:
        """Raise if the bidding loop exhausted max_rounds without satisfying all tokens.

        In compiled mode ``torch._check`` fires a C++ assertion
        (``capture_scalar_outputs=True`` is a precondition — see Unit 19.F.1).
        In eager mode raises ``RuntimeError`` directly.

        Exhausting ``max_rounds`` indicates an extreme routing density case or an
        infeasible configuration where total capacity is insufficient for N * K
        demands. In normal training this should never occur; the default
        ``max_bid_rounds=10`` covers approximately the 98th percentile of routing
        densities.

        Args:
            converged: Scalar bool tensor — True if all tokens have >= K accepted experts.
            max_rounds: The iteration ceiling that was applied, for the error message.
        """
        if torch.compiler.is_compiling():
            torch._check(converged)
        else:
            if not converged.item():
                raise RuntimeError(
                    f"balance_capacity bidding did not converge within {max_rounds} rounds. "
                    f"All tokens must have at least K accepted experts before the loop exits. "
                    f"This indicates either an infeasible configuration (total remaining "
                    f"capacity < N * K) or an extreme routing density. "
                    f"Increase mosrah_overallocation_factor or max_bid_rounds."
                )

    @staticmethod
    def _run_bidding(
            logits: torch.Tensor,
            remaining_capacity: int | torch.Tensor,
            min_choices: int,
            max_rounds: int,
    ) -> torch.Tensor:
        """Deferred-acceptance (Gale-Shapley) bidding solver for joint capacity enforcement.

        Tokens propose experts in descending preference order; experts provisionally
        accept their top-``remaining_capacity`` proposed tokens each round. Proposals
        are monotone (never retracted). The loop continues until every token has at
        least ``min_choices`` accepted experts or ``max_rounds`` is exhausted.

        Both the column bound (per-expert token count ≤ remaining_capacity) and the
        row bound (per-token expert count ≥ min_choices) are satisfied simultaneously
        on the returned mask by construction.

        Args:
            logits: Routing scores of shape (B, N, L).
            remaining_capacity: Per-expert token budget. Scalar int for training;
                (B, L) tensor for inference.
            min_choices: Minimum experts each token must have accepted (K).
            max_rounds: Iteration ceiling; raises via ``_check_bidding_converged``
                if exhausted.

        Returns:
            accepted: (B, N, L) bool — True at positions accepted by the solver.
        """
        # ── initialise loop variables ─────────────────────────────────────────
        #
        # All three loop_vars must be tensors of fixed shape across iterations,
        # as required by torch.while_loop. logits and remaining_capacity are
        # captured read-only by the closures; they do not travel as loop_vars.
        proposals  = torch.zeros_like(logits, dtype=torch.bool)
        acceptances = torch.zeros_like(logits, dtype=torch.bool)
        round_count = torch.zeros((), device=logits.device, dtype=torch.int64)
        max_rounds_t = torch.full((), max_rounds, device=logits.device, dtype=torch.int64)

        def cond_fn(proposals, acceptances, round_count):
            all_satisfied = (acceptances.sum(dim=-1) >= min_choices).all()
            return (round_count < max_rounds_t) & ~all_satisfied

        def body_fn(proposals, acceptances, round_count):
            # ── token proposal step ───────────────────────────────────────────
            #
            # Tokens with fewer than min_choices accepted experts propose their
            # next-best unproposed expert(s). The deficit determines how many new
            # proposals each token makes this round; already-satisfied tokens
            # propose nothing (deficit = 0 → bid_threshold = +inf → no new bids).
            accepted_per_token = acceptances.sum(dim=-1)           # (B, N)
            choices_deficit = (min_choices - accepted_per_token).clamp_min(0)

            unproposed_logits = logits.masked_fill(proposals, float('-inf'))
            bid_threshold = MoSRAHRouter.get_threshold(
                unproposed_logits, dim=-1, n=choices_deficit,
            )
            new_proposals = (
                (unproposed_logits >= bid_threshold)
                & ~proposals
                & (choices_deficit.unsqueeze(-1) > 0)
            )
            updated_proposals = proposals | new_proposals

            # ── expert acceptance step ────────────────────────────────────────
            #
            # Each expert accepts its top-remaining_capacity proposed tokens.
            # Acceptances are recomputed from scratch each round so that a
            # stronger new proposal can displace a weaker prior one.
            proposed_logits = logits.masked_fill(~updated_proposals, float('-inf'))
            accept_threshold = MoSRAHRouter.get_threshold(
                proposed_logits, dim=-2, n=remaining_capacity,
            )
            updated_acceptances = updated_proposals & (proposed_logits >= accept_threshold)

            return updated_proposals, updated_acceptances, round_count + 1

        proposals, acceptances, _ = torch.while_loop(
            cond_fn, body_fn, (proposals, acceptances, round_count),
        )

        converged = (acceptances.sum(dim=-1) >= min_choices).all()
        MoSRAHRouter._check_bidding_converged(converged, max_rounds)
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

        A training fast path and a column-capacity fast path are attempted before
        falling back to the bidding solver:

        1. Training with N ≤ capacity: return logits unchanged.
        2. Column-capacity fast path: if the most permissive column-bound-satisfying
           mask already gives every token at least min_choices choices, return it.
        3. Bidding fallback: deferred-acceptance solver guaranteeing both bounds.

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
        # Two cheaper paths precede the solver:
        #
        #   Training fast path — when N ≤ capacity and all experts start empty,
        #   no expert can overflow regardless of routing. No masking is needed.
        #
        #   Column-capacity fast path — the most permissive mask satisfying the
        #   column bound selects each expert's top-remaining_capacity tokens. If
        #   that mask also satisfies the row bound, both constraints hold and the
        #   solver is skipped entirely.

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

        # Column-capacity fast path: select each expert's top-remaining_capacity
        # tokens — the most permissive mask satisfying the column bound. If it
        # also satisfies the row bound, both constraints hold simultaneously.
        # Mask computation runs under no_grad: the boolean mask is a hard routing
        # decision and must not accumulate gradient memory through the solver.
        with torch.no_grad():
            col_threshold = cls.get_threshold(logits, dim=-2, n=remaining_capacity)
            col_capacity_mask = logits >= col_threshold                # (B, N, L)
        if (col_capacity_mask.sum(dim=-1) >= min_choices).all():
            return logits.masked_fill(~col_capacity_mask, mask_value)

        # Column-capacity mask violates the row bound: routing is concentrated
        # enough that per-expert capacity limits leave some tokens with fewer
        # than min_choices choices. The bidding solver handles this jointly.
        with torch.no_grad():
            accepted = cls._run_bidding(logits, remaining_capacity, min_choices, max_rounds)
        return logits.masked_fill(~accepted, mask_value)
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
